/**
 * The converter.
 *
 * Pipeline (each step is observable through {@link ConvertReport}):
 *
 *   1. decide what must stay fp32 (explicit blocks, overflow strategy)
 *   2. rescale overflowing BatchNorm layers (default) or keep them fp32
 *   3. convert float32 initializers and Constant tensors to float16
 *   4. insert Cast nodes at every fp16 ⇄ fp32 boundary
 *   5. topological sort, validate, encode
 *
 * The result is a *storage* conversion: every float tensor becomes fp16 and the
 * graph runs in fp16. On runtimes without fp16 kernels (onnxruntime-web WASM)
 * that is a download-size win and a speed loss — see README.
 */
import {
  AttrType,
  DataType,
  FP16_MAX,
  type AttributeProto,
  type GraphProto,
  type ModelProto,
  type NodeProto,
  type TensorProto,
} from './onnx.js';
import {
  collectConsumers,
  collectProducers,
  decodeModel,
  encodeModel,
  makeCast,
  readFloat32Tensor,
  renameNodeInputs,
  toNumber as toInt,
  setElemType,
  tensorAbsMax,
  tensorBytes,
  topologicalSort,
  validateGraph,
  writeFloat16Bits,
} from './model.js';
import { DEFAULT_TARGET_MAX, rescaleBatchNorms, type RescaleChange } from './bn.js';
import { float32ToFloat16 } from './fp16.js';

export type OverflowStrategy = 'rescale' | 'block' | 'error';

/**
 * Ops with no float16 kernel in ONNX Runtime (or whose fp16 kernel is known to
 * be missing/broken). They stay in fp32 with Cast boundaries around them.
 *
 * The list mirrors `onnxconverter-common`'s default op block list, trimmed to
 * the ops that are still problematic in current runtimes. `Min`/`Max`/`Mul`
 * style elementwise ops are *not* here: modern ORT and WebGPU run them in fp16.
 */
/**
 * Ops whose outputs are not float16 even when the graph runs in fp16. Used to
 * decide whether a Cast boundary is needed (casting an int64 `Shape` output to
 * float32 would break the graph).
 */
const NON_FLOAT_OUTPUTS: Record<string, number> = {
  Shape: DataType.INT64,
  Size: DataType.INT64,
  ArgMax: DataType.INT64,
  ArgMin: DataType.INT64,
  NonZero: DataType.INT64,
  NonMaxSuppression: DataType.INT64,
  Equal: 9, // bool
  Greater: 9,
  GreaterOrEqual: 9,
  Less: 9,
  LessOrEqual: 9,
  Not: 9,
  And: 9,
  Or: 9,
  Xor: 9,
  IsNaN: 9,
  IsInf: 9,
};

const FLOAT_TYPES = new Set<number>([DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT16, DataType.BFLOAT16]);

/** Dtype produced by a `Constant` node, however the value is stored. */
function constantOutputDtype(node: NodeProto): number | undefined {
  for (const attribute of node.attribute) {
    switch (attribute.name) {
      case 'value':
        if (attribute.t) return attribute.t.dataType;
        break;
      case 'value_float':
      case 'value_floats':
        return DataType.FLOAT;
      case 'value_int':
      case 'value_ints':
        return DataType.INT64;
      case 'value_string':
      case 'value_strings':
        return 8; // string
    }
  }
  return undefined;
}

/**
 * Best-effort element type per tensor *before* conversion.
 *
 * Seeded from initializers, `Constant` payloads, graph inputs and `value_info`,
 * then propagated: `Cast` declares its target, `Shape`/`Size`/… are int64, a
 * node with a float input produces a float, a node whose inputs are all
 * non-float produces a non-float. Everything else stays unknown.
 */
function inferKnownDtypes(graph: GraphProto): Map<string, number> {
  const dtype = new Map<string, number>();
  for (const tensor of graph.initializer) dtype.set(tensor.name, tensor.dataType);
  for (const valueInfo of [...graph.input, ...graph.valueInfo]) {
    const elemType = valueInfo.type?.tensorType?.elemType;
    if (elemType) dtype.set(valueInfo.name, elemType);
  }
  for (const node of graph.node) {
    if (node.opType !== 'Constant') continue;
    const type = constantOutputDtype(node);
    if (type === undefined) continue;
    for (const output of node.output) if (output) dtype.set(output, type);
  }

  // Nodes may appear in any order; a few passes converge in practice.
  for (let pass = 0; pass < 8; pass += 1) {
    let changed = false;
    for (const node of graph.node) {
      if (!node.output.some((output) => output && !dtype.has(output))) continue;
      const inferred = inferNodeDtype(node, dtype);
      if (inferred === undefined) continue;
      for (const output of node.output) {
        if (output && !dtype.has(output)) {
          dtype.set(output, inferred);
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  return dtype;
}

function inferNodeDtype(node: NodeProto, dtype: Map<string, number>): number | undefined {
  if (node.opType === 'Cast') {
    const attribute = node.attribute.find((entry) => entry.name === 'to');
    return attribute ? toInt(attribute.i) : undefined;
  }
  if (node.opType === 'Constant') return constantOutputDtype(node);
  const nonFloat = NON_FLOAT_OUTPUTS[node.opType];
  if (nonFloat !== undefined) return nonFloat;
  if (node.input.length === 0) return undefined;

  let fallback: number | undefined;
  for (const input of node.input) {
    if (!input) continue;
    const type = dtype.get(input);
    if (type === undefined) return undefined;
    if (FLOAT_TYPES.has(type)) return DataType.FLOAT;
    fallback ??= type;
  }
  return fallback;
}

/** True when the node consumes a float tensor (or something we cannot type). */
function hasFloatInput(node: NodeProto, dtype: Map<string, number>): boolean {
  if (node.input.length === 0) return false;
  for (const input of node.input) {
    if (!input) continue;
    const type = dtype.get(input);
    if (type === undefined || FLOAT_TYPES.has(type)) return true;
  }
  return false;
}

export const DEFAULT_FP32_OPS: readonly string[] = [
  'Range',
  'Resize',
  'NonMaxSuppression',
  'RoiAlign',
  'CumSum',
  'Upsample',
  'TopK',
];

export interface ConvertOptions {
  /**
   * Keep graph inputs/outputs in fp32 and Cast inside the graph.
   * Default `true` — callers keep feeding float32 buffers (what ORT's
   * `keep_io_types` does, and what every browser app expects).
   */
  keepIoTypes?: boolean;
  /**
   * What to do with tensors that do not fit in fp16:
   * - `rescale` (default): fix overflowing BatchNorm layers algebraically; any
   *   other overflowing tensor is an error.
   * - `block`: keep the affected nodes in fp32 (Cast boundaries), never silently
   *   lose range.
   * - `error`: abort on the first overflow (use this in CI).
   */
  overflow?: OverflowStrategy;
  /** fp16 representable maximum. */
  limit?: number;
  /** Target max |var| after a BatchNorm rescale. */
  targetMax?: number;
  /**
   * Extra op types that must stay fp32. Merged with {@link DEFAULT_FP32_OPS}
   * unless `defaultBlockOps` is `false`.
   */
  blockOps?: string[];
  /**
   * Apply {@link DEFAULT_FP32_OPS} (Range, Resize, TopK, …). Default `true`.
   * Set to `false` when you know your runtime has fp16 kernels for all of them.
   */
  defaultBlockOps?: boolean;
  /** Node names (exact, `*` wildcard, or `/regex/`) that must stay fp32. */
  blockNodes?: string[];
  /** Convert float tensors attached to `Constant` nodes. Default `true`. */
  convertConstants?: boolean;
  /** Re-sort nodes into topological order after inserting Casts. Default `true`. */
  toposort?: boolean;
  /** Validate the resulting graph structurally. Default `true`. */
  validate?: boolean;
}

export interface ConvertReport {
  inputBytes: number;
  outputBytes: number;
  bytesSaved: number;
  /** `outputBytes / inputBytes` */
  ratio: number;
  convertedTensors: number;
  convertedBytes: number;
  constantsConverted: number;
  /** Pre-existing `Cast(to=float32)` nodes retargeted at float16. */
  retargetedCasts: number;
  keptFloat32: { name: string; reason: string }[];
  insertedCasts: number;
  rescaled: RescaleChange[];
  blockedNodes: string[];
  warnings: string[];
}

export interface ConvertResult {
  model: ModelProto;
  report: ConvertReport;
}

export interface OverflowErrorEntry {
  name: string;
  absMax: number;
  limit: number;
  fixable: boolean;
}

export class OverflowError extends Error {
  readonly tensors: OverflowErrorEntry[];

  constructor(tensors: OverflowErrorEntry[], limit: number) {
    const list = tensors
      .slice(0, 8)
      .map((tensor) => `  ${tensor.name}: max|value| = ${tensor.absMax.toExponential(3)}`)
      .join('\n');
    super(
      `${tensors.length} tensor(s) exceed the fp16 range (${limit}):\n${list}` +
        (tensors.length > 8 ? `\n  … and ${tensors.length - 8} more` : '') +
        `\n\nRun \`onnx-fp16-guard inspect <model>\` for the full report, or use ` +
        `--overflow=rescale (default) / --overflow=block.`,
    );
    this.name = 'OverflowError';
    this.tensors = tensors;
  }
}

function nodeLabel(node: NodeProto, index: number): string {
  return node.name || `${node.opType}#${index}`;
}

/** `*` wildcard and `/regex/` support for `blockNodes`. */
export function matchesPattern(pattern: string, value: string): boolean {
  if (pattern.length > 1 && pattern.startsWith('/') && pattern.endsWith('/')) {
    return new RegExp(pattern.slice(1, -1)).test(value);
  }
  if (pattern.includes('*')) {
    const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
    return new RegExp(`^${escaped}$`).test(value);
  }
  return pattern === value;
}

function floatTensorAttribute(node: NodeProto): AttributeProto | undefined {
  return node.attribute.find(
    (attribute) =>
      attribute.type === AttrType.TENSOR && attribute.t && attribute.t.dataType === DataType.FLOAT,
  );
}

function convertTensor(tensor: TensorProto): number {
  const data = readFloat32Tensor(tensor);
  const bits = float32ToFloat16(data);
  const before = tensorBytes(tensor);
  writeFloat16Bits(tensor, bits);
  return before;
}

/**
 * Convert a decoded model in place and return a report.
 *
 * The model object is mutated for zero-copy performance. Use
 * {@link convertBytes} if you need to keep the original bytes.
 */
export function convertModel(model: ModelProto, options: ConvertOptions = {}): ConvertResult {
  const graph = model.graph;
  const keepIoTypes = options.keepIoTypes ?? true;
  const overflow = options.overflow ?? 'rescale';
  const limit = options.limit ?? FP16_MAX;
  const targetMax = options.targetMax ?? DEFAULT_TARGET_MAX;
  const convertConstants = options.convertConstants ?? true;
  const shouldToposort = options.toposort ?? true;
  const shouldValidate = options.validate ?? true;

  const blocked = new Set<NodeProto>();
  const blockedNodes: string[] = [];
  const warnings: string[] = [];
  const rescaled: RescaleChange[] = [];
  const inputBytes = encodeModel(model).length;

  // Element types as they are *before* any conversion. Used to decide whether
  // an op is a float op at all (an int64 `Range` must not be blocked) and to
  // keep cast placement honest.
  const originalDtypes = inferKnownDtypes(graph);

  const nodeIndex = new Map<NodeProto, number>();
  graph.node.forEach((node, index) => nodeIndex.set(node, index));
  const blockNode = (node: NodeProto) => {
    blocked.add(node);
  };

  // ---------------------------------------------------------------- 1. blocks
  const defaultOps = new Set<string>(options.defaultBlockOps === false ? [] : DEFAULT_FP32_OPS);
  const userOps = new Set<string>(options.blockOps ?? []);
  for (const node of graph.node) {
    if (userOps.has(node.opType)) {
      blockNode(node);
    } else if (defaultOps.has(node.opType) && hasFloatInput(node, originalDtypes)) {
      // Int64 Range/TopK/… never needed fp16, so leave them alone.
      blockNode(node);
    }
  }
  for (const pattern of options.blockNodes ?? []) {
    for (const node of graph.node) {
      const label = nodeLabel(node, nodeIndex.get(node) ?? 0);
      if (matchesPattern(pattern, label) || matchesPattern(pattern, node.name)) blockNode(node);
    }
  }

  // ---------------------------------------------------- 2. overflow strategy
  if (overflow === 'rescale') {
    rescaled.push(...rescaleBatchNorms(model, { limit, targetMax }));
  }

  const graphInputNames = new Set(graph.input.map((valueInfo) => valueInfo.name));
  const initializers = new Set(graph.initializer.map((tensor) => tensor.name));
  const consumers = collectConsumers(graph);

  interface Unconvertible {
    name: string;
    absMax: number;
    reason: string;
    /** True when blocking consumers is enough (no data loss, no guesswork). */
    autoBlock: boolean;
  }
  const unconvertible: Unconvertible[] = [];

  const inspectConvertible = (tensor: TensorProto, name: string) => {
    if (tensor.dataType !== DataType.FLOAT) return;
    if (tensor.externalData && tensor.externalData.length > 0) {
      throw new Error(
        `tensor "${name}" stores its payload in an external file; materialize external data ` +
          `(onnx.external_data_helper.convert_model_from_external_data) before converting.`,
      );
    }
    if (graphInputNames.has(name)) {
      unconvertible.push({ name, absMax: 0, reason: 'is also a graph input', autoBlock: true });
      return;
    }
    const absMax = tensorAbsMax(tensor) ?? 0;
    if (!Number.isFinite(absMax)) {
      unconvertible.push({ name, absMax, reason: 'contains non-finite values', autoBlock: true });
      return;
    }
    if (absMax > limit) {
      unconvertible.push({
        name,
        absMax,
        reason: `|max| ${absMax.toExponential(3)} > fp16 max ${limit}`,
        autoBlock: false,
      });
    }
  };

  for (const tensor of graph.initializer) inspectConvertible(tensor, tensor.name);
  for (const node of graph.node) {
    if (node.opType !== 'Constant') continue;
    if (constantOutputDtype(node) !== DataType.FLOAT) continue;
    const name = node.output[0] || `<constant of ${node.name || node.opType}>`;
    const attribute = floatTensorAttribute(node);
    if (!attribute?.t) {
      // `value_float` / `value_floats`: ONNX has no fp16 attribute type, so the
      // value cannot be converted. Block its consumers instead of mixing types.
      unconvertible.push({
        name,
        absMax: 0,
        reason: 'is a scalar stored as a float attribute; ONNX has no fp16 attribute type',
        autoBlock: true,
      });
    } else if (convertConstants) {
      inspectConvertible(attribute.t, name);
    } else {
      unconvertible.push({
        name,
        absMax: 0,
        reason: 'is a Constant tensor and convertConstants is false',
        autoBlock: true,
      });
    }
  }

  const rangeOverflows = unconvertible.filter((entry) => !entry.autoBlock);

  if (overflow === 'error' && unconvertible.length > 0) {
    throw new OverflowError(
      unconvertible.map((entry) => ({
        name: entry.name,
        absMax: entry.absMax,
        limit,
        fixable: entry.reason.startsWith('|max|'),
      })),
      limit,
    );
  }

  if (overflow === 'rescale' && rangeOverflows.length > 0) {
    // BatchNorm overflows were already repaired above; anything left cannot be
    // fixed algebraically, so refuse instead of silently clamping.
    throw new OverflowError(
      rangeOverflows.map((entry) => ({
        name: entry.name,
        absMax: entry.absMax,
        limit,
        fixable: false,
      })),
      limit,
    );
  }

  // Blocking is safe in every mode for tensors we must not touch (graph inputs,
  // non-finite payloads) and is the whole strategy in `block` mode.
  for (const entry of unconvertible) {
    if (overflow !== 'block' && !entry.autoBlock) continue;
    const users = consumers.get(entry.name) ?? [];
    for (const user of users) blockNode(user);
    if (users.length === 0) {
      warnings.push(`"${entry.name}" ${entry.reason} and has no consumers; left untouched.`);
    }
  }

  // Tensors whose every consumer is fp32 can stay fp32 (the parameters of a
  // blocked BatchNorm, for example). Everything else is converted and gets a
  // Cast at the boundary.
  const keepFloat32 = new Map<string, string>();
  for (const tensor of graph.initializer) {
    if (tensor.dataType !== DataType.FLOAT) continue;
    const users = consumers.get(tensor.name) ?? [];
    if (users.length > 0 && users.every((user) => blocked.has(user))) {
      keepFloat32.set(tensor.name, `all ${users.length} consumer node(s) stay fp32`);
    }
  }

  // ---------------------------------------------- 3. convert float payloads
  let convertedTensors = 0;
  let convertedBytes = 0;
  const fp16Tensors = new Set<string>();

  for (const tensor of graph.initializer) {
    if (tensor.dataType !== DataType.FLOAT) continue;
    if (keepFloat32.has(tensor.name)) continue;
    convertedBytes += convertTensor(tensor);
    convertedTensors += 1;
    fp16Tensors.add(tensor.name);
  }

  let constantsConverted = 0;
  if (convertConstants) {
    for (const node of graph.node) {
      const attribute = floatTensorAttribute(node);
      if (!attribute?.t) continue;
      const output = node.output[0] ?? '';
      const users = consumers.get(output) ?? [];
      if (users.length > 0 && users.every((user) => blocked.has(user))) continue;
      convertedBytes += convertTensor(attribute.t);
      constantsConverted += 1;
      fp16Tensors.add(output);
    }
  }

  // Existing `Cast(…, to=float32)` nodes are part of the graph's arithmetic
  // (shape math, FFT normalisation). If they keep producing fp32 while their
  // consumers become fp16, the graph stops type-checking — retarget them.
  const graphOutputNames = new Set(graph.output.map((valueInfo) => valueInfo.name));
  const anythingConverted = convertedTensors > 0 || constantsConverted > 0;
  let retargetedCasts = 0;
  if (anythingConverted) {
    for (const node of graph.node) {
      if (node.opType !== 'Cast' || blocked.has(node)) continue;
      const attribute = node.attribute.find((entry) => entry.name === 'to');
      if (!attribute || toInt(attribute.i) !== DataType.FLOAT) continue;
      const output = node.output[0] ?? '';
      const users = consumers.get(output) ?? [];
      // A cast that only feeds fp32 nodes can stay fp32 — turning it into
      // fp16 + a cast back to fp32 would be a pointless round-trip.
      if (users.length > 0 && users.every((user) => blocked.has(user)) && !graphOutputNames.has(output)) {
        continue;
      }
      attribute.i = DataType.FLOAT16;
      retargetedCasts += 1;
    }
  }

  // ------------------------------------------------- 4. fp16 ⇄ fp32 boundaries
  let castCount = 0;
  const castTo = (source: string, to: number, prefix: string): string => {
    castCount += 1;
    const output = `${source}__${prefix}${castCount}`;
    graph.node.push(makeCast(source, output, to, `fp16guard_cast_${prefix}_${castCount}`));
    return output;
  };

  // 4a. graph inputs: keep the public type fp32, cast inside the graph.
  if (keepIoTypes && anythingConverted) {
    for (const valueInfo of graph.input) {
      if (valueInfo.type?.tensorType?.elemType !== DataType.FLOAT) continue;
      if (initializers.has(valueInfo.name)) continue;
      const casted = castTo(valueInfo.name, DataType.FLOAT16, 'in');
      for (const consumer of consumers.get(valueInfo.name) ?? []) {
        if (blocked.has(consumer)) continue;
        for (let i = 0; i < consumer.input.length; i += 1) {
          if (consumer.input[i] === valueInfo.name) consumer.input[i] = casted;
        }
      }
    }
  } else if (anythingConverted) {
    for (const valueInfo of graph.input) {
      if (valueInfo.type?.tensorType?.elemType === DataType.FLOAT) setElemType(valueInfo, DataType.FLOAT16);
    }
  }

  // 4b. blocked nodes: every fp16 input gets a Cast, every fp16 output too.
  const blockedSnapshot = [...blocked];
  const producers = collectProducers(graph);

  // Resolve the actual element type of every tensor. Blocked nodes produce
  // fp32; Cast nodes declare it; int-producing ops are respected. Anything not
  // in the map (graph inputs, fp32 initializers) is not fp16.
  const dtypeOf = new Map<string, number>();
  for (const tensor of graph.initializer) dtypeOf.set(tensor.name, tensor.dataType);
  for (const node of graph.node) {
    if (node.opType !== 'Constant') continue;
    const type = constantOutputDtype(node); // already updated by conversion
    if (type === undefined) continue;
    for (const output of node.output) if (output) dtypeOf.set(output, type);
  }
  for (const valueInfo of graph.input) {
    const elemType = valueInfo.type?.tensorType?.elemType;
    if (elemType) dtypeOf.set(valueInfo.name, elemType);
  }
  for (const node of graph.node) {
    if (node.opType === 'Constant') continue;
    let dtype: number;
    if (node.opType === 'Cast') {
      const attribute = node.attribute.find((entry) => entry.name === 'to');
      dtype = attribute ? toInt(attribute.i) : DataType.FLOAT;
    } else {
      const original = inferNodeDtype(node, originalDtypes);
      if (original !== undefined && !FLOAT_TYPES.has(original)) {
        // int64/bool/string plumbing: never fp16, never needs a Cast.
        dtype = original;
      } else {
        dtype = blocked.has(node) ? DataType.FLOAT : DataType.FLOAT16;
      }
    }
    for (const output of node.output) if (output) dtypeOf.set(output, dtype);
  }
  const isFp16Tensor = (name: string): boolean => dtypeOf.get(name) === DataType.FLOAT16;

  const fp32InputCache = new Map<string, string>();
  for (const node of blockedSnapshot) {
    for (let i = 0; i < node.input.length; i += 1) {
      const input = node.input[i];
      if (!input || !isFp16Tensor(input)) continue;
      let casted = fp32InputCache.get(input);
      if (!casted) {
        casted = castTo(input, DataType.FLOAT, 'fp32');
        fp32InputCache.set(input, casted);
      }
      node.input[i] = casted;
    }
  }

  for (const node of blockedSnapshot) {
    for (const output of node.output) {
      if (!output) continue;
      const users = (consumers.get(output) ?? []).filter((user) => !blocked.has(user));
      if (users.length === 0) continue;
      const casted = castTo(output, DataType.FLOAT16, 'fp16');
      for (const user of users) {
        for (let i = 0; i < user.input.length; i += 1) {
          if (user.input[i] === output) user.input[i] = casted;
        }
      }
    }
  }

  // 4c. graph outputs: the graph computes fp16, the caller still gets fp32.
  if (keepIoTypes && anythingConverted) {
    for (const valueInfo of graph.output) {
      if (valueInfo.type?.tensorType?.elemType !== DataType.FLOAT) continue;
      const producer = producers.get(valueInfo.name);
      if (!producer || blocked.has(producer)) continue;
      const fp16Name = `${valueInfo.name}__out_fp16`;
      for (let i = 0; i < producer.output.length; i += 1) {
        if (producer.output[i] === valueInfo.name) producer.output[i] = fp16Name;
      }
      renameNodeInputs(graph, valueInfo.name, fp16Name);
      castCount += 1;
      graph.node.push(makeCast(fp16Name, valueInfo.name, DataType.FLOAT, `fp16guard_cast_out_${castCount}`));
    }
  } else if (anythingConverted) {
    for (const valueInfo of graph.output) {
      if (valueInfo.type?.tensorType?.elemType === DataType.FLOAT) setElemType(valueInfo, DataType.FLOAT16);
    }
  }

  // 4d. keep intermediate value_info in sync so shape inference does not lie.
  const outputNames = new Set(graph.output.map((valueInfo) => valueInfo.name));
  for (const valueInfo of graph.valueInfo) {
    if (outputNames.has(valueInfo.name)) continue;
    if (valueInfo.type?.tensorType?.elemType !== DataType.FLOAT) continue;
    const producer = producers.get(valueInfo.name);
    if (producer && !blocked.has(producer)) setElemType(valueInfo, DataType.FLOAT16);
  }

  // ------------------------------------------------------- 5. finish & report
  for (const node of blocked) blockedNodes.push(nodeLabel(node, nodeIndex.get(node) ?? 0));

  if (shouldToposort) {
    try {
      topologicalSort(graph);
    } catch (error) {
      throw new Error(`failed to topologically sort the converted graph: ${(error as Error).message}`);
    }
  }

  if (shouldValidate) {
    const problems = validateGraph(graph);
    if (problems.length > 0) {
      throw new Error(`converted graph is invalid:\n${problems.slice(0, 10).join('\n')}`);
    }
  }

  const bytes = encodeModel(model);
  const keptFloat32 = [...keepFloat32.entries()].map(([name, reason]) => ({ name, reason }));
  for (const entry of unconvertible) {
    if (!entry.autoBlock) {
      warnings.push(`"${entry.name}" ${entry.reason}; it stays fp32 and its consumers are blocked.`);
    }
  }

  return {
    model,
    report: {
      inputBytes,
      outputBytes: bytes.length,
      bytesSaved: inputBytes - bytes.length,
      ratio: bytes.length / inputBytes,
      convertedTensors,
      convertedBytes,
      constantsConverted,
      retargetedCasts,
      keptFloat32,
      insertedCasts: castCount,
      rescaled,
      blockedNodes,
      warnings,
    },
  };
}

/** Decode, convert and re-encode. The input bytes are never mutated. */
export function convertBytes(
  bytes: Uint8Array,
  options: ConvertOptions = {},
): { bytes: Uint8Array; report: ConvertReport } {
  const { model, report } = convertModel(decodeModel(bytes), options);
  const encoded = encodeModel(model);
  return {
    bytes: encoded,
    report: {
      ...report,
      inputBytes: bytes.length,
      outputBytes: encoded.length,
      bytesSaved: bytes.length - encoded.length,
      ratio: encoded.length / bytes.length,
    },
  };
}
