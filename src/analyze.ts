/**
 * Model inspection: what is in this ONNX file, and what would break in fp16?
 *
 * `inspectModel()` never mutates the model and never writes a file. It is the
 * cheap check you run before spending 30 seconds converting a 200 MB model —
 * and the one you wire into CI with `--fail-on-overflow`.
 */
import { DataType, FP16_MAX, dataTypeName, type ModelProto, type TensorProto } from './onnx.js';
import { collectConsumers, tensorAbsMax, tensorBytes, tensorElementCount, toNumber } from './model.js';
import { findBatchNormOverflows, rescaleFactor, DEFAULT_TARGET_MAX } from './bn.js';

export type OverflowFix = 'rescale-batchnorm' | 'keep-float32' | 'none';

export interface OverflowFinding {
  /** Tensor that cannot be represented in fp16. */
  tensor: string;
  /** Where the tensor lives. */
  kind: 'batch-norm-running-var' | 'initializer' | 'constant';
  absMax: number;
  elements: number;
  bytes: number;
  /** Human-readable consumers, e.g. `BatchNormalization(model.5.conv1.bn_l)`. */
  consumers: string[];
  /** What the converter can do about it. */
  fix: OverflowFix;
  /** Power-of-two factor used by the `rescale-batchnorm` fix. */
  k?: number;
  /** True when no automated fix applies (non-finite values, graph inputs, …). */
  blocking: boolean;
}

export interface DataTypeBucket {
  dataType: number;
  name: string;
  count: number;
  bytes: number;
}

export interface OpsetInfo {
  domain: string;
  version: number;
}

export interface IoInfo {
  name: string;
  elemType: number;
  elemTypeName: string;
  dims: (number | string)[];
}

export interface InspectionReport {
  fileBytes: number;
  irVersion: number;
  opsets: OpsetInfo[];
  producer: string;
  graphName: string;
  counts: { nodes: number; initializers: number; constants: number };
  tensors: {
    count: number;
    payloadBytes: number;
    buckets: DataTypeBucket[];
    float32Count: number;
    float32Bytes: number;
    float16Bytes: number;
    projectedPayloadBytes: number;
  };
  overflow: {
    limit: number;
    /** Tensors that exceed the fp16 range. */
    findings: OverflowFinding[];
    blockingCount: number;
    batchNorm: { total: number; overflowing: number };
  };
  ops: { opType: string; count: number }[];
  io: { inputs: IoInfo[]; outputs: IoInfo[] };
  warnings: string[];
}

export interface InspectOptions {
  /** fp16 representable maximum; defaults to 65504. */
  limit?: number;
  /** Target maximum for the BatchNorm rescale; defaults to 40000. */
  targetMax?: number;
}

function ioInfo(valueInfo: {
  name: string;
  type?: {
    tensorType?: {
      elemType?: number;
      shape?: { dim?: { dimValue?: unknown; dimParam?: string | null }[] } | null;
    } | null;
  } | null;
}): IoInfo {
  const tensorType = valueInfo.type?.tensorType;
  const elemType = tensorType?.elemType ?? 0;
  const dims: (number | string)[] = [];
  for (const dim of tensorType?.shape?.dim ?? []) {
    if (dim.dimParam) dims.push(dim.dimParam);
    else if (dim.dimValue !== undefined && dim.dimValue !== null) dims.push(toNumber(dim.dimValue as number));
    else dims.push('?');
  }
  return { name: valueInfo.name, elemType, elemTypeName: dataTypeName(elemType), dims };
}

/**
 * Inspect a decoded model.
 *
 * Cost: one pass over the float payload (≈60 ms for 200 MB), plus one per
 * overflowing BatchNorm tensor.
 */
export function inspectModel(model: ModelProto, options: InspectOptions = {}): InspectionReport {
  const limit = options.limit ?? FP16_MAX;
  const targetMax = options.targetMax ?? DEFAULT_TARGET_MAX;
  const graph = model.graph;
  const warnings: string[] = [];

  const consumers = collectConsumers(graph);
  const consumerLabel = (tensorName: string): string[] =>
    (consumers.get(tensorName) ?? []).map((node) => `${node.opType}(${node.name || '<unnamed>'})`);

  const buckets = new Map<number, DataTypeBucket>();
  const findings: OverflowFinding[] = [];
  let payloadBytes = 0;
  let float32Bytes = 0;
  let float32Count = 0;
  let float16Bytes = 0;
  let projectedPayloadBytes = 0;

  const record = (dataType: number, bytes: number) => {
    payloadBytes += bytes;
    const bucket = buckets.get(dataType);
    if (bucket) {
      bucket.count += 1;
      bucket.bytes += bytes;
    } else {
      buckets.set(dataType, { dataType, name: dataTypeName(dataType), count: 1, bytes });
    }
  };

  const scanTensor = (tensor: TensorProto, kind: 'initializer' | 'constant', displayName: string) => {
    const bytes = tensorBytes(tensor);
    record(tensor.dataType, bytes);

    if (tensor.dataType === DataType.FLOAT) {
      float32Count += 1;
      float32Bytes += bytes;
      projectedPayloadBytes += bytes / 2;
      const absMax = tensorAbsMax(tensor) ?? 0;
      if (absMax > limit) {
        findings.push({
          tensor: displayName,
          kind: kind === 'constant' ? 'constant' : 'initializer',
          absMax,
          elements: tensorElementCount(tensor),
          bytes,
          consumers: consumerLabel(tensor.name),
          fix: 'keep-float32',
          blocking: !Number.isFinite(absMax),
        });
      }
    } else {
      projectedPayloadBytes += bytes;
      if (tensor.dataType === DataType.FLOAT16) float16Bytes += bytes;
    }
  };

  for (const tensor of graph.initializer) scanTensor(tensor, 'initializer', tensor.name);

  let constants = 0;
  let constantIndex = 0;
  for (const node of graph.node) {
    if (node.opType !== 'Constant') continue;
    constants += 1;
    constantIndex += 1;
    for (const attribute of node.attribute) {
      if (attribute.type !== 4 || !attribute.t) continue; // TENSOR
      const label = `${node.output[0] ?? `<constant:${constantIndex}>`}`;
      scanTensor(attribute.t, 'constant', label);
    }
  }

  // BatchNorm-specific classification: rescaling is preferred over blocking.
  const bnOverflows = findBatchNormOverflows(model, { limit, targetMax });
  const byTensor = new Map(findings.map((finding) => [finding.tensor, finding]));
  for (const bn of bnOverflows) {
    const finding = byTensor.get(bn.variance);
    if (!finding) continue;
    finding.kind = 'batch-norm-running-var';
    finding.consumers = [`BatchNormalization(${bn.node})`];
    if (bn.fixable && bn.inferenceOnly && Number.isFinite(bn.absMaxVariance)) {
      finding.fix = 'rescale-batchnorm';
      finding.k = bn.k;
      finding.blocking = false;
    } else {
      finding.fix = 'keep-float32';
      finding.blocking = false;
      if (!bn.fixable) {
        warnings.push(
          `BatchNorm "${bn.node}" overflows but its parameters are not static initializers; it will stay in fp32.`,
        );
      }
      if (!bn.inferenceOnly) {
        warnings.push(
          `BatchNorm "${bn.node}" has training outputs; keeping it in fp32 changes nothing for inference.`,
        );
      }
    }
  }

  // Anything above the limit that is not a BatchNorm variance has no automatic
  // fix. Report it loudly instead of clamping silently.
  for (const finding of findings) {
    if (finding.kind === 'initializer') {
      warnings.push(
        `tensor "${finding.tensor}" has |max|=${finding.absMax.toExponential(3)} > ${limit} and is not a ` +
          `BatchNorm variance; it will be kept in fp32 (${consumerLabel(finding.tensor).slice(0, 3).join(', ')}).`,
      );
    }
  }

  for (const tensor of graph.initializer) {
    if (tensor.externalData && tensor.externalData.length > 0) {
      warnings.push(
        `tensor "${tensor.name}" stores its payload externally (${tensor.externalData.length} reference(s)); ` +
          `external data is not converted and must be materialized first.`,
      );
    }
  }

  const totalBatchNorms = graph.node.filter((node) => node.opType === 'BatchNormalization').length;
  const ops = new Map<string, number>();
  for (const node of graph.node) ops.set(node.opType, (ops.get(node.opType) ?? 0) + 1);

  const opsets: OpsetInfo[] = model.opsetImport.map((entry) => ({
    domain: entry.domain || 'ai.onnx',
    version: toNumber(entry.version),
  }));
  if (new Set(opsets.map((entry) => entry.domain)).size > 1) {
    warnings.push(
      `model imports ${opsets.length} opsets (${opsets.map((entry) => `${entry.domain}@${entry.version}`).join(', ')}); ` +
        `domain-specific ops are converted as-is.`,
    );
  }

  const blockingCount = findings.filter((finding) => finding.blocking).length;
  if (blockingCount > 0) {
    warnings.push(`${blockingCount} tensor(s) contain non-finite values and cannot be converted.`);
  }

  for (const valueInfo of graph.input) {
    if (!valueInfo.name) warnings.push('graph has an unnamed input');
  }

  return {
    fileBytes: 0,
    irVersion: toNumber(model.irVersion),
    opsets,
    producer: [model.producerName, model.producerVersion].filter(Boolean).join(' '),
    graphName: graph.name,
    counts: { nodes: graph.node.length, initializers: graph.initializer.length, constants },
    tensors: {
      count: graph.initializer.length + constantIndex,
      payloadBytes,
      buckets: [...buckets.values()].sort((a, b) => b.bytes - a.bytes),
      float32Count,
      float32Bytes,
      float16Bytes,
      projectedPayloadBytes,
    },
    overflow: {
      limit,
      findings: findings.sort((a, b) => b.absMax - a.absMax),
      blockingCount,
      batchNorm: { total: totalBatchNorms, overflowing: bnOverflows.length },
    },
    ops: [...ops.entries()]
      .map(([opType, count]) => ({ opType, count }))
      .sort((a, b) => b.count - a.count || a.opType.localeCompare(b.opType)),
    io: {
      inputs: graph.input.map(ioInfo),
      outputs: graph.output.map(ioInfo),
    },
    warnings,
  };
}

/** Convenience re-export so callers can compute the factor for custom repairs. */
export { rescaleFactor };
