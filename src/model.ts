/**
 * ONNX model I/O and tensor helpers.
 *
 * Everything here is synchronous, allocation-heavy but predictable: we decode the
 * whole protobuf once, mutate it in place, and encode it back. A 200 MB model
 * decodes in ~100 ms, so the CLI does not bother with streaming.
 */
import {
  AttrType,
  DataType,
  onnx,
  type GraphProto,
  type LongLike,
  type ModelProto,
  type NodeProto,
  type PbModelProto,
  type TensorProto,
  type ValueInfoProto,
} from './onnx.js';
import { bytesPerElement, float16ToFloat32 } from './fp16.js';

/** Decode a serialized ONNX `ModelProto`. */
export function decodeModel(bytes: Uint8Array): ModelProto {
  return onnx.ModelProto.decode(bytes) as unknown as ModelProto;
}

/** Serialize a `ModelProto` back to bytes. */
export function encodeModel(model: ModelProto): Uint8Array {
  const encoded = onnx.ModelProto.encode(model as unknown as PbModelProto).finish();
  // protobufjs returns a Uint8Array already; copy out of the pooled buffer.
  return new Uint8Array(encoded);
}

/** protobufjs represents int64 as `Long`; normalize to a JS number. */
export function toNumber(value: number | LongLike | null | undefined): number {
  if (typeof value === 'number') return value;
  if (value && typeof value.toNumber === 'function') return value.toNumber();
  return 0;
}

export function tensorDims(tensor: TensorProto): number[] {
  return tensor.dims.map((dim) => toNumber(dim));
}

export function tensorElementCount(tensor: TensorProto): number {
  let count = 1;
  for (const dim of tensorDims(tensor)) count *= dim;
  return count;
}

/** Bytes actually stored for a tensor (raw data if present, else by data type). */
export function tensorBytes(tensor: TensorProto): number {
  if (tensor.rawData && tensor.rawData.length > 0) return tensor.rawData.length;
  return tensorElementCount(tensor) * bytesPerElement(tensor.dataType);
}

export function isFloatTensor(tensor: TensorProto): boolean {
  return tensor.dataType === DataType.FLOAT;
}

/**
 * Read a float32 tensor into a fresh `Float32Array`.
 * Supports both `raw_data` (the normal layout) and the deprecated `float_data`.
 */
export function readFloat32Tensor(tensor: TensorProto): Float32Array {
  if (tensor.dataType !== DataType.FLOAT) {
    throw new TypeError(`tensor "${tensor.name}" is not float32 (dataType=${tensor.dataType})`);
  }
  const count = tensorElementCount(tensor);
  if (tensor.rawData && tensor.rawData.length > 0) {
    if (tensor.rawData.length !== count * 4) {
      throw new Error(
        `tensor "${tensor.name}" declares ${count} elements but has ${tensor.rawData.length} bytes of raw data`,
      );
    }
    // Copy into an aligned buffer before viewing it as Float32Array.
    const copy = new Uint8Array(count * 4);
    copy.set(tensor.rawData.subarray(0, count * 4));
    return new Float32Array(copy.buffer);
  }
  if (tensor.floatData && tensor.floatData.length > 0) {
    return Float32Array.from(tensor.floatData as number[]);
  }
  return new Float32Array(count);
}

/** Write float32 values into a tensor as `raw_data` (drops `float_data`). */
export function writeFloat32Tensor(tensor: TensorProto, data: Float32Array): void {
  const copy = new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  tensor.dataType = DataType.FLOAT;
  tensor.rawData = copy;
  tensor.floatData = [];
  tensor.doubleData = [];
  tensor.int32Data = [];
  tensor.int64Data = [];
  tensor.uint64Data = [];
}

/** Read a float16 tensor as float32 values (decoded, not re-rounded). */
export function readFloat16Tensor(tensor: TensorProto): Float32Array {
  const count = tensorElementCount(tensor);
  if (tensor.rawData && tensor.rawData.length > 0) {
    if (tensor.rawData.length !== count * 2) {
      throw new Error(
        `tensor "${tensor.name}" declares ${count} fp16 elements but has ${tensor.rawData.length} bytes`,
      );
    }
    const copy = new Uint8Array(count * 2);
    copy.set(tensor.rawData.subarray(0, count * 2));
    return float16ToFloat32(new Uint16Array(copy.buffer));
  }
  return new Float32Array(count);
}

/** Store 16-bit half-precision bit patterns in a tensor. */
export function writeFloat16Bits(tensor: TensorProto, bits: Uint16Array): void {
  const copy = new Uint8Array(bits.buffer.slice(bits.byteOffset, bits.byteOffset + bits.byteLength));
  tensor.dataType = DataType.FLOAT16;
  tensor.rawData = copy;
  tensor.floatData = [];
  tensor.doubleData = [];
  tensor.int32Data = [];
  tensor.int64Data = [];
  tensor.uint64Data = [];
}

/**
 * Absolute maximum of a float tensor's payload, or `undefined` for non-float
 * tensors. This is a full scan: for a 200 MB model it costs ~60 ms, and it is
 * the check that would have caught the LaMa overflow before shipping.
 *
 * The scan reads `raw_data` in place when it is 4-byte aligned (the common case)
 * and falls back to a `DataView` otherwise, so it never copies the payload.
 */
export function tensorAbsMax(tensor: TensorProto): number | undefined {
  if (tensor.dataType !== DataType.FLOAT) return undefined;
  const count = tensorElementCount(tensor);
  const bytes = tensor.rawData;

  if (bytes && bytes.length > 0) {
    let max = 0;
    if (bytes.byteOffset % 4 === 0) {
      const view = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.length / 4));
      for (let i = 0; i < view.length; i += 1) {
        const abs = Math.abs(view[i]);
        if (abs > max) max = abs;
      }
      return max;
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < count; i += 1) {
      const abs = Math.abs(view.getFloat32(i * 4, true));
      if (abs > max) max = abs;
    }
    return max;
  }

  if (tensor.floatData && tensor.floatData.length > 0) {
    let max = 0;
    for (const value of tensor.floatData as number[]) {
      const abs = Math.abs(value);
      if (abs > max) max = abs;
    }
    return max;
  }

  return 0;
}

/** Map from tensor name → the node that produces it. */
export function collectProducers(graph: GraphProto): Map<string, NodeProto> {
  const producers = new Map<string, NodeProto>();
  for (const node of graph.node) {
    for (const output of node.output) {
      if (output) producers.set(output, node);
    }
  }
  return producers;
}

/** Map from tensor name → all nodes that read it. */
export function collectConsumers(graph: GraphProto): Map<string, NodeProto[]> {
  const consumers = new Map<string, NodeProto[]>();
  for (const node of graph.node) {
    for (const input of node.input) {
      if (!input) continue;
      const list = consumers.get(input);
      if (list) list.push(node);
      else consumers.set(input, [node]);
    }
  }
  return consumers;
}

/** Replace every reference to `from` in node inputs with `to`. */
export function renameNodeInputs(graph: GraphProto, from: string, to: string): void {
  for (const node of graph.node) {
    for (let i = 0; i < node.input.length; i += 1) {
      if (node.input[i] === from) node.input[i] = to;
    }
  }
}

/** Replace every reference to `from` in node outputs with `to`. */
export function renameNodeOutputs(graph: GraphProto, from: string, to: string): void {
  for (const node of graph.node) {
    for (let i = 0; i < node.output.length; i += 1) {
      if (node.output[i] === from) node.output[i] = to;
    }
  }
}

/**
 * Stable Kahn topological sort, in place.
 *
 * Runtimes (and `onnx.checker`) require nodes to appear in topological order.
 * Inserting Cast nodes at the end of the node list is convenient, but illegal
 * until we re-sort — which is exactly why this exists.
 */
export function topologicalSort(graph: GraphProto): void {
  const available = new Set<string>();
  for (const initializer of graph.initializer) available.add(initializer.name);
  for (const input of graph.input) available.add(input.name);

  const consumers = new Map<string, NodeProto[]>();
  const unresolved = new Map<NodeProto, number>();

  for (const node of graph.node) {
    let missing = 0;
    const seen = new Set<string>();
    for (const input of node.input) {
      if (!input || available.has(input) || seen.has(input)) continue;
      seen.add(input);
      missing += 1;
      const list = consumers.get(input);
      if (list) list.push(node);
      else consumers.set(input, [node]);
    }
    unresolved.set(node, missing);
  }

  const queue: NodeProto[] = [];
  for (const node of graph.node) {
    if (unresolved.get(node) === 0) queue.push(node);
  }

  const ordered: NodeProto[] = [];
  for (let head = 0; head < queue.length; head += 1) {
    const node = queue[head];
    ordered.push(node);
    for (const output of node.output) {
      if (!output) continue;
      available.add(output);
      const list = consumers.get(output);
      if (!list) continue;
      for (const consumer of list) {
        const left = (unresolved.get(consumer) ?? 0) - 1;
        unresolved.set(consumer, left);
        if (left === 0) queue.push(consumer);
      }
    }
  }

  if (ordered.length !== graph.node.length) {
    const stuck = [...unresolved.entries()].filter(([, n]) => n > 0).slice(0, 3);
    const detail = stuck
      .map(([node, n]) => `${node.opType}(${node.name || '<unnamed>'}) waits for ${n} input(s)`)
      .join('; ');
    throw new Error(
      `graph is not a DAG: ${graph.node.length - ordered.length} node(s) unreachable (${detail})`,
    );
  }

  graph.node = ordered;
}

/** Structural sanity check; returns a list of human-readable problems. */
export function validateGraph(graph: GraphProto): string[] {
  const problems: string[] = [];
  const defined = new Set<string>();
  for (const initializer of graph.initializer) defined.add(initializer.name);
  for (const input of graph.input) defined.add(input.name);

  const produced = new Set<string>();
  for (const node of graph.node) {
    for (const input of node.input) {
      if (input && !defined.has(input)) {
        problems.push(`node ${node.opType}(${node.name || '<unnamed>'}) reads undefined tensor "${input}"`);
      }
    }
    for (const output of node.output) {
      if (!output) continue;
      if (produced.has(output)) problems.push(`tensor "${output}" is produced more than once`);
      produced.add(output);
      defined.add(output);
    }
  }

  for (const output of graph.output) {
    if (!defined.has(output.name)) {
      problems.push(`graph output "${output.name}" is never produced`);
    }
  }

  return problems;
}

/** Element type declared on a `ValueInfoProto` (undefined if unknown). */
export function elemTypeOf(valueInfo: ValueInfoProto): number | undefined {
  return valueInfo.type?.tensorType?.elemType ?? undefined;
}

export function setElemType(valueInfo: ValueInfoProto, elemType: number): void {
  if (!valueInfo.type) valueInfo.type = {};
  if (!valueInfo.type.tensorType) valueInfo.type.tensorType = {};
  valueInfo.type.tensorType.elemType = elemType;
}

/** Create a `Cast` node (fp16 ⇄ fp32 boundaries). */
export function makeCast(input: string, output: string, to: number, name: string): NodeProto {
  return {
    opType: 'Cast',
    name,
    input: [input],
    output: [output],
    attribute: [{ name: 'to', type: AttrType.INT, i: to }],
    domain: '',
  };
}

/** Sum of the stored payload of every initializer. */
export function initializerBytes(graph: GraphProto): number {
  let total = 0;
  for (const tensor of graph.initializer) total += tensorBytes(tensor);
  return total;
}
