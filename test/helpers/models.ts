/**
 * Tiny synthetic ONNX models used by the tests.
 *
 * They are built by hand (no Python, no fixtures on disk) so CI needs nothing
 * but Node. Shapes are deliberately small: the properties under test are about
 * dtypes, graph structure and BatchNorm arithmetic, not throughput.
 */
import { encodeModel } from '../../src/model.js';
import type { GraphProto, ModelProto, NodeProto, TensorProto } from '../../src/onnx.js';

export function floatTensor(name: string, dims: number[], values: number[]): TensorProto {
  const data = Float32Array.from(values);
  return {
    name,
    dataType: 1,
    dims,
    rawData: new Uint8Array(data.buffer.slice(0, data.byteLength)),
    floatData: [],
    doubleData: [],
    int32Data: [],
    int64Data: [],
    uint64Data: [],
    externalData: [],
  };
}

export function int64Tensor(name: string, dims: number[], values: bigint[]): TensorProto {
  const data = BigInt64Array.from(values);
  return {
    name,
    dataType: 7,
    dims,
    rawData: new Uint8Array(data.buffer.slice(0, data.byteLength)),
    floatData: [],
    doubleData: [],
    int32Data: [],
    int64Data: [],
    uint64Data: [],
    externalData: [],
  };
}

function node(
  name: string,
  opType: string,
  input: string[],
  output: string[],
  attribute: NodeProto['attribute'] = [],
): NodeProto {
  return { name, opType, input, output, attribute, domain: '' };
}

function tensorType(elemType: number, dims: (number | string)[]): ModelProto['graph']['input'][number] {
  return {
    name: '',
    type: {
      tensorType: {
        elemType,
        shape: {
          dim: dims.map((dim) => (typeof dim === 'number' ? { dimValue: dim } : { dimParam: dim })),
        },
      },
    },
  };
}

function io(name: string, elemType: number, dims: (number | string)[]) {
  return { ...tensorType(elemType, dims), name };
}

const CHANNELS = 3;

/**
 * `X → BN(overflow) → Relu → BN(ok) → Add(Constant) → Cast(float32) → Y`
 *
 * Covers: BatchNorm overflow, constant conversion, Cast retargeting, and the
 * fp32 I/O boundary.
 */
export function buildBatchNormModel(options: { varianceMax?: number } = {}): ModelProto {
  const varianceMax = options.varianceMax ?? 484_322;
  const graph: GraphProto = {
    name: 'bn-fixture',
    node: [
      node(
        'bn1',
        'BatchNormalization',
        ['X', 'bn1.scale', 'bn1.bias', 'bn1.mean', 'bn1.var'],
        ['bn1.out'],
        [{ name: 'epsilon', type: 1, f: 1e-5 }],
      ),
      node('relu', 'Relu', ['bn1.out'], ['relu.out']),
      node(
        'bn2',
        'BatchNormalization',
        ['relu.out', 'bn2.scale', 'bn2.bias', 'bn2.mean', 'bn2.var'],
        ['bn2.out'],
        // no epsilon attribute: the converter must add a scaled one
      ),
      node('add', 'Add', ['bn2.out', 'const.out'], ['add.out']),
      // A pre-existing fp32 cast: must be retargeted at fp16, not left behind.
      node('cast', 'Cast', ['add.out'], ['Y'], [{ name: 'to', type: 2, i: 1 }]),
    ],
    initializer: [
      floatTensor('bn1.scale', [CHANNELS], [1, 1.25, 0.75]),
      floatTensor('bn1.bias', [CHANNELS], [0.1, -0.2, 0.3]),
      floatTensor('bn1.mean', [CHANNELS], [0.5, -0.5, 0.25]),
      floatTensor('bn1.var', [CHANNELS], [varianceMax, varianceMax / 2, 1]),
      floatTensor('bn2.scale', [CHANNELS], [0.9, 1.1, 1]),
      floatTensor('bn2.bias', [CHANNELS], [0, 0.05, -0.05]),
      floatTensor('bn2.mean', [CHANNELS], [0.1, 0.2, -0.1]),
      floatTensor('bn2.var', [CHANNELS], [0.5, 2, 3]),
    ],
    input: [io('X', 1, [1, CHANNELS, 4, 4])],
    output: [io('Y', 1, [1, CHANNELS, 4, 4])],
    valueInfo: [],
  };

  const constant: NodeProto = {
    name: 'const',
    opType: 'Constant',
    input: [],
    output: ['const.out'],
    domain: '',
    attribute: [{ name: 'value', type: 4, t: floatTensor('', [CHANNELS], [0.5, 0.25, 0.125]) }],
  };
  graph.node.push(constant);

  return {
    irVersion: 8,
    opsetImport: [{ domain: '', version: 17 }],
    producerName: 'onnx-fp16-guard-tests',
    graph,
  };
}

/**
 * `Range(0, 4, 1) → Relu → Y` — Range has no fp16 kernel in ONNX Runtime, so it
 * must be blocked by default and surrounded by Casts.
 */
export function buildRangeModel(): ModelProto {
  const graph: GraphProto = {
    name: 'range-fixture',
    node: [
      node('range', 'Range', ['start', 'limit', 'delta'], ['range.out']),
      node('relu', 'Relu', ['range.out'], ['Y']),
    ],
    initializer: [
      floatTensor('start', [], [0]),
      floatTensor('limit', [], [4]),
      floatTensor('delta', [], [1]),
    ],
    input: [],
    output: [io('Y', 1, [4])],
    valueInfo: [],
  };
  return {
    irVersion: 8,
    opsetImport: [{ domain: '', version: 17 }],
    producerName: 'onnx-fp16-guard-tests',
    graph,
  };
}

export function serialize(model: ModelProto): Uint8Array {
  return encodeModel(model);
}

/** Naive NCHW BatchNorm for the numeric-equivalence tests. */
export function batchNormForward(
  input: Float32Array,
  channels: number,
  scale: Float32Array,
  bias: Float32Array,
  mean: Float32Array,
  variance: Float32Array,
  epsilon: number,
): Float32Array {
  const out = new Float32Array(input.length);
  const spatial = input.length / channels;
  for (let c = 0; c < channels; c += 1) {
    const denominator = Math.sqrt(variance[c] + epsilon);
    for (let i = 0; i < spatial; i += 1) {
      const index = c * spatial + i;
      out[index] = ((input[index] - mean[c]) / denominator) * scale[c] + bias[c];
    }
  }
  return out;
}

/**
 * `Range(int64) → Cast(to=float32) → Relu → Y`
 *
 * An int64 Range never needed fp16 and must not be blocked: casting its inputs
 * would produce a mixed-type Range, which ONNX Runtime rejects at load time.
 */
export function buildInt64RangeModel(): ModelProto {
  const graph: GraphProto = {
    name: 'int64-range-fixture',
    node: [
      node('range', 'Range', ['start', 'limit', 'delta'], ['range.out']),
      node('cast', 'Cast', ['range.out'], ['range.float'], [{ name: 'to', type: 2, i: 1 }]),
      node('relu', 'Relu', ['range.float'], ['Y']),
    ],
    initializer: [
      int64Tensor('start', [], [0n]),
      int64Tensor('limit', [], [4n]),
      int64Tensor('delta', [], [1n]),
    ],
    input: [],
    output: [io('Y', 1, [4])],
    valueInfo: [],
  };
  return {
    irVersion: 8,
    opsetImport: [{ domain: '', version: 17 }],
    producerName: 'onnx-fp16-guard-tests',
    graph,
  };
}

/**
 * `X → Add(X, 0.5) → Relu → Y`, where 0.5 is a `Constant` with `value_float`.
 *
 * ONNX has no fp16 attribute type, so the scalar cannot be converted. Its
 * consumer must stay fp32 instead of mixing an fp32 tensor into an fp16 op.
 */
export function buildFloatAttributeModel(): ModelProto {
  const graph: GraphProto = {
    name: 'float-attribute-fixture',
    node: [
      node('add', 'Add', ['X', 'const.out'], ['add.out']),
      node('relu', 'Relu', ['add.out'], ['Y']),
      {
        name: 'const',
        opType: 'Constant',
        input: [],
        output: ['const.out'],
        domain: '',
        attribute: [{ name: 'value_float', type: 1, f: 0.5 }],
      },
    ],
    initializer: [],
    input: [io('X', 1, [1, 3, 4, 4])],
    output: [io('Y', 1, [1, 3, 4, 4])],
    valueInfo: [],
  };
  return {
    irVersion: 8,
    opsetImport: [{ domain: '', version: 17 }],
    producerName: 'onnx-fp16-guard-tests',
    graph,
  };
}
