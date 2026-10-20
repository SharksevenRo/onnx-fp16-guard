import { describe, expect, it } from 'vitest';
import {
  convertBytes,
  convertModel,
  matchesPattern,
  OverflowError,
  DEFAULT_FP32_OPS,
} from '../src/convert.js';
import { inspectModel } from '../src/analyze.js';
import {
  decodeModel,
  encodeModel,
  readFloat16Tensor,
  readFloat32Tensor,
  validateGraph,
} from '../src/model.js';
import {
  batchNormForward,
  buildBatchNormModel,
  buildFloatAttributeModel,
  buildInt64RangeModel,
  buildRangeModel,
  serialize,
} from './helpers/models.js';

const FLOAT = 1;
const FLOAT16 = 10;

function castTarget(node: {
  opType: string;
  attribute: { name?: string; i?: unknown }[];
}): number | undefined {
  if (node.opType !== 'Cast') return undefined;
  const attribute = node.attribute.find((entry) => entry.name === 'to');
  return attribute ? Number(attribute.i) : undefined;
}

describe('inspectModel', () => {
  it('classifies the overflowing BatchNorm and projects the size', () => {
    const bytes = serialize(buildBatchNormModel());
    const report = inspectModel(decodeModel(bytes));
    expect(report.counts.nodes).toBeGreaterThan(4);
    expect(report.overflow.findings).toHaveLength(1);
    const finding = report.overflow.findings[0];
    expect(finding.kind).toBe('batch-norm-running-var');
    expect(finding.fix).toBe('rescale-batchnorm');
    expect(finding.k).toBe(16);
    expect(finding.absMax).toBeCloseTo(484_322, 0);
    expect(report.overflow.batchNorm).toEqual({ total: 2, overflowing: 1 });
    expect(report.tensors.float32Count).toBeGreaterThan(0);
    expect(report.tensors.projectedPayloadBytes).toBeLessThan(report.tensors.payloadBytes);
    expect(report.ops.map((entry) => entry.opType)).toContain('BatchNormalization');
  });

  it('is silent on a model that fits', () => {
    const report = inspectModel(decodeModel(serialize(buildBatchNormModel({ varianceMax: 1000 }))));
    expect(report.overflow.findings).toHaveLength(0);
    expect(report.overflow.blockingCount).toBe(0);
  });
});

describe('convertBytes', () => {
  it('converts every float tensor, rescales the overflow and keeps I/O fp32', () => {
    const source = serialize(buildBatchNormModel());
    const original = new Uint8Array(source);
    const { bytes, report } = convertBytes(source);

    // The input buffer is untouched.
    expect(Array.from(source)).toEqual(Array.from(original));
    expect(report.outputBytes).toBe(bytes.length);
    expect(report.bytesSaved).toBe(report.inputBytes - report.outputBytes);
    expect(report.rescaled).toHaveLength(1);

    // The tensor payload is what shrinks (this fixture has almost no weights, so
    // the extra Cast nodes make the *file* slightly larger — that is expected).
    const payload = (model: ReturnType<typeof decodeModel>) =>
      model.graph.initializer.reduce((total, tensor) => total + (tensor.rawData?.byteLength ?? 0), 0);
    expect(payload(decodeModel(bytes))).toBeLessThan(payload(decodeModel(source)) / 2 + 8);
    expect(report.convertedTensors).toBe(8);
    expect(report.constantsConverted).toBe(1);

    const model = decodeModel(bytes);
    expect(validateGraph(model.graph)).toEqual([]);

    // Every initializer is fp16 now.
    for (const tensor of model.graph.initializer) {
      expect(tensor.dataType, tensor.name).toBe(FLOAT16);
      const elements = tensor.dims.reduce<number>((total, dim) => total * Number(dim), 1);
      expect(tensor.rawData?.byteLength).toBe(elements * 2);
    }

    // Graph inputs/outputs keep their public fp32 type.
    expect(model.graph.input[0].type?.tensorType?.elemType).toBe(FLOAT);
    expect(model.graph.output[0].type?.tensorType?.elemType).toBe(FLOAT);

    // The pre-existing Cast(to=float32) was retargeted at fp16 …
    const retargeted = model.graph.node.find((node) => node.name === 'cast')!;
    expect(castTarget(retargeted)).toBe(FLOAT16);
    expect(report.retargetedCasts).toBe(1);

    // … and exactly two boundary casts were added (input + output).
    expect(report.insertedCasts).toBe(2);
    const inputCast = model.graph.node.find((node) => node.name.startsWith('fp16guard_cast_in_'));
    expect(inputCast && castTarget(inputCast)).toBe(FLOAT16);
    const outputCast = model.graph.node.find((node) => node.name.startsWith('fp16guard_cast_out_'));
    expect(outputCast && castTarget(outputCast)).toBe(FLOAT);
    expect(outputCast?.output).toEqual(['Y']);

    // BatchNorm epsilon was scaled along with var/scale.
    const bn1 = model.graph.node.find((node) => node.name === 'bn1')!;
    expect(bn1.attribute.find((attribute) => attribute.name === 'epsilon')?.f).toBeCloseTo(1e-5 / 256, 12);

    // The whole float payload is below the fp16 max.
    for (const tensor of model.graph.initializer) {
      const values = readFloat16Tensor(tensor);
      for (const value of values) expect(Number.isFinite(value)).toBe(true);
    }
  });

  it('stays numerically equivalent to the fp32 BatchNorm', () => {
    const source = serialize(buildBatchNormModel());
    const model = decodeModel(source);
    const input = Float32Array.from({ length: 3 * 4 * 4 }, (_, index) => Math.cos(index * 0.37) * 5);

    const originalParams = {
      scale: readFloat32Tensor(model.graph.initializer.find((t) => t.name === 'bn1.scale')!),
      bias: readFloat32Tensor(model.graph.initializer.find((t) => t.name === 'bn1.bias')!),
      mean: readFloat32Tensor(model.graph.initializer.find((t) => t.name === 'bn1.mean')!),
      variance: readFloat32Tensor(model.graph.initializer.find((t) => t.name === 'bn1.var')!),
    };
    const expected = batchNormForward(
      input,
      3,
      originalParams.scale,
      originalParams.bias,
      originalParams.mean,
      originalParams.variance,
      1e-5,
    );

    convertModel(model);
    const read = (name: string) => readFloat16Tensor(model.graph.initializer.find((t) => t.name === name)!);
    const bn1 = model.graph.node.find((node) => node.name === 'bn1')!;
    const epsilon = bn1.attribute.find((attribute) => attribute.name === 'epsilon')!.f as number;
    const actual = batchNormForward(
      input,
      3,
      read('bn1.scale'),
      read('bn1.bias'),
      read('bn1.mean'),
      read('bn1.var'),
      epsilon,
    );

    let maxDiff = 0;
    for (let i = 0; i < expected.length; i += 1)
      maxDiff = Math.max(maxDiff, Math.abs(expected[i] - actual[i]));
    // fp16 storage adds ~1e-3 relative error; the rescale itself adds nothing.
    expect(maxDiff).toBeLessThan(0.01);
  });

  it('refuses to guess with overflow=error', () => {
    const source = serialize(buildBatchNormModel());
    expect(() => convertBytes(source, { overflow: 'error' })).toThrow(OverflowError);
    try {
      convertBytes(source, { overflow: 'error' });
    } catch (error) {
      const overflowError = error as OverflowError;
      expect(overflowError.tensors).toHaveLength(1);
      expect(overflowError.tensors[0].name).toBe('bn1.var');
      expect(overflowError.message).toMatch(/exceed the fp16 range/);
    }
  });

  it('keeps the offending layer in fp32 with overflow=block', () => {
    const { bytes, report } = convertBytes(serialize(buildBatchNormModel()), { overflow: 'block' });
    const model = decodeModel(bytes);

    expect(report.blockedNodes).toEqual(['bn1']);
    const bn1 = model.graph.node.find((node) => node.name === 'bn1')!;
    // The parameters of a blocked node stay fp32 …
    for (const name of ['bn1.scale', 'bn1.bias', 'bn1.mean', 'bn1.var']) {
      expect(model.graph.initializer.find((t) => t.name === name)?.dataType).toBe(FLOAT);
      expect(report.keptFloat32.some((entry) => entry.name === name)).toBe(true);
    }
    // … its fp16 output is cast back to fp32 for the blocked node …
    const outCast = model.graph.node.find(
      (node) => castTarget(node) === FLOAT16 && node.input[0] === 'bn1.out',
    );
    expect(outCast).toBeDefined();
    // … and the (fp32) graph input feeds it directly.
    expect(bn1.input[0]).toBe('X');
    expect(validateGraph(model.graph)).toEqual([]);
  });

  it('blocks ops without fp16 kernels by default', () => {
    expect(DEFAULT_FP32_OPS).toContain('Range');
    const source = serialize(buildRangeModel());
    const { bytes, report } = convertBytes(source);

    const model = decodeModel(bytes);
    const range = model.graph.node.find((node) => node.opType === 'Range')!;
    expect(range).toBeDefined();
    expect(report.blockedNodes).toEqual(['range']);

    // The scalar inputs are only consumed by the blocked Range, so they stay
    // fp32 initializers and need no Cast at all.
    for (const name of ['start', 'limit', 'delta']) {
      const tensor = model.graph.initializer.find((entry) => entry.name === name)!;
      expect(tensor.dataType, name).toBe(FLOAT);
      expect(
        report.keptFloat32.some((entry) => entry.name === name),
        name,
      ).toBe(true);
    }
    const producers = new Map(
      model.graph.node.flatMap((node) => node.output.map((output) => [output, node] as const)),
    );
    for (const input of range.input) {
      expect(producers.has(input), input).toBe(false);
    }
    // And its output goes back to fp16 for Relu.
    const outputCast = model.graph.node.find((node) => node.input[0] === range.output[0])!;
    expect(castTarget(outputCast)).toBe(FLOAT16);
    expect(validateGraph(model.graph)).toEqual([]);
  });

  it('leaves int64 plumbing alone (int64 Range is not an fp16 op)', () => {
    const { bytes, report } = convertBytes(serialize(buildInt64RangeModel()));
    const model = decodeModel(bytes);

    expect(report.blockedNodes).toEqual([]);
    // Nothing float exists to convert, so the graph is left exactly as it was.
    expect(report.insertedCasts).toBe(0);

    const range = model.graph.node.find((node) => node.opType === 'Range')!;
    for (const name of ['start', 'limit', 'delta']) {
      expect(model.graph.initializer.find((tensor) => tensor.name === name)?.dataType).toBe(7);
    }
    for (const input of range.input) {
      const producer = model.graph.node.find((node) => node.output.includes(input));
      expect(producer?.opType, input).not.toBe('Cast');
    }

    // The int64 → float Cast is untouched (there is no fp16 payload to feed).
    const inner = model.graph.node.find((node) => node.name === 'cast')!;
    expect(castTarget(inner)).toBe(FLOAT);
    expect(validateGraph(model.graph)).toEqual([]);
  });

  it('keeps fp32 around a Constant stored as a float attribute', () => {
    const { bytes, report } = convertBytes(serialize(buildFloatAttributeModel()));
    const model = decodeModel(bytes);

    // The scalar cannot become fp16, so its consumer runs in fp32.
    expect(report.blockedNodes).toEqual(['add']);
    const constant = model.graph.node.find((node) => node.opType === 'Constant')!;
    expect(constant.attribute.find((attribute) => attribute.name === 'value_float')?.f).toBe(0.5);
    expect(validateGraph(model.graph)).toEqual([]);
  });

  it('can convert the I/O types as well', () => {
    const { bytes, report } = convertBytes(serialize(buildBatchNormModel()), { keepIoTypes: false });
    const model = decodeModel(bytes);
    expect(model.graph.input[0].type?.tensorType?.elemType).toBe(FLOAT16);
    expect(model.graph.output[0].type?.tensorType?.elemType).toBe(FLOAT16);
    expect(report.insertedCasts).toBe(0);
  });

  it('honours explicit blockOps and blockNodes', () => {
    const byOp = convertBytes(serialize(buildBatchNormModel({ varianceMax: 1000 })), { blockOps: ['Relu'] });
    expect(byOp.report.blockedNodes).toEqual(['relu']);

    const byPattern = convertBytes(serialize(buildBatchNormModel({ varianceMax: 1000 })), {
      blockNodes: ['bn*'],
    });
    expect(byPattern.report.blockedNodes.sort()).toEqual(['bn1', 'bn2']);
  });
});

describe('matchesPattern', () => {
  it('supports exact names, globs and /regex/', () => {
    expect(matchesPattern('relu', 'relu')).toBe(true);
    expect(matchesPattern('relu', 'relu2')).toBe(false);
    expect(matchesPattern('bn*', 'bn12')).toBe(true);
    expect(matchesPattern('bn*', 'xbn12')).toBe(false);
    expect(matchesPattern('/^bn\\d+$/', 'bn12')).toBe(true);
    expect(matchesPattern('/^bn\\d+$/', 'bn')).toBe(false);
  });
});

describe('model round-trip', () => {
  it('preserves the graph structure through decode → encode', () => {
    const model = buildBatchNormModel();
    const bytes = encodeModel(model);
    const decoded = decodeModel(bytes);
    expect(decoded.graph.node.map((node) => node.opType)).toEqual(
      model.graph.node.map((node) => node.opType),
    );
    expect(decoded.graph.initializer.map((tensor) => tensor.name)).toEqual(
      model.graph.initializer.map((tensor) => tensor.name),
    );
    for (const tensor of decoded.graph.initializer) {
      const original = model.graph.initializer.find((entry) => entry.name === tensor.name)!;
      expect(tensor.rawData?.byteLength).toBe(original.rawData?.byteLength);
    }
  });
});
