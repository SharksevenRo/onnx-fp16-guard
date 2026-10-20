import { describe, expect, it } from 'vitest';
import { findBatchNormOverflows, rescaleBatchNorms, rescaleFactor } from '../src/bn.js';
import { decodeModel, readFloat32Tensor } from '../src/model.js';
import { batchNormForward, buildBatchNormModel, serialize } from './helpers/models.js';

function loadFixture(): ReturnType<typeof decodeModel> {
  return decodeModel(serialize(buildBatchNormModel()));
}

describe('rescaleFactor', () => {
  it('picks the smallest power of two that fits the target', () => {
    expect(rescaleFactor(40_000, 40_000)).toBe(1);
    expect(rescaleFactor(40_001, 40_000)).toBe(2);
    expect(rescaleFactor(65_504, 40_000)).toBe(2);
    expect(rescaleFactor(160_000, 40_000)).toBe(4);
    expect(rescaleFactor(484_322, 40_000)).toBe(16);
    expect(rescaleFactor(1_000_000, 40_000)).toBe(32);
  });

  it('is a no-op for non-finite input', () => {
    expect(rescaleFactor(Number.POSITIVE_INFINITY, 40_000)).toBe(1);
  });
});

describe('findBatchNormOverflows', () => {
  it('finds only the BatchNorm whose running_var exceeds fp16', () => {
    const model = loadFixture();
    const findings = findBatchNormOverflows(model);
    expect(findings).toHaveLength(1);
    expect(findings[0]).toMatchObject({
      node: 'bn1',
      variance: 'bn1.var',
      scale: 'bn1.scale',
      k: 16,
      fixable: true,
      inferenceOnly: true,
      hasEpsilonAttribute: true,
    });
  });

  it('reports nothing for a model that already fits', () => {
    const model = decodeModel(serialize(buildBatchNormModel({ varianceMax: 1000 })));
    expect(findBatchNormOverflows(model)).toHaveLength(0);
  });
});

describe('rescaleBatchNorms', () => {
  it('applies the algebraic identity exactly (power-of-two scaling)', () => {
    const model = loadFixture();
    const before = model.graph.initializer.find((tensor) => tensor.name === 'bn1.var');
    const beforeValues = readFloat32Tensor(before!);
    const scaleBefore = readFloat32Tensor(model.graph.initializer.find((t) => t.name === 'bn1.scale')!);

    const changes = rescaleBatchNorms(model);
    expect(changes).toHaveLength(1);
    expect(changes[0].k).toBe(16);

    const after = readFloat32Tensor(model.graph.initializer.find((tensor) => tensor.name === 'bn1.var')!);
    const scaleAfter = readFloat32Tensor(model.graph.initializer.find((t) => t.name === 'bn1.scale')!);

    for (let i = 0; i < after.length; i += 1) {
      expect(after[i]).toBeCloseTo(beforeValues[i] / 256, 6);
      expect(scaleAfter[i]).toBeCloseTo(scaleBefore[i] / 16, 6);
    }

    const epsilon = model.graph.node[0].attribute.find((attribute) => attribute.name === 'epsilon');
    expect(epsilon?.f).toBeCloseTo(1e-5 / 256, 12);
  });

  it('keeps the layer numerically identical', () => {
    const model = loadFixture();
    const input = Float32Array.from({ length: 3 * 4 * 4 }, (_, index) => Math.sin(index * 1.7) * 3);

    const read = (name: string) => readFloat32Tensor(model.graph.initializer.find((t) => t.name === name)!);
    const reference = batchNormForward(
      input,
      3,
      read('bn1.scale'),
      read('bn1.bias'),
      read('bn1.mean'),
      read('bn1.var'),
      1e-5,
    );

    rescaleBatchNorms(model);
    const rescaled = batchNormForward(
      input,
      3,
      read('bn1.scale'),
      read('bn1.bias'),
      read('bn1.mean'),
      read('bn1.var'),
      1e-5 / 256,
    );

    let maxDiff = 0;
    for (let i = 0; i < reference.length; i += 1) {
      maxDiff = Math.max(maxDiff, Math.abs(reference[i] - rescaled[i]));
    }
    expect(maxDiff).toBeLessThan(1e-5);
  });

  it('adds a scaled epsilon attribute when the node has none', () => {
    const model = loadFixture();
    // Force the second (normally-scaled) BatchNorm to overflow as well.
    const variance = model.graph.initializer.find((tensor) => tensor.name === 'bn2.var')!;
    variance.rawData = new Uint8Array(Float32Array.from([9e5, 1, 1]).buffer);

    rescaleBatchNorms(model);
    const bn2 = model.graph.node.find((node) => node.name === 'bn2')!;
    const epsilon = bn2.attribute.find((attribute) => attribute.name === 'epsilon');
    expect(epsilon).toBeDefined();
    expect(epsilon?.type).toBe(1);
    // K = 32 for 9e5 → epsilon scales by 1/1024.
    expect(epsilon?.f).toBeCloseTo(1e-5 / 1024, 14);
  });
});
