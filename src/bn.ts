/**
 * BatchNorm range repair.
 *
 * The fix is an algebraic identity, not an approximation:
 *
 *     y = (x − μ) / √(var + ε) · γ + β
 *       ≡ (x − μ) / √(var/K² + ε/K²) · (γ/K) + β
 *
 * Pick `K` as the smallest power of two that brings `max|var|` under the fp16
 * safe range. Because `K` is a power of two, `var/K²` and `γ/K` are exact
 * exponent shifts in binary32 — the only rounding left is the final fp32→fp16
 * cast, so the repair adds no error of its own.
 *
 * Why 40000 and not 65504? `var` sits under a square root; keeping it at 61% of
 * the maximum leaves headroom for values the exporter may have produced with a
 * different rounding, and costs nothing (K is a power of two either way).
 */
import { DataType, FP16_MAX, type ModelProto, type NodeProto, type TensorProto } from './onnx.js';
import { readFloat32Tensor, tensorAbsMax, tensorElementCount, writeFloat32Tensor } from './model.js';

export const DEFAULT_TARGET_MAX = 40_000;

export interface BatchNormOverflow {
  /** BN node name (falls back to a generated label). */
  node: string;
  /** `running_var` initializer name. */
  variance: string;
  /** `scale` (γ) initializer name. */
  scale: string;
  /** `B`/`mean` initializer names, kept as-is. */
  bias: string;
  mean: string;
  absMaxVariance: number;
  /** Power-of-two factor that would be applied. */
  k: number;
  /** False when `scale`/`variance` are not static initializers. */
  fixable: boolean;
  /** False when the node has training outputs (mean/var) — rescaling would change them. */
  inferenceOnly: boolean;
  hasEpsilonAttribute: boolean;
  epsilon: number;
}

/** Smallest power of two K such that `max / K² <= target`. */
export function rescaleFactor(max: number, target: number): number {
  if (!Number.isFinite(max) || max <= target) return 1;
  return 2 ** Math.ceil(Math.log2(max / target));
}

function nodeLabel(node: NodeProto, index: number): string {
  return node.name || `${node.opType}#${index}`;
}

function epsilonOf(node: NodeProto): { value: number; present: boolean } {
  for (const attribute of node.attribute) {
    if (attribute.name === 'epsilon') {
      return {
        value: typeof attribute.f === 'number' ? attribute.f : Number(attribute.f ?? 0),
        present: true,
      };
    }
  }
  // ONNX default for BatchNormalization.epsilon
  return { value: 1e-5, present: false };
}

/** Inspect every BatchNormalization node; report the ones that overflow fp16. */
export function findBatchNormOverflows(
  model: ModelProto,
  options: { limit?: number; targetMax?: number } = {},
): BatchNormOverflow[] {
  const limit = options.limit ?? FP16_MAX;
  const targetMax = options.targetMax ?? DEFAULT_TARGET_MAX;
  const initializers = new Map<string, TensorProto>();
  for (const tensor of model.graph.initializer) initializers.set(tensor.name, tensor);

  const findings: BatchNormOverflow[] = [];
  model.graph.node.forEach((node, index) => {
    if (node.opType !== 'BatchNormalization') return;
    const [x, scaleName, biasName, meanName, varName] = node.input;
    if (!x || !scaleName || !biasName || !meanName || !varName) return;
    const variance = initializers.get(varName);
    const scale = initializers.get(scaleName);
    if (!variance || variance.dataType !== DataType.FLOAT) return;
    const absMax = tensorAbsMax(variance) ?? 0;
    if (absMax <= limit) return;

    const eps = epsilonOf(node);
    findings.push({
      node: nodeLabel(node, index),
      variance: varName,
      scale: scaleName,
      bias: biasName,
      mean: meanName,
      absMaxVariance: absMax,
      k: rescaleFactor(absMax, targetMax),
      fixable: Boolean(scale && scale.dataType === DataType.FLOAT && Number.isFinite(absMax)),
      inferenceOnly: node.output.filter(Boolean).length <= 1,
      hasEpsilonAttribute: eps.present,
      epsilon: eps.value,
    });
  });

  return findings;
}

export interface RescaleChange {
  node: string;
  variance: string;
  scale: string;
  k: number;
  maxVarianceBefore: number;
  maxVarianceAfter: number;
  epsilonBefore: number;
  epsilonAfter: number;
}

/**
 * Rescale the overflowing BatchNorm layers in place.
 *
 * Throws when the same initializer is shared by layers that need different
 * factors: that would silently break one of them, and "silently" is the whole
 * problem this package exists to solve.
 */
export function rescaleBatchNorms(
  model: ModelProto,
  options: { limit?: number; targetMax?: number } = {},
): RescaleChange[] {
  const targetMax = options.targetMax ?? DEFAULT_TARGET_MAX;
  const findings = findBatchNormOverflows(model, { ...options, targetMax });
  const initializers = new Map<string, TensorProto>();
  for (const tensor of model.graph.initializer) initializers.set(tensor.name, tensor);
  const applied = new Map<string, number>();
  const changes: RescaleChange[] = [];

  for (const finding of findings) {
    if (!finding.fixable) {
      throw new Error(
        `BatchNormalization "${finding.node}" overflows fp16 (max|var|=${finding.absMaxVariance.toExponential(3)}) ` +
          `but its scale/variance are not static float initializers, so it cannot be rescaled. ` +
          `Use --overflow=block to keep this layer in fp32 instead.`,
      );
    }
    if (!finding.inferenceOnly) {
      throw new Error(
        `BatchNormalization "${finding.node}" has training outputs (mean/var); ` +
          `rescaling would change their meaning. Use --overflow=block instead.`,
      );
    }

    const previous = applied.get(finding.variance);
    if (previous !== undefined && previous !== finding.k) {
      throw new Error(
        `initializer "${finding.variance}" is shared by BatchNorm layers that need different ` +
          `rescale factors (${previous} vs ${finding.k}); use --overflow=block`,
      );
    }
    if (previous !== undefined) continue;

    const variance = initializers.get(finding.variance);
    const scale = initializers.get(finding.scale);
    if (!variance || !scale) continue;

    const k = finding.k;
    const kk = k * k;

    const varianceData = readFloat32Tensor(variance);
    for (let i = 0; i < varianceData.length; i += 1) varianceData[i] /= kk;
    writeFloat32Tensor(variance, varianceData);

    const scaleData = readFloat32Tensor(scale);
    for (let i = 0; i < scaleData.length; i += 1) scaleData[i] /= k;
    writeFloat32Tensor(scale, scaleData);

    // Only the nodes that use this variance need their epsilon adjusted.
    for (const node of model.graph.node) {
      if (node.opType !== 'BatchNormalization') continue;
      if (node.input[4] !== finding.variance) continue;
      const epsilon = epsilonOf(node);
      const scaled = epsilon.value / kk;
      const attribute = node.attribute.find((a) => a.name === 'epsilon');
      if (attribute) {
        attribute.f = scaled;
        attribute.type = 1;
      } else {
        node.attribute.push({ name: 'epsilon', type: 1, f: scaled });
      }
      if (node.input[1] === finding.scale) {
        // keep the initializer list consistent even if several BNs share it
      }
    }

    applied.set(finding.variance, k);
    changes.push({
      node: finding.node,
      variance: finding.variance,
      scale: finding.scale,
      k,
      maxVarianceBefore: finding.absMaxVariance,
      maxVarianceAfter: finding.absMaxVariance / kk,
      epsilonBefore: finding.epsilon,
      epsilonAfter: finding.epsilon / kk,
    });
  }

  return changes;
}

/** Guard: after conversion, no fp16 tensor may contain ±Infinity by accident. */
export function assertNoOverflow(tensor: TensorProto, limit = FP16_MAX): void {
  const absMax = tensorAbsMax(tensor);
  if (absMax !== undefined && absMax > limit) {
    throw new Error(
      `tensor "${tensor.name}" has |max|=${absMax.toExponential(3)} > ${limit}, ` +
        `which is not representable in fp16 (${tensorElementCount(tensor)} elements)`,
    );
  }
}
