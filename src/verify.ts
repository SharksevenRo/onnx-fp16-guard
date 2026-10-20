/**
 * Optional numerical verification.
 *
 * Structural checks prove the graph is still well-formed; they cannot prove the
 * fp16 model still computes the same function. For that you need a runtime, so
 * this module loads `onnxruntime-node` lazily and runs both models on the same
 * random input. It is an optional peer dependency: the converter itself stays
 * dependency-free and browser-safe.
 *
 * A healthy fp16 conversion of a CNN lands around 60–80 dB PSNR against fp32;
 * a model broken by a clamped BatchNorm variance lands around 10–20 dB. The
 * default 30 dB threshold catches the latter with a wide margin.
 */
import { decodeModel } from './model.js';

export interface VerifyOptions {
  /** Explicit feeds, overriding the generated ones (name → typed array). */
  feeds?: Record<string, Float32Array | Int32Array | BigInt64Array | Uint8Array>;
  /** Deterministic PRNG seed for generated inputs. Default 42. */
  seed?: number;
  /** Pass threshold in dB. Default 30. */
  minPsnrDb?: number;
  /** Amplitude of generated float inputs. Default 1. */
  amplitude?: number;
}

export interface OutputComparison {
  name: string;
  elements: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
  rmsDiff: number;
  psnrDb: number;
  cosineSimilarity: number;
}

export interface VerifyReport {
  outputs: OutputComparison[];
  passed: boolean;
  thresholdDb: number;
  worstPsnrDb: number;
  runtimeVersion: string;
}

interface OrtTensorLike {
  data: Float32Array | Int32Array | BigInt64Array | Uint8Array;
  dims: readonly number[];
  type: string;
}

interface OrtSessionLike {
  inputNames: string[];
  outputNames: string[];
  inputMetadata?: unknown;
  run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>>;
}

interface OrtModuleLike {
  InferenceSession: { create(pathOrBuffer: string | Uint8Array, options?: unknown): Promise<OrtSessionLike> };
  Tensor: new (
    type: string,
    data: Float32Array | Int32Array | BigInt64Array | Uint8Array,
    dims: number[],
  ) => OrtTensorLike;
  env?: { versions?: { common?: string } };
}

async function loadOrt(): Promise<OrtModuleLike> {
  try {
    // Dynamic + non-literal specifier: bundlers leave it alone, so the package
    // stays browser-safe and the dependency stays optional.
    const specifier = 'onnxruntime-node';
    return (await import(/* @vite-ignore */ specifier)) as unknown as OrtModuleLike;
  } catch {
    throw new Error(
      'verifyModels() needs onnxruntime-node. Install it as a dev dependency:\n' +
        '  npm install --no-save onnxruntime-node\n' +
        'The converter itself does not need it.',
    );
  }
}

/** Deterministic xorshift32 PRNG so failures are reproducible. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

function dimsOf(metadata: unknown): number[] | undefined {
  if (!metadata || typeof metadata !== 'object') return undefined;
  const record = metadata as Record<string, unknown>;
  const raw = record.shape ?? record.dims;
  if (!Array.isArray(raw)) return undefined;
  return raw.map((dim) => (typeof dim === 'number' ? dim : Number(dim)));
}

function typeOf(metadata: unknown): number | undefined {
  if (!metadata || typeof metadata !== 'object') return undefined;
  const record = metadata as Record<string, unknown>;
  const raw = record.type ?? record.elemType;
  if (typeof raw === 'number') return raw;
  if (typeof raw === 'string') {
    const normalized = raw.toLowerCase();
    if (normalized.includes('float') && !normalized.includes('16')) return 1;
    if (normalized.includes('float16')) return 10;
    if (normalized.includes('int64')) return 7;
    if (normalized.includes('int32')) return 6;
    if (normalized.includes('uint8')) return 2;
    if (normalized.includes('bool')) return 9;
  }
  return undefined;
}

function metadataFor(session: OrtSessionLike, name: string): unknown {
  const metadata = session.inputMetadata;
  if (Array.isArray(metadata)) {
    return metadata.find((entry) => (entry as { name?: string })?.name === name);
  }
  if (metadata && typeof metadata === 'object') {
    return (metadata as Record<string, unknown>)[name];
  }
  return undefined;
}

function makeFeed(
  ort: OrtModuleLike,
  name: string,
  metadata: unknown,
  random: () => number,
  amplitude: number,
): OrtTensorLike {
  const dims = (dimsOf(metadata) ?? [1]).map((dim) => (Number.isFinite(dim) && dim > 0 ? dim : 1));
  const count = dims.reduce((total, dim) => total * dim, 1);
  const elemType = typeOf(metadata);

  if (elemType === 7) {
    return new ort.Tensor('int64', new BigInt64Array(count), dims);
  }
  if (elemType === 6) {
    return new ort.Tensor('int32', new Int32Array(count), dims);
  }
  if (elemType === 2 || elemType === 9) {
    return new ort.Tensor(elemType === 2 ? 'uint8' : 'bool', new Uint8Array(count), dims);
  }
  const data = new Float32Array(count);
  for (let i = 0; i < count; i += 1) data[i] = (random() * 2 - 1) * amplitude;
  return new ort.Tensor('float32', data, dims);
}

function compare(reference: Float32Array, candidate: Float32Array): Omit<OutputComparison, 'name'> {
  const length = Math.min(reference.length, candidate.length);
  let maxAbsDiff = 0;
  let sumAbs = 0;
  let sumSquared = 0;
  let dot = 0;
  let normRef = 0;
  let normCandidate = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < length; i += 1) {
    const a = reference[i];
    const b = candidate[i];
    const diff = b - a;
    const abs = Math.abs(diff);
    if (abs > maxAbsDiff) maxAbsDiff = abs;
    sumAbs += abs;
    sumSquared += diff * diff;
    dot += a * b;
    normRef += a * a;
    normCandidate += b * b;
    if (a < min) min = a;
    if (a > max) max = a;
  }

  const mse = length > 0 ? sumSquared / length : 0;
  const range = Number.isFinite(max - min) && max > min ? max - min : 1;
  const psnrDb = mse === 0 ? Number.POSITIVE_INFINITY : 10 * Math.log10((range * range) / mse);
  const denominator = Math.sqrt(normRef) * Math.sqrt(normCandidate);

  return {
    elements: length,
    maxAbsDiff,
    meanAbsDiff: length > 0 ? sumAbs / length : 0,
    rmsDiff: Math.sqrt(mse),
    psnrDb,
    cosineSimilarity: denominator > 0 ? dot / denominator : 1,
  };
}

/**
 * Run the reference and candidate models on identical inputs and compare their
 * outputs.
 *
 * @param reference path or bytes of the fp32 model
 * @param candidate path or bytes of the converted model
 */
export async function verifyModels(
  reference: string | Uint8Array,
  candidate: string | Uint8Array,
  options: VerifyOptions = {},
): Promise<VerifyReport> {
  const ort = await loadOrt();
  const random = createRandom(options.seed ?? 42);
  const amplitude = options.amplitude ?? 1;
  const thresholdDb = options.minPsnrDb ?? 30;

  const [referenceSession, candidateSession] = await Promise.all([
    ort.InferenceSession.create(reference),
    ort.InferenceSession.create(candidate),
  ]);

  const feeds: Record<string, OrtTensorLike> = {};
  for (const name of referenceSession.inputNames) {
    const provided = options.feeds?.[name];
    if (provided) {
      const dims = dimsOf(metadataFor(referenceSession, name)) ?? [provided.length];
      const type =
        provided instanceof Float32Array ? 'float32' : provided instanceof BigInt64Array ? 'int64' : 'int32';
      feeds[name] = new ort.Tensor(type, provided, dims);
    } else {
      feeds[name] = makeFeed(ort, name, metadataFor(referenceSession, name), random, amplitude);
    }
  }

  const [referenceOutputs, candidateOutputs] = await Promise.all([
    referenceSession.run(feeds),
    candidateSession.run(feeds),
  ]);

  const outputs: OutputComparison[] = [];
  for (const name of referenceSession.outputNames) {
    const a = referenceOutputs[name];
    const b = candidateOutputs[name];
    if (!a || !b) continue;
    const comparison = compare(a.data as Float32Array, b.data as Float32Array);
    outputs.push({ name, ...comparison });
  }

  const worstPsnrDb = outputs.reduce(
    (worst, output) => Math.min(worst, output.psnrDb),
    Number.POSITIVE_INFINITY,
  );

  return {
    outputs,
    passed: outputs.length > 0 && worstPsnrDb >= thresholdDb,
    thresholdDb,
    worstPsnrDb,
    runtimeVersion: ort.env?.versions?.common ?? 'unknown',
  };
}

/** Decode a model just to read its declared input types (used by the CLI). */
export function describeInputs(
  bytes: Uint8Array,
): { name: string; elemType: number; dims: (number | string)[] }[] {
  const model = decodeModel(bytes);
  return model.graph.input.map((valueInfo) => {
    const tensorType = valueInfo.type?.tensorType;
    const dims: (number | string)[] = [];
    for (const dim of tensorType?.shape?.dim ?? []) {
      if (dim.dimParam) dims.push(dim.dimParam);
      else if (dim.dimValue !== undefined && dim.dimValue !== null) dims.push(Number(dim.dimValue));
      else dims.push('?');
    }
    return { name: valueInfo.name, elemType: tensorType?.elemType ?? 0, dims };
  });
}
