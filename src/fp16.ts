/**
 * IEEE-754 binary32 ⇄ binary16 conversion.
 *
 * These primitives are deliberately allocation-free per element: quantizing a
 * 200 MB model is ~51 M values, so a `Float32Array` is written element by
 * element into a `Uint16Array` with a single shared scratch buffer.
 *
 * The conversion is round-to-nearest, ties-to-even — the same rule as NumPy's
 * `astype(np.float16)` and IEEE-754 `convertToFloat16`, which is what ONNX
 * runtimes expect. Values above {@link FP16_MAX} become `Infinity` (this is the
 * silent failure mode this package exists to prevent — see `analyze.ts`).
 */

const scratchF32 = new Float32Array(1);
const scratchU32 = new Uint32Array(scratchF32.buffer);

/** Convert one float32 value to its 16-bit half-precision representation. */
export function float32ToFloat16Bits(value: number): number {
  scratchF32[0] = value;
  const x = scratchU32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;

  if (exp === 0xff) {
    // Infinity or NaN. Keep NaN quiet and non-zero payload.
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  }

  const e = exp - 127 + 15;

  if (e >= 0x1f) return sign | 0x7c00; // overflow → Infinity
  if (e <= 0) {
    if (e < -10) return sign; // underflow → signed zero
    const shift = 14 - e; // 14 … 24
    const full = mant | 0x800000;
    let m = full >>> shift;
    const rem = full & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && (m & 1) === 1)) m += 1;
    return sign | m;
  }

  let m = mant >>> 13;
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (m & 1) === 1)) {
    m += 1;
    if (m === 0x400) {
      m = 0;
      const next = e + 1;
      if (next >= 0x1f) return sign | 0x7c00;
      return sign | (next << 10) | m;
    }
  }
  return sign | (e << 10) | m;
}

/** Convert a half-precision bit pattern back to a double (exact for binary16). */
export function float16BitsToFloat32(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Convert a float32 buffer to half-precision bits. */
export function float32ToFloat16(src: Float32Array, out?: Uint16Array): Uint16Array {
  const dst = out ?? new Uint16Array(src.length);
  if (dst.length < src.length) {
    throw new RangeError(`output buffer too small: ${dst.length} < ${src.length}`);
  }
  for (let i = 0; i < src.length; i += 1) dst[i] = float32ToFloat16Bits(src[i]);
  return dst;
}

/** Convert half-precision bits back to float32 values. */
export function float16ToFloat32(src: Uint16Array, out?: Float32Array): Float32Array {
  const dst = out ?? new Float32Array(src.length);
  if (dst.length < src.length) {
    throw new RangeError(`output buffer too small: ${dst.length} < ${src.length}`);
  }
  for (let i = 0; i < src.length; i += 1) dst[i] = float16BitsToFloat32(src[i]);
  return dst;
}

/** Bytes required to store `elements` values at the given ONNX data type. */
export function bytesPerElement(dataType: number): number {
  switch (dataType) {
    case 1: // FLOAT
    case 6: // INT32
      return 4;
    case 7: // INT64
    case 11: // DOUBLE
      return 8;
    case 10: // FLOAT16
    case 16: // BFLOAT16
    case 5: // INT16
    case 4: // UINT16
      return 2;
    case 8: // STRING
      return 1;
    default:
      return 1;
  }
}
