import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  float16BitsToFloat32,
  float16ToFloat32,
  float32ToFloat16,
  float32ToFloat16Bits,
} from '../src/fp16.js';

function fromBits(bits: number): number {
  return new Float32Array(new Uint32Array([bits]).buffer)[0];
}

interface Vector {
  f32bits: number;
  fp16bits: number;
}

const fixture = JSON.parse(
  readFileSync(new URL('./fixtures/fp16-vectors.json', import.meta.url), 'utf8'),
) as { generatedBy: string; vectors: Vector[] };

describe('float32 → float16', () => {
  it('matches NumPy on 267 reference vectors (RNE, subnormals, overflow, NaN)', () => {
    let checked = 0;
    for (const vector of fixture.vectors) {
      const value = fromBits(vector.f32bits);
      const bits = float32ToFloat16Bits(value);
      if (Number.isNaN(value)) {
        // Any quiet NaN is acceptable; the sign/payload need not match NumPy.
        expect((bits & 0x7c00) === 0x7c00 && (bits & 0x3ff) !== 0).toBe(true);
        continue;
      }
      expect(bits, `value=${value} (0x${vector.f32bits.toString(16)})`).toBe(vector.fp16bits);
      checked += 1;
    }
    expect(checked).toBeGreaterThan(250);
  });

  it('rounds ties to even', () => {
    // 1 + 2^-11 is exactly halfway between 1 and 1 + 2^-10.
    expect(float32ToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00);
    // 1 + 3 * 2^-11 is halfway between 1 + 2^-10 and 1 + 2 * 2^-10 → rounds up.
    expect(float32ToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02);
  });

  it('handles the fp16 range edges', () => {
    expect(float32ToFloat16Bits(65504)).toBe(0x7bff);
    expect(float32ToFloat16Bits(65519)).toBe(0x7bff);
    expect(float32ToFloat16Bits(65520)).toBe(0x7c00); // → Infinity
    expect(float32ToFloat16Bits(Number.POSITIVE_INFINITY)).toBe(0x7c00);
    expect(float32ToFloat16Bits(Number.NEGATIVE_INFINITY)).toBe(0xfc00);
    expect(float32ToFloat16Bits(-0)).toBe(0x8000);
  });

  it('flushes values below the subnormal range to zero', () => {
    expect(float32ToFloat16Bits(2 ** -24)).toBe(0x0001);
    expect(float32ToFloat16Bits(2 ** -25)).toBe(0x0000); // tie → even (zero)
    expect(float32ToFloat16Bits(-(2 ** -25))).toBe(0x8000);
    expect(float32ToFloat16Bits(1e-8)).toBe(0x0000);
  });

  it('converts arrays', () => {
    const source = Float32Array.from([0, 1, -2, 65504, 1e-8]);
    const bits = float32ToFloat16(source);
    expect(Array.from(bits)).toEqual([0x0000, 0x3c00, 0xc000, 0x7bff, 0x0000]);
    expect(Array.from(float16ToFloat32(bits))).toEqual([0, 1, -2, 65504, 0]);

    const out = new Uint16Array(5);
    expect(float32ToFloat16(source, out)).toBe(out);
    expect(() => float32ToFloat16(source, new Uint16Array(2))).toThrow(/too small/);
  });
});

describe('float16 → float32', () => {
  it('decodes every representable value exactly', () => {
    for (let bits = 0; bits < 0x10000; bits += 1) {
      const value = float16BitsToFloat32(bits);
      const exponent = (bits >>> 10) & 0x1f;
      const mantissa = bits & 0x3ff;
      if (exponent === 0x1f) {
        expect(Number.isNaN(value) || Math.abs(value) === Number.POSITIVE_INFINITY).toBe(true);
        continue;
      }
      if (Number.isNaN(value)) continue;
      // Decoding is exact, so re-encoding must return the same bit pattern.
      expect(float32ToFloat16Bits(value), `bits=0x${bits.toString(16)} value=${value}`).toBe(bits);
      if (exponent === 0 && mantissa === 0) expect(value).toBe(bits === 0x8000 ? -0 : 0);
    }
  });
});
