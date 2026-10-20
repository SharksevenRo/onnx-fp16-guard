# Why fp16 breaks models (and why nobody tells you)

> English · [简体中文](zh-CN/why-fp16-breaks.md)

> This document is the technical background for the guard implemented in
> [`src/bn.ts`](../src/bn.ts). The numbers come from a real model — LaMa, 208 MB,
> 17,480 nodes — see [case-study-lama.md](case-study-lama.md).

## 1. float16 is not "float32 with fewer decimals"

binary16 spends 1 bit on sign, 5 on the exponent and 10 on the mantissa:

| Property                   | float32     | float16     |
| -------------------------- | ----------- | ----------- |
| Largest finite value       | 3.4 × 10³⁸  | **65,504**  |
| Smallest normal            | 1.2 × 10⁻³⁸ | 6.1 × 10⁻⁵  |
| Smallest subnormal         | 1.4 × 10⁻⁴⁵ | 5.96 × 10⁻⁸ |
| Significant decimal digits | ~7.2        | ~3.3        |
| Machine epsilon            | 1.19 × 10⁻⁷ | 9.77 × 10⁻⁴ |

The exponent range shrinks by **2²⁴** (from 8 bits to 5). Converting a value
above 65,504 does not round it — the nearest representable value is `Infinity`.
Most tools avoid the `Infinity` by _clamping_ to 65,504 instead, which is
arguably worse: the number looks finite and plausible, so nothing downstream
notices.

## 2. Where large values hide

Weights are usually small: in LaMa, `p50 = 0.43`, `p99 = 2.52`. But BatchNorm
layers store **running statistics**, not weights:

```
y = (x − μ) / sqrt(var + ε) · γ + β
```

`var` is the variance of the layer input accumulated during training. In a
network trained with large activations it is entirely normal for `var` to reach
10⁵–10⁶. In LaMa:

```
max|weight| = 484,322.56        (fp16 limit: 65,504)
tensors above the limit: 16 of 606 — all BatchNorm running_var, 2,176 numbers
```

Because `var` sits under a square root, a clamped `var` does not produce a small
error. It produces a **systematic scale error** on that layer:

| Conversion                               | stored max\|var\| | effect on that layer            |
| ---------------------------------------- | ----------------: | ------------------------------- |
| fp32 reference                           |        484,322.56 | —                               |
| clamp at 65,504 (`onnxconverter-common`) |         65,504.00 | `1/√var` is 2.7× too large      |
| bare `astype(float16)`                   |               `∞` | `√∞ = ∞` → output is constant β |

Sixteen broken layers out of 75 is enough to destroy the output while leaving
the file structurally valid, loadable, and plausible-looking.

## 3. The evidence that precision was never the problem

If fp16 failed because of 10-bit mantissas, then reducing precision _without_
changing the type should hurt just as much. It does not:

| Variant (still float32)         | Mantissa | Masked-region PSNR | Fidelity vs fp32 |
| ------------------------------- | -------: | -----------------: | ---------------: |
| `fp32_w13`                      |  13 bits |           28.88 dB |    95.9–109.0 dB |
| `fp32_w16` (= fp16 precision)   |  10 bits |           28.88 dB |     74.5–96.6 dB |
| `fp32_wbf16` (= bf16 precision) |   7 bits |           28.83 dB |     66.9–76.9 dB |
| fp32 baseline                   |  23 bits |           28.88 dB |                — |

Cutting the mantissa to **7 bits** costs 0.05 dB. The model is precision-
insensitive and range-sensitive. Any tool that only thinks about precision will
miss the bug.

## 4. Why every converter is silent

- `onnxconverter-common.float16.convert_float_to_float16` clips values with
  `max_finite_val`, which defaults to **10,000** — a 48× error on LaMa's
  variances — and logs nothing.
- `onnxruntime.tools.convert_onnx_models_to_float16` uses the fp16 maximum
  (65,504) and also logs nothing.
- `model.astype(np.float16)` / `TensorProto` casts produce `inf` and emit a
  NumPy warning at best.

The conversion is "successful" in every case: the file is valid, the graph
type-checks, the session loads. The failure is purely numeric, which is exactly
the class of failure that is expensive to catch after shipping.

## 5. The repair

For a BatchNorm layer, rescaling the variance by a power of two is exact:

```
y = (x − μ) / √(var + ε) · γ + β
  = (x − μ) / √(var/K² + ε/K²) · (γ/K) + β
```

Choose `K = 2^⌈log₂(max|var| / 40000)⌉`. Then:

- `var/K²` and `γ/K` are exponent shifts in binary32 — no rounding is introduced
  by the repair itself;
- the layer computes the same function for every input;
- the graph gains **zero** nodes (no Casts, no fp32 islands).

For LaMa, `K = 16`: `484,322 → 1,892`, `γ → γ/16`, `ε → ε/256`.

The alternative — keeping those 16 layers in fp32 — is also correct and is
available as `--overflow=block`; it costs 34 KB of fp32 parameters plus Cast
nodes.

## 6. Detecting it yourself

```ts
import { decodeModel, inspectModel } from 'onnx-fp16-guard';

const report = inspectModel(decodeModel(bytes));
for (const finding of report.overflow.findings) {
  console.log(finding.tensor, finding.absMax, finding.fix);
}
```

`inspectModel` scans initializers **and** `Constant` node payloads, so tensors
embedded in the graph are covered too. Wire it into CI:

```bash
onnx-fp16-guard inspect model.onnx --fail-on-overflow
```

Exit code 2 means "this model cannot be represented in fp16 without a decision".
That is the guard: a decision, not a clamp.

## Further reading

- [Case study: LaMa](case-study-lama.md)
- [How the converter works](how-it-works.md)
- ONNX Runtime float16 tooling:
  <https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers>
- `onnxconverter-common`: <https://github.com/microsoft/onnxconverter-common>
