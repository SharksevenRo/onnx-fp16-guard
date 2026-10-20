# Case study: LaMa, 208 MB → 102 MB without losing quality

> English · [简体中文](zh-CN/case-study-lama.md)

LaMa is a large-mask inpainting model: 51 M parameters, 17,480 nodes, an FFT
block (FFC) in almost every layer. We run it fully client-side in
[ClearPix](https://clearpix.org) — a browser photo editor —
so download size is a product feature, not an optimisation.

## The starting point

| Property           | Value                                                        |
| ------------------ | ------------------------------------------------------------ |
| File               | `lama_fp32.onnx`, 208,044,816 bytes (198 MiB)                |
| Initializers       | 606, all float32, 51.0 M parameters                          |
| Dominant structure | FFC layers: 17,022 of 17,480 nodes                           |
| Weight magnitude   | p50 0.43 · p90 1.24 · p99 2.52 · **max 484,322.56**          |
| Tensors > 65,504   | **16** — all BatchNorm `running_var`, 2,176 numbers in total |

## The failure

Converting with `onnxconverter-common` (or ORT's float16 tool) halved the file
and produced a model that loads, runs, and outputs garbage. On our 14-image
benchmark (ground-truth masks):

| Variant                         | Masked-region PSNR | Full-frame PSNR | Fidelity vs fp32 |      Size |
| ------------------------------- | -----------------: | --------------: | ---------------: | --------: |
| fp32 baseline                   |           29.74 dB |        43.23 dB |                — | 198.4 MiB |
| fp16, naive converter           |       **15.81 dB** |    **29.30 dB** |     27.6–28.6 dB | 102.8 MiB |
| fp16, rescaled `K=16`           |           29.73 dB |        43.23 dB |     **73.05 dB** | 102.8 MiB |
| fp16, 16 BatchNorms frozen fp32 |           29.73 dB |        43.23 dB |          73.1 dB | 102.8 MiB |

The naive conversion lost **13.9 dB in the masked region**. The repaired model
is 0.01 dB from fp32 — inside the noise of the benchmark.

Nothing about the _weights_ changed between those rows. Only 16 variance
tensors, rescaled by a power of two:

```
K = 2^⌈log₂(484,322 / 40,000)⌉ = 16
running_var: 484,322 → 1,892       gamma: γ/16       epsilon: ε/256
```

## What `onnx-fp16-guard` does on this model today

```
$ onnx-fp16-guard convert lama_fp32.onnx -o lama.fp16.onnx --report report.json
Converted
  198 MB → 101 MB (−97.2 MB, 49.0%)
  606 initializer(s) + 148 Constant tensor(s) → float16, 111 Cast node(s) inserted
  ratio 51.0% of the original file

BatchNorm rescale
node                                                        K  max|var| before    after
/generator/model/model.5/conv1/bn_l/BatchNormalization   2e+1          4.84e+5  1.89e+3
… 15 more

  16 of 75 BatchNormalization layer(s) need attention
```

Highlights from the report:

- 606/606 initializers converted; the fp16 payload is **101.98 MB**, identical to
  the model we ship in production.
- 16 BatchNorm layers rescaled; **zero fp32 nodes** for them (no casts, no
  runtime cost).
- 108 float `Range` nodes (part of the FFT machinery) kept in fp32 automatically —
  ONNX Runtime has no fp16 `Range` kernel, so an fp16 `Range` fails at load time
  with `Type 'tensor(float16)' of input parameter … of operator (Range) is invalid`.
  The other 72 `Range` nodes in the model are int64 position encodings; they are
  left untouched because they were never fp16 candidates.
- 148 of the 364 float `Constant` tensors (FFT bases) converted; the rest are
  consumed by blocked `Range` nodes and stay fp32.
- The whole conversion takes **under a second** for a 208 MB model.

Validated with ONNX Runtime 1.30 on identical inputs:

| Comparison (same 512² input)       |   Full frame | Masked region |
| ---------------------------------- | -----------: | ------------: |
| this converter vs fp32             |     37.58 dB |      27.20 dB |
| shipped fp16 model vs fp32         |     37.59 dB |      27.21 dB |
| **this converter vs shipped fp16** | **76.71 dB** |  **66.33 dB** |

Two observations. First, the new converter and the model we ship are
numerically the same model (66–77 dB agreement). Second, against fp32 both fp16
models differ noticeably _on pure random noise_ — that is the worst case for
fp16 accumulation and not what matters; on our 14 real images the masked-region
quality is 0.01 dB from fp32.

## Runtime

| Path                   | Time per 512² tile |
| ---------------------- | -----------------: |
| WASM, fp32 model       |             2.75 s |
| WASM, fp16 model       |             4.06 s |
| **WebGPU, fp16 model** |         **0.86 s** |

fp16 is a WebGPU feature. On WASM the model gets cast back internally, so it is
1.5× _slower_; ClearPix ships both files and picks at runtime. On WebGPU the same
model runs 4.7× faster than on WASM.

## Lessons that generalise

1. **Check range before precision.** 7-bit mantissas cost 0.05 dB here; one
   clamped variance cost 13.9 dB.
2. **BatchNorm statistics are the usual suspect.** They are the only common
   weights whose magnitude is tied to activation scale.
3. **Silent clamping is the real bug.** A converter that refuses (or reports)
   turns a week of debugging into a one-line fix.
4. **Some ops cannot run in fp16 at all** (`Range`, `Resize`, `TopK`, …). Keep
   them fp32 explicitly; do not discover it from a load-time type error.

## Source material

The original investigation, with all raw results, is in the ClearPix repository:

- [FINDINGS.md](https://clearpix.org/blob/main/lab/lama-quant/FINDINGS.md)
- [Blog: the overflowed tensor](https://clearpix.org/blob/main/blogs/13-2026-10-17-fp16-quantization-quality-loss-overflowed-tensor.md)
- [Blog: fp16 ONNX + WebGPU in the browser](https://clearpix.org/blob/main/blogs/03-2026-10-01-fp16-onnx-webgpu-browser-inference.md)
- [Blog: the WebGPU Slice off-by-one](https://clearpix.org/blob/main/blogs/14-2026-10-18-lama-webgpu-slice-off-by-one.md)
