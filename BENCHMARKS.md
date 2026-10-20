# Benchmarks

Everything here is reproducible on a laptop with the model file. No GPU cluster,
no hidden test set.

## LaMa (inpainting, 51 M parameters)

Model: `lama_fp32.onnx`, 208,044,816 bytes, 17,480 nodes, opset 17.
Hardware for the timings: Apple M-series, Node 25, 16 GB RAM.

### Conversion

```bash
npm run build
node scripts/bench.mjs /path/to/lama_fp32.onnx
```

```
lama_fp32.onnx
  198.4 MiB → 101.2 MiB  (−49%)
  inspect 107 ms · convert 680 ms
  16 BatchNorm rescale(s) · 108 blocked node(s) · 0 fp32 tensor(s) · 111 cast(s)
```

| Metric                                |                                     Value |
| ------------------------------------- | ----------------------------------------: |
| Input                                 |                 208,044,816 B (198.4 MiB) |
| Output                                |                 106,087,355 B (101.2 MiB) |
| Reduction                             |                                    49.0 % |
| `inspect` time                        |                                    107 ms |
| `convert` (decode + convert + encode) |                                    680 ms |
| Initializers converted                |                                 606 / 606 |
| `Constant` tensors converted          | 148 / 364 (the rest feed blocked `Range`) |
| BatchNorm layers rescaled             |                                  16 of 75 |
| Nodes kept in fp32                    |                   108 float `Range` (FFT) |
| Cast nodes inserted                   |                                       111 |
| Peak RSS                              |   ~1.0 GB (decoded + converted + encoded) |

The fp16 tensor payload is **101.98 MB**, byte-for-byte the same size as the
production model ClearPix ships.

### Quality

14 real images with ground-truth masks, masked-region PSNR (higher is better):

| Variant                     | Masked-region PSNR | vs fp32   |
| --------------------------- | -----------------: | --------- |
| fp32                        |           29.74 dB | —         |
| fp16, naive (clamped)       |           15.81 dB | −13.93 dB |
| fp16, rescaled (this tool)  |           29.73 dB | −0.01 dB  |
| fp16, 16 BatchNorms blocked |           29.73 dB | −0.01 dB  |

Output-level agreement with fp32 on the same inputs (14-image range):
**73.0–73.1 dB** for both repair strategies.

### Reproducing the agreement check on any machine with Python

```bash
python -m pip install onnxruntime numpy
python scripts/psnr.py lama_fp32.onnx lama.fp16.onnx
```

```
output[0] PSNR 37.58 dB · max|Δ| 7.610e+01 · RMSE 1.0e+01 · cosine 0.9997343
PASS: worst PSNR 37.58 dB (threshold 30.00 dB)
```

(Random input is the worst case for fp16 — real images land at 73 dB. Both the
converter's output and the shipped fp16 model measure 37.58 / 37.59 dB here,
which is the point: they are the same model.)

### Runtime (browser)

LaMa, 512×512 tile, onnxruntime-web, Apple silicon:

| Path             | Time / tile |
| ---------------- | ----------: |
| WASM, fp32       |      2.75 s |
| WASM, fp16       |      4.06 s |
| **WebGPU, fp16** |  **0.86 s** |

## Small models

For models without BatchNorm statistics near the limit, the conversion is
essentially `size / 2`:

| Model                  |         fp32 |         fp16 | Rescaled | Blocked | Notes                    |
| ---------------------- | -----------: | -----------: | -------: | ------: | ------------------------ |
| `realesr-general-x4v3` |  4,866,417 B |  2,440,026 B |        0 |       1 | `Resize`; no overflow    |
| `ppocr_v5_mobile_det`  |  4,748,769 B |  2,407,737 B |        0 |       6 | `Resize`/`TopK` pipeline |
| `migan`                | 29,541,096 B | 14,800,800 B |        0 |      14 | 14 tensors kept fp32     |
| `real_esrgan_x4`       | 67,051,616 B | 33,654,592 B |        0 |       2 | no overflow              |

All four load in ONNX Runtime 1.30 and produce fp32 outputs; the conversion is
−49 to −50 % in every case.

## Methodology

- Sizes are file sizes on disk; conversion adds a handful of `Cast` nodes, so a
  model whose weights are a tiny fraction of the file may grow slightly.
- Timings are wall-clock, single run, warm file cache. They are meant as orders
  of magnitude, not a competitive benchmark.
- Quality numbers come from the ClearPix lab run; raw JSON is in
  [`lab/lama-quant/results`](https://github.com/SharksevenRo/clearpix/tree/main/lab/lama-quant/results).

## Adding your own results

Run the benchmark with `--json` and open a PR with the numbers (model name,
license, size, hardware). Models with an interesting overflow story are
especially welcome — they make the guard better.
