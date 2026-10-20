# FAQ

> English · [简体中文](zh-CN/faq.md)

### Does it need Python, onnxruntime or a GPU?

No. Conversion is pure TypeScript over the protobuf bytes. Only the optional
`verify` command needs `onnxruntime-node`, and it is not a dependency.

### How is this different from `onnxconverter-common` / ORT's fp16 tool?

They convert; this one also **checks whether the conversion is representable**
and repairs the one common case that is not (BatchNorm `running_var` above
65,504). They clamp silently — `onnxconverter-common` at 10,000 by default —
which produced a −13.9 dB regression in our model. It also blocks ops that have
no fp16 kernels, so the converted graph still loads.

### Can I use it with `onnxruntime-web`?

Yes, that is the main target. The converter runs in Node/CI; the resulting model
loads in `onnxruntime-web`. See [docs/browser.md](browser.md).

### Will fp16 make inference faster?

On WebGPU, CUDA, CoreML and DirectML: usually yes, roughly in proportion to the
bandwidth reduction. On WASM/CPU: no — onnxruntime-web casts back to fp32
internally, and we measured a 1.5× slowdown. Keep an fp32 model for CPU-only
clients.

### Can it run in the browser?

Yes. `convertBytes()` uses only typed arrays. Budget ~3× the model size in peak
memory; use a Web Worker for models above ~100 MB.

### What happens with a model that does not fit in fp16?

- `--overflow=rescale` (default): BatchNorm variances are repaired; any _other_
  overflowing tensor aborts the conversion with a list.
- `--overflow=block`: the affected nodes stay fp32, with casts around them.
- `--overflow=error`: abort immediately. This is the CI mode.

Nothing is ever clamped without telling you.

### Why 40,000 as the rescale target instead of 65,504?

Because `var` goes under a square root and the margin costs nothing: `K` must be
a power of two either way. It also leaves headroom for exporters whose
statistics differ slightly from the ones we measured.

### Does it handle int8 or 4-bit quantization?

No. Different problem: int8 needs calibration data and error accumulation
analysis. This package is deliberately fp16-only.

### My model uses `If`/`Loop`. Is it supported?

Partially. Subgraphs are not traversed in v0.1 — those tensors stay fp32, so the
model still works, but the size win is smaller. Tracked in
[ROADMAP.md](../ROADMAP.md).

### The converted file is _larger_ than the original. Why?

For models whose weights are a small fraction of the file, the inserted `Cast`
nodes and protobuf overhead can outweigh the payload savings. `inspect` prints
the projected payload so you can check before converting.

### Is the JSON output stable?

Yes — the fields printed by `inspect --json` and `convert --json` are part of the
public API and follow semver (additions are minor, removals are major).

### How do I cite it?

See [CITATION.cff](../CITATION.cff).
