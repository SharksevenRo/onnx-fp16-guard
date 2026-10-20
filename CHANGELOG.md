# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pre-bundled browser build: `onnx-fp16-guard/browser` (and `dist/browser.js`)
  ships the library with `onnx-proto` inlined, so a page can convert models
  without a bundler, an import map or a CDN.
- GitHub Pages workflow that publishes the browser example as a static site
  (`site/index.html` landing page + `examples/browser/` + `dist/browser.js`).
- Simplified Chinese documentation: `README.zh-CN.md`, `BENCHMARKS.zh-CN.md` and
  `docs/zh-CN/` (why fp16 breaks, how it works, CLI, API, browser, LaMa case
  study, FAQ).

## [0.1.0] — 2026-10-20

First public release, extracted from the fp16 pipeline of the
[ClearPix](https://clearpix.org) browser photo editor.

### Added

- `inspectModel()` — read-only analysis of dtypes, storage, op histogram, I/O and
  every tensor outside the fp16 range, including payloads embedded in
  `Constant` nodes.
- BatchNorm overflow repair: power-of-two rescale of `running_var`, `scale` and
  `epsilon` (`--overflow=rescale`, the default), fp32 freezing
  (`--overflow=block`), or a hard failure (`--overflow=error`).
- Bit-exact float32 → float16 conversion (round-to-nearest-even, subnormals,
  overflow to infinity) with no runtime dependency.
- Graph surgery: retargeting of existing `Cast(to=float32)` nodes, fp16 ⇄ fp32
  boundaries around blocked nodes, int64/bool-aware cast placement, stable
  topological sort, structural validation.
- Default fp32 block list for ops without fp16 kernels in ONNX Runtime
  (`Range`, `Resize`, `NonMaxSuppression`, `RoiAlign`, `CumSum`, `Upsample`,
  `TopK`).
- CLI (`onnx-fp16-guard`, alias `onnx-fp16`) with `inspect`, `convert` and
  `verify`, JSON reports and meaningful exit codes.
- Optional numerical verification through `onnxruntime-node`
  (`verifyModels()` / `verify`).
- Browser-safe core; drag-and-drop example in `examples/browser`.
- Documentation: [why fp16 breaks](docs/why-fp16-breaks.md),
  [how it works](docs/how-it-works.md), [CLI](docs/cli.md), [API](docs/api.md),
  [browser guide](docs/browser.md), [LaMa case study](docs/case-study-lama.md),
  [benchmarks](BENCHMARKS.md).

### Validation

- 267 NumPy-generated reference vectors plus an exhaustive round-trip over all
  65,536 fp16 bit patterns.
- LaMa 208 MB: 606/606 initializers converted, 16 BatchNorms rescaled, 49 %
  smaller, loads and runs in ONNX Runtime 1.30, agrees with the shipped fp16
  model at 66.3 dB (masked region) / 76.7 dB (full frame).
- Four additional real models (ESRGAN ×2, PP-OCR, MI-GAN) convert, load and run.

[Unreleased]: https://github.com/SharksevenRo/onnx-fp16-guard/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SharksevenRo/onnx-fp16-guard/releases/tag/v0.1.0
