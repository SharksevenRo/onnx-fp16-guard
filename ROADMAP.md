# Roadmap

Priorities for the next releases. Issues and PRs against any of these are
welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## 0.2 — coverage and confidence

- [ ] **Subgraph traversal** (`If`, `Loop`, `Scan` bodies) so models with
      control flow get a full conversion instead of a partial one.
- [ ] **Layer-wise report** in `inspect`: which layers consume the largest
      tensors and what blocks them.
- [ ] **`--dry-run`** for `convert` (report without writing bytes).
- [ ] **Golden-model CI**: a small e2e model checked against `onnxruntime-node`
      in CI, so converter changes are validated numerically, not only
      structurally.
- [ ] **`--min-psnr` gate in the library** (`convertAndVerify` convenience).

## 0.3 — mixed precision

- [ ] **Per-op precision policy** (e.g. keep attention `Softmax` and
      `LayerNormalization` in fp32 by default for transformer models).
- [ ] **Sensitivity probe**: run both models on a few random inputs and
      automatically block the layers whose removal of precision costs the most
      PSNR.
- [ ] **`--report-html`**: a self-contained page showing what changed, for
      reviewers who do not read protobufs.

## Later

- [ ] bfloat16 output (`bf16` has fp32's exponent range — the overflow problem
      disappears, at the cost of browser support).
- [ ] int8 post-training quantization with calibration data (separate package;
      different trade-offs).
- [ ] A "bad conversion" gallery: models that fail in interesting ways, with the
      guard that catches them.

## Non-goals

- Training-time quantization.
- Model conversion between frameworks (PyTorch → ONNX, etc.).
- A general ONNX editor. The graph surgery here exists to serve fp16 conversion,
  not to be a toolkit.
