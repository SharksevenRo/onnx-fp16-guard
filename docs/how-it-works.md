# How it works

> English · [简体中文](zh-CN/how-it-works.md)

`convertModel()` is a single pass over a decoded `ModelProto`. Nothing is
approximated, and every decision is recorded in the returned report.

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐
│ 1. blocks   │──▶│ 2. overflow  │──▶│ 3. convert    │──▶│ 4. casts   │──▶│ 5. finish  │
│ ops + nodes │   │ rescale/block│   │ f32 → f16     │   │ boundaries │   │ sort+check │
└─────────────┘   └──────────────┘   └───────────────┘   └────────────┘   └────────────┘
```

## 1. Decide what must stay fp32

Three sources feed the "blocked" set:

1. **`DEFAULT_FP32_OPS`** — ops that have no fp16 kernel in ONNX Runtime:
   `Range`, `Resize`, `NonMaxSuppression`, `RoiAlign`, `CumSum`, `Upsample`,
   `TopK`. These are the ops where a converted graph would fail to _load_, not
   just lose accuracy. Disable with `defaultBlockOps: false`, extend with
   `blockOps`.
2. **`blockNodes`** — exact names, `glob*` patterns, or `/regex/`.
3. **Overflow** — with `--overflow=block`, every node consuming a tensor that
   does not fit in fp16 is blocked (see below).

## 2. Apply the overflow strategy

| Mode      | BatchNorm overflow                        | Other overflow               |
| --------- | ----------------------------------------- | ---------------------------- |
| `rescale` | algebraic repair, zero nodes added        | error with a list of tensors |
| `block`   | block the layer (fp32 parameters + Casts) | block the consuming node     |
| `error`   | abort with `OverflowError`                | abort                        |

`rescale` is the default because it is exact, free at runtime, and keeps the
graph 100 % fp16. `block` never loses range and is the right choice when the
parameters are not static initializers (e.g. produced by another node) or when
the BatchNorm has training outputs (`mean`/`var`), which rescaling would alter.

Implementation notes:

- `K` is always a power of two, so `var/K²`, `γ/K` and `ε/K²` are exact in
  binary32. The only rounding is the final fp32 → fp16 cast.
- When `epsilon` is absent, the ONNX default (1e-5) is **added** as an attribute
  so it can be scaled. Forgetting this is a classic source of small, silent
  drift.
- If one initializer is shared by BatchNorm layers that need different `K`,
  conversion stops instead of guessing.

## 3. Convert the float payload

- Every `float32` initializer becomes `float16` via round-to-nearest-even
  (bit-exact against NumPy, including subnormals and overflow to infinity).
- `raw_data` is rewritten; the deprecated `float_data` is emitted as `raw_data`.
- Float tensors inside `Constant` nodes are converted too — LaMa keeps 364 of
  them (FFT basis matrices) in the graph body.
- Tensors whose **every** consumer is blocked stay `float32`: converting them
  would only add a cast back.
- Existing `Cast(…, to=float32)` nodes are retargeted at fp16 — otherwise a
  graph converted "everything except the casts" stops type-checking. Casts that
  feed only blocked nodes are left alone (no pointless fp16→fp32 round-trip).

## 4. Insert fp16 ⇄ fp32 boundaries

The converter maintains an element-type map for every tensor, built from:

- initializer and `Constant` payload dtypes,
- `Cast` `to` attributes,
- graph input types,
- known non-float outputs (`Shape`, `Size`, `ArgMax`, `NonZero` → int64;
  `Equal`, `Less`, `Not`, `IsNaN` → bool),
- blocked nodes → fp32.

A cast is inserted **only when the resolved dtype actually differs**, so int64
inputs of blocked ops are not wrapped in float casts.

Boundaries:

| Boundary                           | What happens                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| Graph input (fp32, `keepIoTypes`)  | `Cast(fp32 → fp16)` inside the graph; the name and type of the input are unchanged         |
| Blocked node input (fp16)          | `Cast(fp16 → fp32)`, cached and shared by consumers                                        |
| Blocked node output (fp32)         | `Cast(fp32 → fp16)` for non-blocked consumers                                              |
| Graph output (fp32, `keepIoTypes`) | the producer emits `name__out_fp16`, a final `Cast` restores the public name and fp32 type |

With `keepIoTypes: false` the casts are omitted and the I/O `ValueInfo` element
types are switched to fp16 — useful when the caller already has fp16 buffers.

## 5. Finish

- **Topological sort** (stable Kahn) — newly appended `Cast` nodes would
  otherwise come after their consumers, which ONNX Runtime rejects.
- **Structural validation** — every node input must be defined, no tensor may be
  produced twice, every graph output must exist.
- **Report** — byte counts, converted tensors, kept-fp32 tensors, inserted casts,
  BatchNorm rescales, blocked nodes, warnings.

## What is _not_ handled

Honest limits of v0.1:

- **Subgraphs** (`If`, `Loop`, `Scan` bodies) are not traversed. If a model
  stores float initializers inside a subgraph, they stay fp32. The graph still
  loads; only part of the size win is lost.
- **External data** (`TensorProto` with an `external_data` entry) is rejected
  with a clear error — materialize it first
  (`onnx.external_data_helper.convert_model_from_external_data`).
- **Sparse initializers** are ignored.
- **int8 / int4 quantization** is out of scope; this is an fp16 tool.
- The op block list is a _default_, not a runtime capability probe. If your
  runtime lacks an fp16 kernel we did not list, pass `blockOps`.

## Reading the report

```ts
const { report } = convertBytes(bytes);
// report.inputBytes     208044816
// report.outputBytes    106087355
// report.convertedTensors       606
// report.constantsConverted     148   (only the ones not consumed by blocked ops)
// report.retargetedCasts        432
// report.insertedCasts          111
// report.rescaled               [ { node, k: 16, maxVarianceBefore: 484322, … } ]
// report.blockedNodes           [ '…/Range_1', … ]
// report.keptFloat32            [ { name: '…', reason: 'all 1 consumer node(s) stay fp32' } ]
```
