# Contributing

Thanks for taking the time to improve `onnx-fp16-guard`. This document is short
on ceremony and long on "how to run the tests".

## Setup

```bash
git clone https://github.com/SharksevenRo/onnx-fp16-guard.git
cd onnx-fp16-guard
npm install
npm test
```

Requirements: Node 18.17+ (22 LTS recommended). No Python is needed for the
library or its tests.

## Scripts

| Command              | What it does                                   |
| -------------------- | ---------------------------------------------- |
| `npm test`           | 36 unit tests, no network, < 1 s               |
| `npm run test:watch` | vitest in watch mode                           |
| `npm run typecheck`  | `tsc --noEmit` (strict)                        |
| `npm run lint`       | ESLint (typescript-eslint, flat config)        |
| `npm run format`     | Prettier                                       |
| `npm run build`      | tsup → `dist/index.js`, `dist/cli.js`, `.d.ts` |
| `npm run bench`      | build + `scripts/bench.mjs`                    |

CI runs typecheck, lint, format check, build and test on Node 20, 22 and 24.

## Project layout

```
src/
  onnx.ts      structural types for the ONNX protobuf
  fp16.ts      bit-exact float32 ⇄ float16
  model.ts     model I/O, tensor helpers, toposort, validation
  analyze.ts   inspectModel()
  bn.ts        BatchNorm overflow detection + algebraic rescale
  convert.ts   the converter
  verify.ts    optional onnxruntime-node comparison
  format.ts    CLI rendering
  cli.ts       argument parsing, exit codes
test/
  helpers/     synthetic ONNX models built in code
  fixtures/    NumPy-generated fp16 reference vectors
docs/
```

The core (`src/*.ts` except `cli.ts`/`verify.ts`) must not import Node
built-ins — it is published for browsers too.

## What a good change looks like

- **A test that fails before the change.** For converter bugs, build a tiny
  synthetic model in `test/helpers/models.ts`; that is faster and more readable
  than checking in a binary fixture.
- **A note in `docs/how-it-works.md`** if the graph surgery changes.
- **A CHANGELOG entry** under `## [Unreleased]`.
- **No new runtime dependencies** without a discussion in an issue first. The
  package ships exactly one (`onnx-proto`, the official ONNX protobuf
  definitions).

## Adding an op to the fp32 block list

`DEFAULT_FP32_OPS` in `src/convert.ts` exists because some ops have no fp16
kernel in ONNX Runtime. To add one:

1. Reproduce the failure: convert a model that uses the op, load it in
   `onnxruntime-node` or `onnxruntime` (Python) and capture the error.
2. Add the op to `DEFAULT_FP32_OPS`.
3. Add a case to `test/convert.test.ts` (see `buildRangeModel`).
4. Mention it in `docs/how-it-works.md` and the CHANGELOG.

Ops that merely _lose precision_ do not belong in the list — the list is for
correctness/load failures only.

## Reporting a bad conversion

Open an issue with:

- the model name, license and a link (or a minimal reproduction built from
  `test/helpers/models.ts`),
- `onnx-fp16-guard inspect model.onnx --json` output,
- the conversion report (`--report report.json`),
- what you expected vs what happened.

Please do not attach proprietary model weights. A synthetic reproduction or the
inspection report is almost always enough.

## Commit messages

Conventional-ish, imperative, scoped when useful:

```
fix(convert): keep int64 inputs of blocked ops free of float casts
docs: explain the 40000 target in the rescale factor
test: cover Cast retargeting for Constant outputs
```

## Pull requests

- Keep the diff focused; unrelated refactors go in a separate PR.
- `npm run typecheck && npm run lint && npm run format:check && npm test` must
  pass.
- Describe the _failure mode_ you are fixing, not just the code change.
- By contributing you agree that your contribution is licensed under
  Apache-2.0 (see [LICENSE](LICENSE)).

## Code of conduct

Participation is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md).
Report unacceptable behaviour through the private advisory link in that
document.
