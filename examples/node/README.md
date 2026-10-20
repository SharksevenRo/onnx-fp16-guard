# Node example

Batch-convert a directory and optionally verify every output.

```bash
# from the repository root
npm run build
npm install            # links nothing; the example imports the package by name

cd examples/node
npm install onnx-fp16-guard      # or: npm link ../..
node convert.mjs ../../path/to/models --out ./out --verify
```

Output:

```
lama_fp32.onnx                              198.4 MiB →  101.3 MiB  (49.0% smaller, 0.68s)
  16 BatchNorm layer(s) rescaled, K = 16
  180 node(s) kept in fp32
  verify PASS · worst PSNR 37.58 dB
```

Flags:

| Flag                | Default   | Meaning                                       |
| ------------------- | --------- | --------------------------------------------- |
| `--out <dir>`       | input dir | where `*.fp16.onnx` is written                |
| `--overflow <mode>` | `rescale` | `rescale`, `block` or `error`                 |
| `--verify`          | off       | run `verifyModels` (needs `onnxruntime-node`) |

Exit codes: `0` success, `1` a file failed, `2` overflow with `--overflow=error`.

The script writes files before verifying, which is fine for a nightly job but not
for CI — for that, convert to a temp path and only publish after `verify` passes.
See [docs/cli.md](../../docs/cli.md#ci-recipes).
