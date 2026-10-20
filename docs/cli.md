# CLI reference

> English · [简体中文](zh-CN/cli.md)

```
onnx-fp16-guard <command> [options]

Commands:
  inspect <model.onnx>                       analyse without writing
  convert <model.onnx> [-o <out.onnx>]       convert
  verify  <reference.onnx> <candidate.onnx>  numeric comparison (needs onnxruntime-node)
```

Run `onnx-fp16-guard --help` for the compact list.

## `inspect`

```bash
onnx-fp16-guard inspect model.onnx
```

Prints the graph summary, tensor storage by dtype, the projected fp16 size, every
tensor above the fp16 range with the fix that would apply, and the op histogram.

```
✖ 16 tensor(s) exceed the fp16 range (65504)
tensor                                           max|value|  kind                    fix
model.generator.model.5.conv1.bn_l.running_var      4.84e+5  batch-norm-running-var  rescale-batchnorm
…
  16 of 75 BatchNormalization layer(s) need attention
```

| Option               | Default | Meaning                                         |
| -------------------- | ------: | ----------------------------------------------- |
| `--json`             |   false | print the full report as JSON (schema below)    |
| `--fail-on-overflow` |   false | exit with code **2** if any tensor does not fit |
| `--limit <n>`        |   65504 | fp16 range limit                                |
| `--target-max <n>`   |   40000 | target `max                                     | var | ` used to compute the rescale factor |

`inspect` never writes files and never mutates the model.

### JSON schema (stable)

```jsonc
{
  "fileBytes": 208044816,
  "irVersion": 8,
  "opsets": [{ "domain": "ai.onnx", "version": 17 }],
  "producer": "pytorch 2.3.0",
  "graphName": "main_graph",
  "counts": { "nodes": 17480, "initializers": 606, "constants": 7054 },
  "tensors": {
    "count": 7660,
    "payloadBytes": 203967756,
    "float32Count": 970,
    "float32Bytes": 203967756,
    "float16Bytes": 0,
    "projectedPayloadBytes": 101983878,
    "buckets": [{ "dataType": 1, "name": "float32", "count": 970, "bytes": 203967756 }],
  },
  "overflow": {
    "limit": 65504,
    "blockingCount": 0,
    "batchNorm": { "total": 75, "overflowing": 16 },
    "findings": [
      {
        "tensor": "model.generator.model.5.conv1.bn_l.running_var",
        "kind": "batch-norm-running-var",
        "absMax": 484322.56,
        "elements": 128,
        "bytes": 512,
        "consumers": ["BatchNormalization(/generator/…/BatchNormalization)"],
        "fix": "rescale-batchnorm",
        "k": 16,
        "blocking": false,
      },
    ],
  },
  "ops": [{ "opType": "Constant", "count": 7054 }],
  "io": {
    "inputs": [{ "name": "image", "elemType": 1, "elemTypeName": "float32", "dims": ["batch", 3, 512, 512] }],
    "outputs": [],
  },
  "warnings": [],
}
```

## `convert`

```bash
onnx-fp16-guard convert model.onnx -o model.fp16.onnx
```

The default output path is `<model>.fp16.onnx`. Existing files are never
overwritten unless you pass `--force`.

| Option                | Default             | Meaning                                                    |
| --------------------- | ------------------- | ---------------------------------------------------------- |
| `-o, --out <path>`    | `<model>.fp16.onnx` | output file                                                |
| `--overflow <mode>`   | `rescale`           | `rescale` \| `block` \| `error`                            |
| `--no-keep-io`        | keep fp32           | make graph inputs/outputs fp16 as well                     |
| `--block-op <op>`     | —                   | keep an op type in fp32; repeatable                        |
| `--block-node <pat>`  | —                   | keep node names in fp32 (`*` globs, `/regex/`); repeatable |
| `--no-default-blocks` | defaults on         | allow fp16 for `Range`, `Resize`, `TopK`, …                |
| `--no-constants`      | constants on        | do not convert `Constant` node tensors                     |
| `--no-toposort`       | sort on             | skip the topological sort (only for debugging)             |
| `--no-validate`       | validate on         | skip structural validation (only for debugging)            |
| `--report <path>`     | —                   | write the conversion report as JSON                        |
| `--json`              | false               | print the report as JSON                                   |
| `--quiet`             | false               | print only the output path                                 |
| `--force`             | false               | overwrite an existing output file                          |
| `--limit <n>`         | 65504               | fp16 range limit                                           |
| `--target-max <n>`    | 40000               | target `max                                                | var | ` for the BatchNorm rescale |

Examples:

```bash
# CI: fail if the model would be corrupted, never write unsafely
onnx-fp16-guard convert model.onnx --overflow=error -o /tmp/check.onnx

# Mixed precision: keep attention softmax maths in fp32
onnx-fp16-guard convert model.onnx --block-op Softmax --block-op LayerNormalization

# Keep only the vision tower in fp32
onnx-fp16-guard convert model.onnx --block-node '/^vision\..*/'
```

## `verify`

```bash
npm install --no-save onnxruntime-node
onnx-fp16-guard verify model.onnx model.fp16.onnx
```

Runs both models on the same generated input (deterministic, seed 42) and
reports per-output PSNR, max absolute difference and cosine similarity.

```
onnxruntime-node 1.30.0
  output: PSNR 73.05 dB · max|Δ| 3.4e-02 · cosine 0.999999
Pass: worst PSNR 73.05 dB ≥ 30 dB
```

| Option            | Default | Meaning                             |
| ----------------- | ------: | ----------------------------------- |
| `--min-psnr <db>` |      30 | pass threshold                      |
| `--seed <n>`      |      42 | PRNG seed for generated inputs      |
| `--amplitude <x>` |       1 | amplitude of generated float inputs |
| `--json`          |   false | machine-readable output             |

`onnxruntime-node` is an optional peer dependency. If it is missing, `verify`
prints the install command and exits 1; everything else keeps working.

## Exit codes

| Code | Meaning                                                          |
| ---: | ---------------------------------------------------------------- |
|    0 | success                                                          |
|    1 | usage error, I/O error, or invalid model                         |
|    2 | fp16 overflow (`inspect --fail-on-overflow`, `--overflow=error`) |
|    3 | verification failed (PSNR below threshold)                       |

## CI recipes

```yaml
# GitHub Actions: never publish a broken fp16 model
- run: npx onnx-fp16-guard inspect model.onnx --fail-on-overflow
- run: npx onnx-fp16-guard convert model.onnx -o model.fp16.onnx
- run: npm i --no-save onnxruntime-node && npx onnx-fp16-guard verify model.onnx model.fp16.onnx
```
