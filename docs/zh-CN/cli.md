# CLI 参考

> English · **简体中文**

```
onnx-fp16-guard <command> [options]

Commands:
  inspect <model.onnx>                       只分析，不写文件
  convert <model.onnx> [-o <out.onnx>]       转换
  verify  <reference.onnx> <candidate.onnx>  数值对比（需要 onnxruntime-node）
```

运行 `onnx-fp16-guard --help` 查看简表。

## `inspect`

```bash
onnx-fp16-guard inspect model.onnx
```

输出图摘要、按 dtype 统计的张量存储、预计的 fp16 体积、每个超出 fp16 范围的张量及其适用的修复方式，以及算子直方图。

```
✖ 16 tensor(s) exceed the fp16 range (65504)
tensor                                           max|value|  kind                    fix
model.generator.model.5.conv1.bn_l.running_var      4.84e+5  batch-norm-running-var  rescale-batchnorm
…
  16 of 75 BatchNormalization layer(s) need attention
```

| 选项                 | 默认值 | 含义                                    |
| -------------------- | -----: | --------------------------------------- |
| `--json`             |  false | 以 JSON 输出完整报告（schema 见下）     |
| `--fail-on-overflow` |  false | 任何张量装不下时以退出码 **2** 结束     |
| `--limit <n>`        |  65504 | fp16 范围上限                           |
| `--target-max <n>`   |  40000 | 计算重缩放系数时使用的目标 `max\|var\|` |

`inspect` 从不写文件，也从不修改模型。

### JSON schema（稳定）

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

默认输出路径是 `<model>.fp16.onnx`。除非传入 `--force`，否则绝不覆盖已存在的文件。

| 选项                  | 默认值              | 含义                                                   |
| --------------------- | ------------------- | ------------------------------------------------------ |
| `-o, --out <path>`    | `<model>.fp16.onnx` | 输出文件                                               |
| `--overflow <mode>`   | `rescale`           | `rescale` \| `block` \| `error`                        |
| `--no-keep-io`        | 保持 fp32           | 连图的输入/输出也改成 fp16                             |
| `--block-op <op>`     | —                   | 让某类算子保持 fp32；可重复                            |
| `--block-node <pat>`  | —                   | 让匹配的节点名保持 fp32（`*` 通配、`/regex/`）；可重复 |
| `--no-default-blocks` | 默认开启            | 允许 `Range`、`Resize`、`TopK` 等使用 fp16             |
| `--no-constants`      | 默认转换            | 不转换 `Constant` 节点里的张量                         |
| `--no-toposort`       | 默认排序            | 跳过拓扑排序（仅用于调试）                             |
| `--no-validate`       | 默认校验            | 跳过结构校验（仅用于调试）                             |
| `--report <path>`     | —                   | 把转换报告写成 JSON                                    |
| `--json`              | false               | 以 JSON 打印报告                                       |
| `--quiet`             | false               | 只打印输出路径                                         |
| `--force`             | false               | 覆盖已存在的输出文件                                   |
| `--limit <n>`         | 65504               | fp16 范围上限                                          |
| `--target-max <n>`    | 40000               | BatchNorm 重缩放的目标 `max\|var\|`                    |

示例：

```bash
# CI：如果模型会被损坏就失败，绝不写出不安全的文件
onnx-fp16-guard convert model.onnx --overflow=error -o /tmp/check.onnx

# 混合精度：attention softmax 的数学计算留在 fp32
onnx-fp16-guard convert model.onnx --block-op Softmax --block-op LayerNormalization

# 只让 vision tower 保持 fp32
onnx-fp16-guard convert model.onnx --block-node '/^vision\..*/'
```

## `verify`

```bash
npm install --no-save onnxruntime-node
onnx-fp16-guard verify model.onnx model.fp16.onnx
```

用相同的生成输入（确定性，种子 42）分别跑两个模型，报告每个输出的 PSNR、最大绝对误差与余弦相似度。

```
onnxruntime-node 1.30.0
  output: PSNR 73.05 dB · max|Δ| 3.4e-02 · cosine 0.999999
Pass: worst PSNR 73.05 dB ≥ 30 dB
```

| 选项              | 默认值 | 含义                     |
| ----------------- | -----: | ------------------------ |
| `--min-psnr <db>` |     30 | 通过阈值                 |
| `--seed <n>`      |     42 | 生成输入使用的 PRNG 种子 |
| `--amplitude <x>` |      1 | 生成的浮点输入幅值       |
| `--json`          |  false | 机器可读输出             |

`onnxruntime-node` 是可选的 peer dependency。如果没装，`verify` 会打印安装命令并以 1 退出；其他功能不受影响。

## 退出码

| 代码 | 含义                                                          |
| ---: | ------------------------------------------------------------- |
|    0 | 成功                                                          |
|    1 | 用法错误、I/O 错误，或模型无效                                |
|    2 | fp16 溢出（`inspect --fail-on-overflow`、`--overflow=error`） |
|    3 | 验证失败（PSNR 低于阈值）                                     |

## CI 配方

```yaml
# GitHub Actions：绝不发布损坏的 fp16 模型
- run: npx onnx-fp16-guard inspect model.onnx --fail-on-overflow
- run: npx onnx-fp16-guard convert model.onnx -o model.fp16.onnx
- run: npm i --no-save onnxruntime-node && npx onnx-fp16-guard verify model.onnx model.fp16.onnx
```
