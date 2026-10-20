# 案例研究：LaMa，208 MB → 102 MB，画质无损

> English · **简体中文**

LaMa 是一个大遮罩图像修复模型：5100 万参数、17,480 个节点，几乎每一层都带 FFT 模块（FFC）。我们在 [ClearPix](https://clearpix.org)（一个浏览器端照片编辑器）里让它完全在客户端运行，因此下载体积是产品特性，而不是可选的优化。

## 起点

| 属性               | 数值                                                        |
| ------------------ | ----------------------------------------------------------- |
| 文件               | `lama_fp32.onnx`，208,044,816 字节（198 MiB）               |
| 初始化器           | 606 个，全部 float32，共 5100 万参数                        |
| 主要结构           | FFC 层：17,480 个节点中的 17,022 个                         |
| 权重幅值           | p50 0.43 · p90 1.24 · p99 2.52 · **最大 484,322.56**        |
| 超过 65,504 的张量 | **16** 个——全部是 BatchNorm 的 `running_var`，共 2,176 个数 |

## 失败

用 `onnxconverter-common`（或 ORT 的 float16 工具）转换后，文件减半，产出的模型能加载、能运行，输出结果却完全不可用。在我们的 14 张图 benchmark 上（使用真值遮罩）：

| 变体                            | 遮罩区域 PSNR |    整帧 PSNR | 与 fp32 一致度 |      体积 |
| ------------------------------- | ------------: | -----------: | -------------: | --------: |
| fp32 基线                       |      29.74 dB |     43.23 dB |              — | 198.4 MiB |
| fp16，普通转换工具              |  **15.81 dB** | **29.30 dB** |   27.6–28.6 dB | 102.8 MiB |
| fp16，重缩放 `K=16`             |      29.73 dB |     43.23 dB |   **73.05 dB** | 102.8 MiB |
| fp16，16 个 BatchNorm 冻结 fp32 |      29.73 dB |     43.23 dB |        73.1 dB | 102.8 MiB |

普通转换让遮罩区域损失了 **13.9 dB**。修复后的模型与 fp32 相差 0.01 dB——在 benchmark 的噪声范围内。

上面这些结果之间，**权重**没有任何区别。只有 16 个方差张量被 2 的幂重缩放了：

```
K = 2^⌈log₂(484,322 / 40,000)⌉ = 16
running_var: 484,322 → 1,892       gamma: γ/16       epsilon: ε/256
```

## `onnx-fp16-guard` 在这个模型上的现状

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

报告中的要点：

- 606/606 个初始化器全部转换；fp16 权重数据共 **101.98 MB**，与我们线上生产模型完全一致。
- 16 个 BatchNorm 层被重缩放；这些层**没有任何 fp32 节点**（没有 Cast，也没有运行时开销）。
- 108 个浮点 `Range` 节点（FFT 机制的一部分）被自动保持为 fp32——ONNX Runtime 没有 fp16 `Range` kernel，fp16 的 `Range` 会在加载时报错：
  `Type 'tensor(float16)' of input parameter … of operator (Range) is invalid`。
  模型里另外 72 个 `Range` 节点是 int64 位置编码；它们从未是 fp16 候选，因此原样保留。
- 364 个浮点 `Constant` 张量中有 148 个被转换（FFT 基），其余供被屏蔽的 `Range` 节点使用，保持 fp32。
- 整个转换过程对 208 MB 的模型**不到一秒**。

用 ONNX Runtime 1.30 在相同输入上验证：

| 对比（同一 512² 输入）         |         整帧 |     遮罩区域 |
| ------------------------------ | -----------: | -----------: |
| 本转换器 vs fp32               |     37.58 dB |     27.20 dB |
| 线上 fp16 模型 vs fp32         |     37.59 dB |     27.21 dB |
| **本转换器 vs 线上 fp16 模型** | **76.71 dB** | **66.33 dB** |

两点观察。第一，新转换器与线上模型在数值上就是同一个模型（一致度 66–77 dB）。第二，在**纯随机噪声**上，两个 fp16 模型相对 fp32 都有明显差异——这是 fp16 累加的最坏情况，并不代表实际表现；在我们的 14 张真实图片上，遮罩区域画质与 fp32 只差 0.01 dB。

## 运行时

| 路径                  | 每块 512² 耗时 |
| --------------------- | -------------: |
| WASM，fp32 模型       |         2.75 s |
| WASM，fp16 模型       |         4.06 s |
| **WebGPU，fp16 模型** |     **0.86 s** |

fp16 是 WebGPU 的特性。在 WASM 上模型会在内部被转回 fp32，因此**慢** 1.5 倍；ClearPix 会同时部署两份文件，运行时再选择。同一个模型在 WebGPU 上比 WASM 快 4.7 倍。

## 可推广的经验

1. **先查范围，再谈精度。** 这里把尾数压到 7 位只损失 0.05 dB；一个被截断的方差损失了 13.9 dB。
2. **BatchNorm 统计量是头号嫌疑。** 它们是唯一一类量级与激活尺度绑定的常见权重。
3. **不声不响的截断才是真正的 bug。** 一个会主动报错（或至少会报告）的转换器，能把一周的调试缩短成一行修复。
4. **有些算子根本无法在 fp16 下运行**（`Range`、`Resize`、`TopK` 等）。明确让它们保持 fp32，不要等加载时的类型错误来告诉你。

## 原始材料

最初的调查与全部原始结果在 ClearPix 仓库中：

- [FINDINGS.md](https://clearpix.org/blob/main/lab/lama-quant/FINDINGS.md)
- [博客：那个溢出的张量](https://clearpix.org/blob/main/blogs/13-2026-10-17-fp16-quantization-quality-loss-overflowed-tensor.md)
- [博客：浏览器中的 fp16 ONNX + WebGPU](https://clearpix.org/blob/main/blogs/03-2026-10-01-fp16-onnx-webgpu-browser-inference.md)
- [博客：WebGPU Slice 的 off-by-one](https://clearpix.org/blob/main/blogs/14-2026-10-18-lama-webgpu-slice-off-by-one.md)
