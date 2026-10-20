# Benchmark

> English · **简体中文**

这里的一切都能在一台笔记本上、用模型文件复现。不需要 GPU 集群，也没有隐藏测试集。

## LaMa（图像修复，5100 万参数）

模型：`lama_fp32.onnx`，208,044,816 字节，17,480 个节点，opset 17。
耗时测试硬件：Apple M 系列，Node 25，16 GB 内存。

### 转换

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

| 指标                        |                                    数值 |
| --------------------------- | --------------------------------------: |
| 输入                        |               208,044,816 B (198.4 MiB) |
| 输出                        |               106,087,355 B (101.2 MiB) |
| 体积缩减                    |                                  49.0 % |
| `inspect` 耗时              |                                  107 ms |
| `convert`（解码+转换+编码） |                                  680 ms |
| 转换的初始化器              |                               606 / 606 |
| 转换的 `Constant` 张量      |  148 / 364（其余被屏蔽的 `Range` 消费） |
| 重缩放的 BatchNorm 层       |                                 16 / 75 |
| 保持 fp32 的节点            |               108 个浮点 `Range`（FFT） |
| 插入的 Cast 节点            |                                     111 |
| 峰值内存                    | 约 1.0 GB（解码 + 转换 + 编码同时存在） |

fp16 张量载荷为 **101.98 MB**，与 ClearPix 线上生产模型的字节数完全相同。

### 画质

14 张真实图片，真值遮罩，遮罩区域 PSNR（越高越好）：

| 变体                       | 遮罩区域 PSNR | 相对 fp32 |
| -------------------------- | ------------: | --------- |
| fp32                       |      29.74 dB | —         |
| fp16，普通（截断）         |      15.81 dB | −13.93 dB |
| fp16，重缩放（本工具）     |      29.73 dB | −0.01 dB  |
| fp16，16 个 BatchNorm 屏蔽 |      29.73 dB | −0.01 dB  |

在相同输入下与 fp32 的输出级一致度（14 张图范围）：两种修复策略均为 **73.0–73.1 dB**。

### 在任何有 Python 的机器上复现一致度检查

```bash
python -m pip install onnxruntime numpy
python scripts/psnr.py lama_fp32.onnx lama.fp16.onnx
```

```
output[0] PSNR 37.58 dB · max|Δ| 7.610e+01 · RMSE 1.0e+01 · cosine 0.9997343
PASS: worst PSNR 37.58 dB (threshold 30.00 dB)
```

（随机输入是 fp16 的最坏情况——真实图片能达到 73 dB。转换器的输出与线上 fp16 模型在这里分别是 37.58 / 37.59 dB，这正说明它们是同一个模型。）

### 运行时（浏览器）

LaMa，512×512 分块，onnxruntime-web，Apple silicon：

| 路径             |   每块耗时 |
| ---------------- | ---------: |
| WASM，fp32       |     2.75 s |
| WASM，fp16       |     4.06 s |
| **WebGPU，fp16** | **0.86 s** |

## 小模型

对于没有接近上限的 BatchNorm 统计量的模型，转换基本就是「体积除以 2」：

| 模型                   |         fp32 |         fp16 | 重缩放 | 屏蔽 | 说明                 |
| ---------------------- | -----------: | -----------: | -----: | ---: | -------------------- |
| `realesr-general-x4v3` |  4,866,417 B |  2,439,928 B |      0 |    1 | `Resize`；无溢出     |
| `ppocr_v5_mobile_det`  |  4,748,769 B |  2,406,545 B |      0 |    6 | `Resize`/`TopK` 流程 |
| `migan`                | 29,541,096 B | 14,799,796 B |      0 |   14 | 14 个张量保持 fp32   |
| `real_esrgan_x4`       | 67,051,616 B | 33,654,392 B |      0 |    2 | 无溢出               |

四个模型都能在 ONNX Runtime 1.30 中加载并输出 fp32；体积缩减均为 −49% 到 −50%。

## 方法说明

- 体积是磁盘上的文件大小；转换会新增少量 `Cast` 节点，因此权重只占文件极小比例的模型可能略微变大。
- 耗时是单次运行的墙钟时间，文件缓存已预热。它们只用于给出数量级，不是竞品对比 benchmark。
- 画质数据来自 ClearPix 实验室的那次运行；原始 JSON 在
  [`lab/lama-quant/results`](https://clearpix.org/tree/main/lab/lama-quant/results)。

## 提交你自己的结果

用 `--json` 跑 benchmark，然后带着数字提 PR（模型名、许可证、体积、硬件）。尤其欢迎有有趣溢出故事的模型——它们会让这个防护工具变得更好。
