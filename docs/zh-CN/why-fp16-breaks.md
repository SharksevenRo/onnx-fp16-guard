# 为什么 fp16 会毁掉模型（以及为什么没人告诉你）

> English · **简体中文**

> 本文是 [`src/bn.ts`](../../src/bn.ts) 中防护逻辑的技术背景。数据来自一个真实模型——LaMa，208 MB，17,480 个节点——详见[案例研究](case-study-lama.md)。

## 1. float16 不是「小数位少一点的 float32」

binary16 用 1 位符号、5 位指数、10 位尾数：

| 属性           | float32     | float16     |
| -------------- | ----------- | ----------- |
| 最大有限值     | 3.4 × 10³⁸  | **65,504**  |
| 最小正规数     | 1.2 × 10⁻³⁸ | 6.1 × 10⁻⁵  |
| 最小次正规数   | 1.4 × 10⁻⁴⁵ | 5.96 × 10⁻⁸ |
| 有效十进制位数 | ~7.2        | ~3.3        |
| 机器 epsilon   | 1.19 × 10⁻⁷ | 9.77 × 10⁻⁴ |

指数范围缩小了 **2²⁴** 倍（从 8 位降到 5 位）。转换一个大于 65,504 的值并不是「舍入」它——最近的可表示值是 `Infinity`。多数工具为了避免 `Infinity`，选择**截断**到 65,504，这可以说更糟：数值看起来有限、合理，于是下游没有任何东西会察觉。

## 2. 大数值藏在哪里

权重通常很小：在 LaMa 中 `p50 = 0.43`，`p99 = 2.52`。但 BatchNorm 层存的是**滑动统计量**，不是权重：

```
y = (x − μ) / sqrt(var + ε) · γ + β
```

`var` 是该层输入在训练期间累积的方差。在用大激活值训练的网络里，`var` 达到 10⁵–10⁶ 完全正常。在 LaMa 中：

```
max|weight| = 484,322.56        (fp16 上限: 65,504)
超出上限的张量: 606 个中的 16 个 —— 全部是 BatchNorm running_var，共 2,176 个数
```

因为 `var` 在平方根里面，被截断的 `var` 不会产生一个小误差，而是让该层出现**系统性的尺度错误**：

| 转换方式                                | 存储的 max\|var\| | 对该层的影响          |
| --------------------------------------- | ----------------: | --------------------- |
| fp32 参考                               |        484,322.56 | —                     |
| 截断到 65,504（`onnxconverter-common`） |         65,504.00 | `1/√var` 偏大 2.7 倍  |
| 直接 `astype(float16)`                  |               `∞` | `√∞ = ∞` → 输出恒为 β |

75 层里有 16 层坏掉，就足以毁掉最终输出，而文件在结构上依然合法、可加载、看起来正常。

## 3. 证据：精度从来不是问题

如果 fp16 失败是因为 10 位尾数，那么**只降低精度、不改变类型**应该造成同样的损失。事实并非如此：

| 变体（仍是 float32）        | 尾数位 | 遮罩区域 PSNR | 与 fp32 的一致度 |
| --------------------------- | -----: | ------------: | ---------------: |
| `fp32_w13`                  |  13 位 |      28.88 dB |    95.9–109.0 dB |
| `fp32_w16`（= fp16 精度）   |  10 位 |      28.88 dB |     74.5–96.6 dB |
| `fp32_wbf16`（= bf16 精度） |   7 位 |      28.83 dB |     66.9–76.9 dB |
| fp32 基线                   |  23 位 |      28.88 dB |                — |

把尾数砍到 **7 位**只损失 0.05 dB。这个模型对精度不敏感，对**范围**敏感。任何只盯着精度的工具都会漏掉这个 bug。

## 4. 为什么所有转换器都不吭声

- `onnxconverter-common.float16.convert_float_to_float16` 用 `max_finite_val` 截断数值，默认值是 **10,000**——在 LaMa 的方差上就是 48 倍误差——而且不打任何日志。
- `onnxruntime.tools.convert_onnx_models_to_float16` 用 fp16 最大值（65,504），同样不打日志。
- `model.astype(np.float16)` / `TensorProto` 强转会产生 `inf`，最好的情况也只是抛出一条 NumPy 警告。

这些转换在每种情况下都「成功」了：文件合法、图通过类型检查、会话能加载。失败纯粹是数值层面的——而这恰恰是上线之后最难发现的一类失败。

## 5. 修复

对 BatchNorm 层来说，把方差按 2 的幂缩放是精确的：

```
y = (x − μ) / √(var + ε) · γ + β
  = (x − μ) / √(var/K² + ε/K²) · (γ/K) + β
```

取 `K = 2^⌈log₂(max|var| / 40000)⌉`，于是：

- `var/K²` 与 `γ/K` 在 binary32 中只是指数位的平移——修复本身不引入任何舍入；
- 该层对任意输入都计算同一个函数；
- 计算图**不增加任何节点**（没有 Cast，也没有 fp32 孤岛）。

对 LaMa，`K = 16`：`484,322 → 1,892`，`γ → γ/16`，`ε → ε/256`。

另一种做法——让这 16 层保持 fp32——同样正确，对应 `--overflow=block`；代价是 34 KB 的 fp32 参数以及若干 Cast 节点。

## 6. 自己检测

```ts
import { decodeModel, inspectModel } from 'onnx-fp16-guard';

const report = inspectModel(decodeModel(bytes));
for (const finding of report.overflow.findings) {
  console.log(finding.tensor, finding.absMax, finding.fix);
}
```

`inspectModel` 会扫描初始化器**以及** `Constant` 节点的载荷，所以嵌在图里的张量也能覆盖到。把它接进 CI：

```bash
onnx-fp16-guard inspect model.onnx --fail-on-overflow
```

退出码 2 表示「这个模型无法在不做决定的前提下转成 fp16」。这就是本工具的「防护」：给你一个决定，而不是一次截断。

## 延伸阅读

- [案例研究：LaMa](case-study-lama.md)
- [转换器是如何工作的](how-it-works.md)
- ONNX Runtime 的 float16 工具：
  <https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers>
- `onnxconverter-common`：<https://github.com/microsoft/onnxconverter-common>
