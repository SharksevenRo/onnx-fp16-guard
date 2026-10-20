# onnx-fp16-guard

> English · **简体中文**

**拒绝悄悄毁掉你模型的 ONNX fp16 转换工具。**

[![npm](https://img.shields.io/npm/v/onnx-fp16-guard.svg)](https://www.npmjs.com/package/onnx-fp16-guard)
[![CI](https://github.com/SharksevenRo/onnx-fp16-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/SharksevenRo/onnx-fp16-guard/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![node](https://img.shields.io/badge/node-%E2%89%A518.17-brightgreen.svg)](package.json)

```bash
npx onnx-fp16-guard inspect model.onnx
```

```
✖ 16 tensor(s) exceed the fp16 range (65504)
tensor                                           max|value|  kind                    fix
model.generator.model.5.conv1.bn_l.running_var      4.84e+5  batch-norm-running-var  rescale-batchnorm
model.generator.model.25.running_var                4.41e+5  batch-norm-running-var  rescale-batchnorm
…
  16 of 75 BatchNormalization layer(s) need attention
```

几乎所有主流 fp16 转换工具都会把这些数值**截断到 65504**，然后一言不发地把模型交给你——而这个模型输出的已经是垃圾。本工具会先告诉你，然后修好它。

- **零依赖**，除 ONNX protobuf 定义外不需要任何运行时——纯 TypeScript，不依赖 Python，也不依赖 onnxruntime。
- **随处可跑**——Node 18+、CI，或者直接在浏览器里（核心代码不引用任何 Node 内置模块）。
- **默认开启防护**——`--overflow=rescale` 用代数恒等式修复 BatchNorm；`--overflow=error` 把转换变成 CI 门禁。

---

## 13.9 dB 的那个 bug

2026 年 10 月，我们上线了一个浏览器端的图像修复模型（LaMa，208 MB），并尝试用常规的 float16 转换把下载体积减半。结果模型看起来坏掉了：遮罩区域 PSNR 从 **29.74 dB 掉到 15.81 dB**。

权重没问题，精度也没问题（我们把尾数砍到 7 位，肉眼看不出损失）。问题出在**数值范围**：606 个张量里有 16 个——全部是 BatchNorm 的 `running_var`——最大值高达 **484,322**，远超 fp16 能表示的最大有限值 **65,504**。

```
model.5.conv1.bn_l.running_var = 484,322.56   → fp16 截断 → 65,504.0
1 / sqrt(var)                  偏大 2.7×      → 该层输出被放大 2.7×
```

没有任何工具警告我们。`onnxconverter-common` 默认用更激进的 `max_finite_val=10000`（误差 48×），而且它的 `float16` 转换器不打任何日志。直接 `astype(float16)` 更糟：会产生 `inf`，而 `sqrt(inf)` 会让这一层的输出变成常数。

修复方法是一个代数恒等式——把 `var`、`scale`、`epsilon` 同时除以 2 的幂：

```
y = (x − μ) / √(var + ε) · γ + β
  ≡ (x − μ) / √(var/K² + ε/K²) · (γ/K) + β          K = 2^⌈log₂(max|var| / 40000)⌉

484,322 / 16² = 1,892      → 装得进 fp16，函数不变，不增加任何节点
```

| 方案                            | 遮罩区域 PSNR | 相对 fp32 | 体积      |
| ------------------------------- | ------------: | --------- | --------- |
| fp32（基线）                    |      29.74 dB | —         | 198 MiB   |
| fp16，普通转换工具              |  **15.81 dB** | −13.93 dB | 102.8 MiB |
| fp16，16 个 BatchNorm 冻结 fp32 |      29.73 dB | −0.01 dB  | 102.8 MiB |
| **fp16，重缩放（本工具）**      |  **29.73 dB** | −0.01 dB  | 102.8 MiB |

完整分析：[docs/zh-CN/case-study-lama.md](docs/zh-CN/case-study-lama.md) ·
[为什么 fp16 会毁掉模型](docs/zh-CN/why-fp16-breaks.md)。

## 安装

```bash
npm install onnx-fp16-guard          # 作为库使用
npx onnx-fp16-guard --help           # 直接用 CLI，无需安装
```

需要 Node 18.17+，仅支持 ESM。浏览器与边缘运行时同样可用。

## CLI

```bash
# 这个模型里有什么？转换后哪些地方会出问题？
onnx-fp16-guard inspect model.onnx

# 转换（自动重缩放溢出的 BatchNorm，输入/输出保持 fp32）
onnx-fp16-guard convert model.onnx -o model.fp16.onnx

# 转换，然后证明输出仍然与原模型一致
npm install --no-save onnxruntime-node
onnx-fp16-guard verify model.onnx model.fp16.onnx
```

| 参数                  | 含义                                                              |
| --------------------- | ----------------------------------------------------------------- |
| `--overflow=rescale`  | **默认**——用代数方法修复溢出的 BatchNorm 层                       |
| `--overflow=block`    | 让受影响的层保持 fp32（插入 `Cast` 边界，无精度损失）             |
| `--overflow=error`    | 只要有任何张量超过 65504 就中止——适合放进 CI                      |
| `--fail-on-overflow`  | `inspect` 发现张量装不下时以退出码 2 结束                         |
| `--block-op <op>`     | 让某类算子保持 fp32（可重复；`Range`/`Resize`/`TopK` 默认已屏蔽） |
| `--block-node <glob>` | 让匹配的节点保持 fp32（可重复，支持 `*` 与 `/regex/`）            |
| `--no-keep-io`        | 连图的输入/输出也改成 fp16                                        |

其余参数见 [docs/zh-CN/cli.md](docs/zh-CN/cli.md)。

## API

```ts
import { readFileSync, writeFileSync } from 'node:fs';
import { convertBytes, inspectModel, decodeModel } from 'onnx-fp16-guard';

const source = readFileSync('model.onnx');

// 1. 先看再动（不写文件，也不修改模型）
const report = inspectModel(decodeModel(source));
if (report.overflow.findings.length > 0) {
  console.log(report.overflow.findings.map((f) => `${f.tensor}: ${f.absMax} (${f.fix})`));
}

// 2. 转换
const { bytes, report: conversion } = convertBytes(source, {
  overflow: 'rescale', // 'rescale' | 'block' | 'error'
  keepIoTypes: true, // 图的输入/输出保持 fp32
  blockOps: ['Resize'], // 额外让 Resize 保持 fp32
});

writeFileSync('model.fp16.onnx', bytes);
console.log(`${conversion.inputBytes} → ${conversion.outputBytes} bytes`);
```

浏览器里也是同一套 API：

```ts
import { convertBytes } from 'onnx-fp16-guard';

const bytes = new Uint8Array(await file.arrayBuffer());
const { bytes: fp16, report } = convertBytes(bytes);
console.log(`saved ${report.bytesSaved} bytes`);
```

API 参考：[docs/zh-CN/api.md](docs/zh-CN/api.md)。

## 它做了什么

1. **检查**每一个浮点张量——初始化器**以及** `Constant` 节点的载荷——找出超出 `[−65504, 65504]` 的值。
2. **修复**溢出的 BatchNorm 层：用上面的 2 的幂重缩放（或者冻结为 fp32，或者直接拒绝——由你决定）。
3. **转换** float32 → float16，采用就近舍入到偶数（RNE），与 NumPy 的 `astype(np.float16)` 逐位一致，包括次正规数与上溢为无穷。
4. **保持计算图合法**：把已有的 `Cast(to=float32)` 改为 fp16；在被屏蔽的算子两侧插入 fp16 ⇄ fp32 的 `Cast`；正确识别 int64/bool 张量；做拓扑排序，并校验结构。
5. **如实报告**改动——评审能看到每一处变化，绝无静默截断。

## 什么时候该用

**该用**：模型要跑在有 fp16 kernel 的运行时上（WebGPU、CUDA、CoreML、DirectML），并且下载体积很重要。字节数减半，输出不变。

**谨慎用**：WASM。onnxruntime-web 的 WASM 后端没有原生 fp16 计算，fp16 模型在那里会**更慢**（内部要插入 Cast、再用 fp32 计算）。给纯 CPU 设备留一份 fp32 副本——或者先跑 benchmark。

**别用**：权重的确需要超过 65504、且不是 BatchNorm 统计量的模型。工具会报告它们，并按你的选择拒绝转换（`--overflow=rescale`）、保持 fp32（`--overflow=block`）或让 CI 失败（`--overflow=error`）。它从不静默截断。

## 验证

测试集用**外部真值**校验转换器，而不是拿它自己当参照：

- 267 组由 NumPy `astype(np.float16)` 生成的参考向量——随机量级、次正规数、临界值、`inf`、`NaN`——外加对**全部 65,536 个 fp16 位型**的穷举往返测试。
- BatchNorm 重缩放做数值验证：重缩放后的 fp32 层在舍入前与原层误差小于 1e-5；转成 fp16 存储后，对 O(1) 量级的输入绝对误差仍小于 0.01。
- 在 **LaMa 208 MB** 上端到端验证：606/606 个初始化器全部转换，16 个 BatchNorm 被重缩放，fp16 权重共 101.98 MB（与我们线上模型完全一致），在 ONNX Runtime 1.30 中可加载可推理，并且在相同输入下与线上 fp16 模型的一致度为 **66.3 dB**（遮罩区域）/ **76.7 dB**（整帧）。

```bash
npm test                          # 36 个测试，不联网，< 1 秒
node scripts/bench.mjs model.onnx # 体积与耗时（见 BENCHMARKS.md）
```

## 文档

| 文档                                                | 内容                                  |
| --------------------------------------------------- | ------------------------------------- |
| [why-fp16-breaks.md](docs/zh-CN/why-fp16-breaks.md) | 范围与精度的诊断，附证据              |
| [how-it-works.md](docs/zh-CN/how-it-works.md)       | 图改写、dtype 推断、Cast 插入规则     |
| [cli.md](docs/zh-CN/cli.md)                         | 所有参数与退出码                      |
| [api.md](docs/zh-CN/api.md)                         | 类型、函数、选项                      |
| [browser.md](docs/zh-CN/browser.md)                 | 在浏览器里转换，以及在浏览器里跑 fp16 |
| [case-study-lama.md](docs/zh-CN/case-study-lama.md) | 208 MB 图像修复模型的完整案例         |
| [BENCHMARKS.md](BENCHMARKS.zh-CN.md)                | 如何复现这些数字                      |
| [faq.md](docs/zh-CN/faq.md)                         | 简答                                  |

英文原文：[README.md](README.md)。

## FAQ

**需要 Python 或 onnxruntime 吗？**
不需要。转换是对 protobuf 字节做的纯 TypeScript 处理。只有可选的 `verify` 命令需要 `onnxruntime-node`。

**也能做 int8 量化吗？**
不能——只支持 fp16。int8 需要校准数据，权衡点也完全不同。

**模型会变快吗？**
在 WebGPU/CUDA/CoreML 上通常会，因为内存带宽减半。在 WASM/CPU 上不会，反而更慢。请自行测量。

**模型本来就是 fp16 呢？**
`inspect` 会报告 fp16 载荷，`convert` 则是空操作。

**为什么叫这个名字？**
因为它守住了所有其他工具都会静默执行的那一步转换。

## 许可证

[Apache-2.0](LICENSE)。代码抽取自 [ClearPix](https://clearpix.org) 浏览器端照片编辑器——在那里，同一份代码让一个 208 MB 的修复模型以 102 MB 运行而不损失画质。
