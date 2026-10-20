# 常见问题

> English · **简体中文**

### 需要 Python、onnxruntime 或 GPU 吗？

不需要。转换是对 protobuf 字节做的纯 TypeScript 处理。只有可选的 `verify` 命令需要 `onnxruntime-node`，而且它不是本包的依赖。

### 它和 `onnxconverter-common` / ORT 的 fp16 工具有什么区别？

它们只负责转换；本工具还会**检查这次转换在数值上是否可表示**，并修复唯一常见的不可能情况（BatchNorm `running_var` 超过 65,504）。它们静默截断——`onnxconverter-common` 默认截到 10,000——这让我们的模型退化 −13.9 dB。本工具还会屏蔽没有 fp16 kernel 的算子，因此转换后的图依然能加载。

### 能和 `onnxruntime-web` 一起用吗？

能，这正是主要目标场景。转换器在 Node/CI 中运行；产出的模型在 `onnxruntime-web` 中加载。见 [docs/zh-CN/browser.md](browser.md)。

### fp16 会让推理更快吗？

在 WebGPU、CUDA、CoreML 和 DirectML 上：通常会，大致与带宽下降成正比。在 WASM/CPU 上：不会——onnxruntime-web 会在内部转回 fp32，我们实测慢了 1.5 倍。给纯 CPU 客户端保留一份 fp32 模型。

### 能在浏览器里运行吗？

能。`convertBytes()` 只使用 typed array。峰值内存按模型体积的约 3 倍预留；超过 100 MB 的模型请放进 Web Worker。

### 如果模型装不进 fp16 会怎样？

- `--overflow=rescale`（默认）：修复 BatchNorm 方差；任何**其他**溢出张量都会让转换中止并列出清单。
- `--overflow=block`：受影响的节点保持 fp32，两侧插入 Cast。
- `--overflow=error`：立即中止。这是 CI 模式。

任何情况下都不会在不告知你的前提下截断。

### 为什么重缩放目标是 40,000，而不是 65,504？

因为 `var` 在平方根里面，而这点余量不花任何代价：无论如何 `K` 都必须是 2 的幂。它还能为那些统计量与我们所测略有差异的导出器留出空间。

### 支持 int8 或 4-bit 量化吗？

不支持。那是另一类问题：int8 需要校准数据和误差累积分析。本包刻意只做 fp16。

### 我的模型用了 `If`/`Loop`，支持吗？

部分支持。v0.1 不会遍历子图——这些张量会保持 fp32，所以模型仍然可用，只是体积收益变小。见 [ROADMAP.md](../../ROADMAP.md)。

### 转换后的文件比原文件**更大**，为什么？

当权重只占文件的一小部分时，新增的 `Cast` 节点和 protobuf 开销可能超过载荷节省的字节。`inspect` 会打印预计载荷，转换前就能判断。

### JSON 输出稳定吗？

稳定——`inspect --json` 和 `convert --json` 打印的字段属于公共 API，遵循 semver（新增字段是 minor，删除字段是 major）。

### 如何引用？

见 [CITATION.cff](../../CITATION.cff)。
