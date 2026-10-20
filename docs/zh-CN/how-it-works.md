# 工作原理

> English · **简体中文**

`convertModel()` 对已解码的 `ModelProto` 做一次线性遍历。它不做任何近似，每一个决定都记录在返回的报告里。

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐
│ 1. 屏蔽     │──▶│ 2. 溢出处理  │──▶│ 3. 转换       │──▶│ 4. Cast    │──▶│ 5. 收尾    │
│ 算子 + 节点 │   │ 重缩放/保持  │   │ f32 → f16     │   │ 边界       │   │ 排序+校验  │
└─────────────┘   └──────────────┘   └───────────────┘   └────────────┘   └────────────┘
```

## 1. 决定哪些内容必须留在 fp32

有三类来源会进入「被屏蔽」集合：

1. **`DEFAULT_FP32_OPS`**——ONNX Runtime 里没有 fp16 kernel 的算子：`Range`、`Resize`、`NonMaxSuppression`、`RoiAlign`、`CumSum`、`Upsample`、`TopK`。对这些算子来说，转换后的图会**加载失败**，而不只是掉精度。可用 `defaultBlockOps: false` 关闭，或用 `blockOps` 追加。
2. **`blockNodes`**——精确名字、`glob*` 模式，或 `/regex/`。
3. **溢出处理**——在 `--overflow=block` 下，任何消费了「装不进 fp16 的张量」的节点都会被屏蔽（见下文）。

## 2. 应用溢出策略

| 模式      | BatchNorm 溢出               | 其他溢出             |
| --------- | ---------------------------- | -------------------- |
| `rescale` | 代数修复，不增加任何节点     | 报错并列出张量       |
| `block`   | 屏蔽该层（fp32 参数 + Cast） | 屏蔽消费该张量的节点 |
| `error`   | 抛出 `OverflowError` 中止    | 中止                 |

默认是 `rescale`，因为它精确、运行时零开销，并且让整张图 100% 保持 fp16。`block` 永不损失范围，适合参数不是静态初始化器（例如由其他节点算出）的情况，也适合 BatchNorm 带训练输出（`mean`/`var`）的情况——对这些输出做重缩放会改变语义。

实现要点：

- `K` 始终取 2 的幂，因此 `var/K²`、`γ/K`、`ε/K²` 在 binary32 中都是精确的。唯一的舍入来自最后一步 fp32 → fp16。
- 如果节点没有 `epsilon` 属性，会**补上** ONNX 的默认值（1e-5）再参与缩放。忘记这一点，往往就会引入难以察觉的细小偏差。
- 如果同一个初始化器被需要不同 `K` 的多个 BatchNorm 层共享，转换会直接停止，而不是猜测。

## 3. 转换浮点数据

- 每个 `float32` 初始化器都通过就近舍入到偶数（RNE）变成 `float16`（与 NumPy 逐位一致，包括次正规数与上溢为无穷）。
- 重写 `raw_data`；已废弃的 `float_data` 也统一输出为 `raw_data`。
- `Constant` 节点里的浮点张量同样会被转换——LaMa 有 364 个这样的张量（FFT 基矩阵）直接存在计算图里。
- 如果某个张量的**所有**消费者都被屏蔽，它保持 `float32`：转换它只会多出一个转回去的 Cast。
- 已有的 `Cast(…, to=float32)` 节点会被改成 fp16——否则「除了 Cast 之外全都转换了」的图会通不过类型检查。只供被屏蔽节点使用的 Cast 保持原样（避免无意义的 fp16→fp32 往返）。

## 4. 插入 fp16 ⇄ fp32 边界

转换器为每个张量维护一张元素类型表，来源包括：

- 初始化器与 `Constant` 内嵌数据的 dtype；
- `Cast` 的 `to` 属性；
- 图的输入类型；
- 已知的非浮点输出（`Shape`、`Size`、`ArgMax`、`NonZero` → int64；`Equal`、`Less`、`Not`、`IsNaN` → bool）；
- 被屏蔽的节点 → fp32。

**只有当解析出的 dtype 确实不同时**才插入 Cast，因此被屏蔽算子的 int64 输入不会被套上浮点 Cast。

边界规则：

| 边界                          | 处理方式                                                                |
| ----------------------------- | ----------------------------------------------------------------------- |
| 图输入（fp32，`keepIoTypes`） | 在图内插入 `Cast(fp32 → fp16)`；输入的名字与类型保持不变                |
| 被屏蔽节点的输入（fp16）      | 插入 `Cast(fp16 → fp32)`，缓存后与多个消费者共享                        |
| 被屏蔽节点的输出（fp32）      | 对未被屏蔽的消费者插入 `Cast(fp32 → fp16)`                              |
| 图输出（fp32，`keepIoTypes`） | 生产者改用 `name__out_fp16`，最后由一个 `Cast` 还原公开名字与 fp32 类型 |

使用 `keepIoTypes: false` 时会省略这些 Cast，并把输入/输出 `ValueInfo` 的元素类型直接改成 fp16——当调用方本来就持有 fp16 缓冲区时很有用。

## 5. 收尾

- **拓扑排序**（稳定的 Kahn 算法）——新追加的 `Cast` 节点否则会排在其消费者之后，ONNX Runtime 会拒绝这种图。
- **结构校验**——每个节点输入都必须有定义，同一个张量不能被生产两次，每个图输出都必须存在。
- **报告**——字节数、转换的张量、保持 fp32 的张量、插入的 Cast、BatchNorm 重缩放、被屏蔽的节点、警告。

## 哪些情况**没有**处理

v0.1 的诚实边界：

- **子图**（`If`、`Loop`、`Scan` 的 body）不会被遍历。如果模型在子图里存浮点初始化器，它们会保持 fp32。图仍能加载，只是体积收益少了一部分。
- **外部数据**（带 `external_data` 条目的 `TensorProto`）会被明确拒绝并报错——请先把它转成内嵌数据（`onnx.external_data_helper.convert_model_from_external_data`）。
- **稀疏初始化器**会被忽略。
- **int8 / int4 量化**不在范围内；这是 fp16 工具。
- 算子屏蔽列表只是**默认值**，不是运行时能力探测。如果你的运行时缺少某个我们没列出的 fp16 kernel，请用 `blockOps` 指定。

## 读懂报告

```ts
const { report } = convertBytes(bytes);
// report.inputBytes     208044816
// report.outputBytes    106087355
// report.convertedTensors       606
// report.constantsConverted     148   （只统计未被屏蔽节点消费的那些）
// report.retargetedCasts        432
// report.insertedCasts          111
// report.rescaled               [ { node, k: 16, maxVarianceBefore: 484322, … } ]
// report.blockedNodes           [ '…/Range_1', … ]
// report.keptFloat32            [ { name: '…', reason: 'all 1 consumer node(s) stay fp32' } ]
```
