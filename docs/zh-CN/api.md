# API 参考

> English · **简体中文**

本包仅支持 ESM，并自带 TypeScript 声明。核心入口（`onnx-fp16-guard`）不引用任何 Node 内置模块，可在浏览器、Worker 和边缘运行时中原样运行。只有 `verifyModels()` 和 CLI 会接触 Node API。

```ts
import {
  // 高层 API
  inspectModel,
  convertModel,
  convertBytes,
  verifyModels,

  // protobuf / 张量
  decodeModel,
  encodeModel,
  readFloat32Tensor,
  readFloat16Tensor,
  writeFloat32Tensor,
  writeFloat16Bits,
  tensorDims,
  tensorElementCount,
  tensorBytes,
  tensorAbsMax,
  initializerBytes,

  // 图操作
  collectProducers,
  collectConsumers,
  topologicalSort,
  validateGraph,
  renameNodeInputs,
  renameNodeOutputs,
  makeCast,
  elemTypeOf,
  setElemType,

  // BatchNorm 修复
  findBatchNormOverflows,
  rescaleBatchNorms,
  rescaleFactor,

  // 转换常量
  DEFAULT_FP32_OPS,
  DEFAULT_TARGET_MAX,
  FP16_MAX,
  DataType,
  AttrType,

  // fp16 原语
  float32ToFloat16,
  float32ToFloat16Bits,
  float16ToFloat32,
  float16BitsToFloat32,

  // 报告格式化
  formatInspection,
  formatConversion,
  formatBytes,
  OverflowError,
} from 'onnx-fp16-guard';
```

类型：`ModelProto`、`GraphProto`、`NodeProto`、`TensorProto`、`AttributeProto`、`ValueInfoProto`、`InspectionReport`、`ConvertOptions`、`ConvertReport`、`VerifyReport`、`BatchNormOverflow`、`RescaleChange`。

---

## 转换

### `convertBytes(bytes, options?) → { bytes, report }`

最安全的入口：解码、转换、重新编码。传入的数组不会被修改。

```ts
const { bytes, report } = convertBytes(source, { overflow: 'rescale' });
```

### `convertModel(model, options?) → { model, report }`

原地修改已解码的模型（零拷贝，对 200 MB 的模型很重要）并返回它。如果你已经解码了模型，或想继续持有这个对象，就用这个。

```ts
const model = decodeModel(source);
const { report } = convertModel(model, { keepIoTypes: true });
const bytes = encodeModel(model);
```

### `ConvertOptions`

| 选项               | 类型                              | 默认值      | 说明                                  |
| ------------------ | --------------------------------- | ----------- | ------------------------------------- |
| `keepIoTypes`      | `boolean`                         | `true`      | 图输入/输出保持 fp32，在图内插入 Cast |
| `overflow`         | `'rescale' \| 'block' \| 'error'` | `'rescale'` | BatchNorm 溢出策略                    |
| `limit`            | `number`                          | `65504`     | fp16 范围上限                         |
| `targetMax`        | `number`                          | `40000`     | 重缩放的目标 `max\|var\|`             |
| `blockOps`         | `string[]`                        | `[]`        | 额外保持 fp32 的算子类型              |
| `blockNodes`       | `string[]`                        | `[]`        | 保持 fp32 的节点名（`*`、`/regex/`）  |
| `defaultBlockOps`  | `boolean`                         | `true`      | 是否应用 `DEFAULT_FP32_OPS`           |
| `convertConstants` | `boolean`                         | `true`      | 是否转换 `Constant` 节点里的浮点数据  |
| `toposort`         | `boolean`                         | `true`      | 插入 Cast 后是否重新排序节点          |
| `validate`         | `boolean`                         | `true`      | 是否对结果做结构校验                  |

### `ConvertReport`

| 字段                 | 类型                 | 说明                                     |
| -------------------- | -------------------- | ---------------------------------------- |
| `inputBytes`         | `number`             | 转换前的模型大小                         |
| `outputBytes`        | `number`             | 转换后的大小                             |
| `bytesSaved`         | `number`             | `inputBytes − outputBytes`               |
| `ratio`              | `number`             | `outputBytes / inputBytes`               |
| `convertedTensors`   | `number`             | 转成 fp16 的初始化器数量                 |
| `convertedBytes`     | `number`             | 这些初始化器转换前的字节数               |
| `constantsConverted` | `number`             | 转换的 `Constant` 张量数量               |
| `retargetedCasts`    | `number`             | 从 `Cast(to=float32)` 改成 fp16 的节点数 |
| `keptFloat32`        | `{ name, reason }[]` | 有意保留为 fp32 的张量                   |
| `insertedCasts`      | `number`             | 新增的 `Cast` 节点数                     |
| `rescaled`           | `RescaleChange[]`    | 被修复的 BatchNorm 层                    |
| `blockedNodes`       | `string[]`           | 保持 fp32 的节点                         |
| `warnings`           | `string[]`           | 非致命提示                               |

当策略禁止溢出时，`convertModel`/`convertBytes` 会抛出 `OverflowError`。它携带 `tensors: { name, absMax, limit, fixable }[]`。

---

## 检查

### `inspectModel(model, options?) → InspectionReport`

只读分析。开销：遍历一遍浮点数据（200 MB 模型约 60 ms）。

```ts
const report = inspectModel(decodeModel(bytes), { limit: 65504, targetMax: 40000 });
report.overflow.findings; // 超出范围的张量
report.tensors.projectedPayloadBytes; // 预计的 fp16 数据量
report.ops; // 已排序的算子直方图
```

`InspectionReport` 的字段见 [cli.md](cli.md)（CLI 输出的就是同一个对象，额外多一个 `file`）。

---

## BatchNorm 修复

### `findBatchNormOverflows(model, options?) → BatchNormOverflow[]`

找出 `running_var` 超过上限的 `BatchNormalization` 节点。每条结果包含 `node`、`variance`、`scale`、`absMaxVariance`、2 的幂 `k`，以及是否 `fixable`（静态初始化器）和 `inferenceOnly`。

### `rescaleBatchNorms(model, options?) → RescaleChange[]`

原地应用修复。以下情况会抛错：

- 参数不是静态浮点初始化器；
- 节点带训练输出（`mean`/`var`）；
- 同一个初始化器被需要不同系数的层共享；
- 方差中包含非有限值。

遇到上述任一情况，请改用 `--overflow=block`。

### `rescaleFactor(max, target) → number`

满足 `max / K² ≤ target` 的最小 2 的幂 `K`。

---

## 验证（可选）

### `verifyModels(reference, candidate, options?) → Promise<VerifyReport>`

用 `onnxruntime-node` 在相同输入（确定性 PRNG）上跑两个模型并比较输出。

```ts
import { verifyModels } from 'onnx-fp16-guard';

const result = await verifyModels('model.onnx', 'model.fp16.onnx', { minPsnrDb: 30 });
// result.passed, result.worstPsnrDb, result.outputs[i].psnrDb
```

当 `onnxruntime-node` 不存在时会抛错并提示安装命令。它**不是**本包的依赖。

---

## 底层工具

### 模型 I/O

```ts
decodeModel(bytes: Uint8Array): ModelProto
encodeModel(model: ModelProto): Uint8Array
```

类型声明是手写的 ONNX protobuf 结构视图：`decode()` 之后必然存在的字段声明为必需，其余字段在编码时原样保留（包括未知字段）。

### 张量

```ts
tensorDims(t): number[]
tensorElementCount(t): number
tensorBytes(t): number // 实际存储的字节数
tensorAbsMax(t): number | undefined // 全量扫描，不拷贝
readFloat32Tensor(t): Float32Array // rawData 或已废弃的 floatData
readFloat16Tensor(t): Float32Array // 解码为 f32 数值
writeFloat32Tensor(t, data): void
writeFloat16Bits(t, bits): void
```

### 图

```ts
collectProducers(graph): Map<string, NodeProto>
collectConsumers(graph): Map<string, NodeProto[]>
topologicalSort(graph): void // 稳定 Kahn，遇到环会抛错
validateGraph(graph): string[] // 结构正常时返回 []
renameNodeInputs(graph, from, to): void
renameNodeOutputs(graph, from, to): void
makeCast(input, output, to, name): NodeProto
elemTypeOf(valueInfo): number | undefined
setElemType(valueInfo, elemType): void
```

### fp16 原语

```ts
float32ToFloat16Bits(value: number): number // 16 位模式，RNE
float16BitsToFloat32(bits: number): number
float32ToFloat16(src: Float32Array, out?: Uint16Array): Uint16Array
float16ToFloat32(src: Uint16Array, out?: Float32Array): Float32Array
```

与 NumPy 的 `astype(np.float16)` 逐位一致；用 267 组参考向量加上对全部 65,536 个位型的穷举往返测试验证。

---

## 常用配方

### 批量转换整个目录（Node）

```ts
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { convertBytes } from 'onnx-fp16-guard';

for (const file of readdirSync('models').filter((f) => f.endsWith('.onnx'))) {
  const source = readFileSync(`models/${file}`);
  try {
    const { bytes, report } = convertBytes(source);
    writeFileSync(`models/${file.replace('.onnx', '.fp16.onnx')}`, bytes);
    console.log(file, report.ratio.toFixed(3));
  } catch (error) {
    console.error(file, (error as Error).message);
  }
}
```

### 自定义修复钩子

`rescaleBatchNorms` 与 `convertModel` 是分开的，因此你可以在转换之前检查（甚至修改）修复后的模型：

```ts
import { decodeModel, rescaleBatchNorms, convertModel } from 'onnx-fp16-guard';

const model = decodeModel(source);
const changes = rescaleBatchNorms(model, { targetMax: 20_000 });
console.log(changes); // 先确认将要发生什么
convertModel(model, { overflow: 'error' }); // 现在已没有任何溢出
```

### 输入本身就是 fp16

`inspect` 也能处理 fp16 模型；`convert` 会跳过非浮点张量，并报告 ratio 约为 1。
