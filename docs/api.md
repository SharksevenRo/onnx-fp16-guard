# API reference

The package is ESM-only and ships its own TypeScript declarations. Its core
(`onnx-fp16-guard`) imports no Node built-ins and runs unchanged in browsers,
workers and edge runtimes. Only `verifyModels()` and the CLI touch Node APIs.

```ts
import {
  // high level
  inspectModel,
  convertModel,
  convertBytes,
  verifyModels,

  // protobuf / tensors
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

  // graph helpers
  collectProducers,
  collectConsumers,
  topologicalSort,
  validateGraph,
  renameNodeInputs,
  renameNodeOutputs,
  makeCast,
  elemTypeOf,
  setElemType,

  // BatchNorm repair
  findBatchNormOverflows,
  rescaleBatchNorms,
  rescaleFactor,

  // conversion constants
  DEFAULT_FP32_OPS,
  DEFAULT_TARGET_MAX,
  FP16_MAX,
  DataType,
  AttrType,

  // fp16 primitives
  float32ToFloat16,
  float32ToFloat16Bits,
  float16ToFloat32,
  float16BitsToFloat32,

  // reporting
  formatInspection,
  formatConversion,
  formatBytes,
  OverflowError,
} from 'onnx-fp16-guard';
```

Types: `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`,
`AttributeProto`, `ValueInfoProto`, `InspectionReport`, `ConvertOptions`,
`ConvertReport`, `VerifyReport`, `BatchNormOverflow`, `RescaleChange`.

---

## Converting

### `convertBytes(bytes, options?) → { bytes, report }`

The safe path: decodes, converts, re-encodes. The input array is not mutated.

```ts
const { bytes, report } = convertBytes(source, { overflow: 'rescale' });
```

### `convertModel(model, options?) → { model, report }`

Mutates the decoded model in place (zero-copy, important for 200 MB models) and
returns it. Use this if you already decoded the model or want to keep working
with the object.

```ts
const model = decodeModel(source);
const { report } = convertModel(model, { keepIoTypes: true });
const bytes = encodeModel(model);
```

### `ConvertOptions`

| Option             | Type                              | Default     | Description                                           |
| ------------------ | --------------------------------- | ----------- | ----------------------------------------------------- |
| `keepIoTypes`      | `boolean`                         | `true`      | Keep graph inputs/outputs fp32, cast inside the graph |
| `overflow`         | `'rescale' \| 'block' \| 'error'` | `'rescale'` | BatchNorm overflow strategy                           |
| `limit`            | `number`                          | `65504`     | fp16 range limit                                      |
| `targetMax`        | `number`                          | `40000`     | target `max\|var\|` for the rescale                   |
| `blockOps`         | `string[]`                        | `[]`        | extra op types to keep in fp32                        |
| `blockNodes`       | `string[]`                        | `[]`        | node names (`*`, `/regex/`) to keep in fp32           |
| `defaultBlockOps`  | `boolean`                         | `true`      | apply `DEFAULT_FP32_OPS`                              |
| `convertConstants` | `boolean`                         | `true`      | convert float payloads in `Constant` nodes            |
| `toposort`         | `boolean`                         | `true`      | re-sort nodes after inserting casts                   |
| `validate`         | `boolean`                         | `true`      | structural validation of the result                   |

### `ConvertReport`

| Field                | Type                 | Description                                        |
| -------------------- | -------------------- | -------------------------------------------------- |
| `inputBytes`         | `number`             | size of the model before conversion                |
| `outputBytes`        | `number`             | size after conversion                              |
| `bytesSaved`         | `number`             | `inputBytes − outputBytes`                         |
| `ratio`              | `number`             | `outputBytes / inputBytes`                         |
| `convertedTensors`   | `number`             | initializers converted to fp16                     |
| `convertedBytes`     | `number`             | bytes of those initializers before conversion      |
| `constantsConverted` | `number`             | `Constant` tensors converted                       |
| `retargetedCasts`    | `number`             | existing `Cast(to=float32)` nodes switched to fp16 |
| `keptFloat32`        | `{ name, reason }[]` | tensors deliberately left fp32                     |
| `insertedCasts`      | `number`             | `Cast` nodes added                                 |
| `rescaled`           | `RescaleChange[]`    | BatchNorm layers repaired                          |
| `blockedNodes`       | `string[]`           | nodes kept in fp32                                 |
| `warnings`           | `string[]`           | non-fatal notes                                    |

`OverflowError` is thrown by `convertModel`/`convertBytes` when the strategy
forbids an overflow. It carries `tensors: { name, absMax, limit, fixable }[]`.

---

## Inspecting

### `inspectModel(model, options?) → InspectionReport`

Read-only analysis. Cost: one pass over the float payload (~60 ms for 200 MB).

```ts
const report = inspectModel(decodeModel(bytes), { limit: 65504, targetMax: 40000 });
report.overflow.findings; // tensors that do not fit
report.tensors.projectedPayloadBytes; // estimated fp16 payload
report.ops; // op histogram, sorted
```

`InspectionReport` fields are documented in [cli.md](cli.md#json-schema-stable)
(the CLI emits the same object, plus `file`).

---

## BatchNorm repair

### `findBatchNormOverflows(model, options?) → BatchNormOverflow[]`

Finds `BatchNormalization` nodes whose `running_var` exceeds the limit. Each
finding reports `node`, `variance`, `scale`, `absMaxVariance`, the power-of-two
`k`, and whether it is `fixable` (static initializers) and `inferenceOnly`.

### `rescaleBatchNorms(model, options?) → RescaleChange[]`

Applies the repair in place. Throws when

- the parameters are not static float initializers,
- the node has training outputs (`mean`/`var`),
- one initializer is shared by layers that need different factors,
- a variance contains non-finite values.

Use `--overflow=block` when any of those apply.

### `rescaleFactor(max, target) → number`

Smallest power of two `K` with `max / K² ≤ target`.

---

## Verifying (optional)

### `verifyModels(reference, candidate, options?) → Promise<VerifyReport>`

Runs both models with `onnxruntime-node` on identical inputs (deterministic
PRNG) and compares outputs.

```ts
import { verifyModels } from 'onnx-fp16-guard';

const result = await verifyModels('model.onnx', 'model.fp16.onnx', { minPsnrDb: 30 });
// result.passed, result.worstPsnrDb, result.outputs[i].psnrDb
```

Throws with an install hint when `onnxruntime-node` is not present. It is _not_
a dependency of this package.

---

## Low-level helpers

### Model I/O

```ts
decodeModel(bytes: Uint8Array): ModelProto
encodeModel(model: ModelProto): Uint8Array
```

The type declarations are a hand-written structural view of the ONNX protobuf:
fields guaranteed after `decode()` are required, everything else is preserved
verbatim on encode (unknown fields included).

### Tensors

```ts
tensorDims(t): number[]
tensorElementCount(t): number
tensorBytes(t): number                  // stored payload size
tensorAbsMax(t): number | undefined     // full scan, no copy
readFloat32Tensor(t): Float32Array      // rawData or deprecated floatData
readFloat16Tensor(t): Float32Array      // decoded to f32 values
writeFloat32Tensor(t, data): void
writeFloat16Bits(t, bits): void
```

### Graph

```ts
collectProducers(graph): Map<string, NodeProto>
collectConsumers(graph): Map<string, NodeProto[]>
topologicalSort(graph): void            // stable Kahn, throws on cycles
validateGraph(graph): string[]          // [] when structurally sound
renameNodeInputs(graph, from, to): void
renameNodeOutputs(graph, from, to): void
makeCast(input, output, to, name): NodeProto
elemTypeOf(valueInfo): number | undefined
setElemType(valueInfo, elemType): void
```

### fp16 primitives

```ts
float32ToFloat16Bits(value: number): number   // 16-bit pattern, RNE
float16BitsToFloat32(bits: number): number
float32ToFloat16(src: Float32Array, out?: Uint16Array): Uint16Array
float16ToFloat32(src: Uint16Array, out?: Float32Array): Float32Array
```

Bit-exact against NumPy's `astype(np.float16)`; validated on 267 reference
vectors plus an exhaustive round-trip over all 65,536 patterns.

---

## Recipes

### Convert a whole directory (Node)

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

### Custom repair hook

`rescaleBatchNorms` and `convertModel` are separate so you can inspect (and even
modify) the repaired model before conversion:

```ts
import { decodeModel, rescaleBatchNorms, convertModel } from 'onnx-fp16-guard';

const model = decodeModel(source);
const changes = rescaleBatchNorms(model, { targetMax: 20_000 });
console.log(changes); // verify what will change
convertModel(model, { overflow: 'error' }); // now nothing overflows
```

### Using an already-fp16 input

`inspect` works on fp16 models too; `convert` leaves non-float tensors alone and
reports a ratio of ~1.
