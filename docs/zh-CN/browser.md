# 在浏览器中使用

> English · **简体中文**

这里要分开回答两个问题：

1. [能在浏览器里跑转换器吗？](#在浏览器中转换) —— 能，核心代码不依赖 Node。
2. [该在浏览器里跑**转换后的模型**吗？](#在浏览器中运行-fp16-模型) —— 取决于后端。WebGPU：可以。WASM：先测一测。

---

## 在浏览器中转换

转换器只是对 `Uint8Array` 做纯计算：不访问文件系统，不需要 WASM，也不要求 Worker。

```html
<input id="file" type="file" accept=".onnx" />
<script type="module">
  import { convertBytes, inspectModel, decodeModel } from 'onnx-fp16-guard';

  document.getElementById('file').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    const source = new Uint8Array(await file.arrayBuffer());

    // 先出报告：开销小、只读，并且能告诉用户接下来会发生什么。
    const inspection = inspectModel(decodeModel(source));
    console.log(
      `projected ${(inspection.tensors.projectedPayloadBytes / 2 ** 20).toFixed(0)} MiB, ` +
        `${inspection.overflow.findings.length} tensor(s) above the fp16 range`,
    );

    const { bytes, report } = convertBytes(source);
    const url = URL.createObjectURL(new Blob([bytes], { type: 'application/octet-stream' }));
    const link = Object.assign(document.createElement('a'), { href: url, download: 'model.fp16.onnx' });
    link.click();
    console.log(`saved ${(report.bytesSaved / 2 ** 20).toFixed(0)} MiB`);
  });
</script>
```

完整的拖拽示例在 [`examples/browser`](../../examples/browser)（无需构建步骤；Chrome 下通过 `file://` 打开即可，详见该目录中的说明）。

### 内存

转换器会同时持有解码后的 protobuf、新的张量数据和编码后的输出。峰值内存请按**模型体积的约 3 倍**预留：

| 模型体积 | 峰值内存 |
| -------: | -------: |
|    50 MB |  ~150 MB |
|   200 MB |  ~600 MB |

对超过 100 MB 的模型，请把转换放进 Web Worker，避免页面卡死，并且调用一次 `convertBytes`，不要逐个张量处理。

### 打包器

本包是 ESM，不需要配置任何 Node polyfill。它依赖 CommonJS 的 `onnx-proto`；Vite、Rollup、webpack 和 esbuild 都能处理。

```ts
// vite / rollup / webpack
import { convertBytes } from 'onnx-fp16-guard';
```

如果你的目标浏览器没有 `BigInt64Array`（非常老的浏览器），CLI 仍然可用；`verifyModels` 需要 Node。

---

## 在浏览器中运行 fp16 模型

转换器只改变存储方式。fp16 是否**更快**取决于执行后端。

### WebGPU：值得这么做的原因

onnxruntime-web 的 WebGPU 后端有原生 fp16 kernel。数据量减半意味着上传量减半、内存带宽减半，还能用上硬件的 fp16 ALU。

```ts
import * as ort from 'onnxruntime-web/webgpu';

const session = await ort.InferenceSession.create('/models/model.fp16.onnx', {
  executionProviders: ['webgpu'],
  graphOptimizationLevel: 'all',
});
```

在我们自己的 benchmark 中（LaMa 512×512 分块，Apple silicon 上的无头 Chrome），改用 fp16 后 WebGPU 达到**每块 0.86 s，而 WASM 是每块 4.06 s**——相差 4.7 倍。

### WASM：通常会变慢

WASM 后端没有原生 fp16 计算。fp16 模型会被加载、在内部转回 fp32、再按原样执行——你付出了解码和 Cast 的代价。在我们的测量中，同一个模型在 WASM 上比它的 fp32 版本**慢**约 1.5 倍。

实用规则：

```ts
const webgpu = typeof navigator !== 'undefined' && 'gpu' in navigator;
const model = webgpu ? 'model.fp16.onnx' : 'model.onnx'; // 两份都要留
const providers = webgpu ? ['webgpu'] : ['wasm'];
```

做特性检测；不要以为有 `navigator.gpu` 就等于有可用的适配器（iOS Safari 对 WebGPU 的支持还很新且不完整）。始终为纯 CPU 客户端保留一份 fp32 回退。

### 会话启动检查清单

- `onnxruntime-web` 的版本与 ORT 工具的版本会漂移；上线前用 `verifyModels` 验证。
- 按需设置 `ort.env.wasm.numThreads`：多线程 WASM 需要跨源隔离响应头（`Cross-Origin-Opener-Policy`、`Cross-Origin-Embedder-Policy`）。
- `.onnx` 用 `Content-Type: application/octet-stream` 发送，并配上长期缓存头；每个构建对应的文件都是不可变的。
- 为 `verify` 和 CPU 设备保留 fp32 模型；fp16 副本是交付优化，不是替代品。

### 关于没有 fp16 kernel 的算子

`Range`、`Resize`、`TopK`、`NonMaxSuppression`、`RoiAlign`、`CumSum` 和 `Upsample` 在 ONNX Runtime 中没有 fp16 kernel。转换器默认让它们保持 fp32，并在两侧插入 `Cast`——这正是转换后的图仍然能加载的原因。如果你的运行时**确实**支持它们，传 `defaultBlockOps: false`（或 `--no-default-blocks`）来省掉这些 Cast。

## 延伸阅读

- [工作原理](how-it-works.md) —— Cast 插在哪里，为什么
- [案例研究：LaMa](case-study-lama.md) —— 一个 208 MB 模型以 102 MB 上线
- ClearPix 工程笔记：
  [浏览器中的 fp16](https://clearpix.org/blob/main/blogs/03-2026-10-01-fp16-onnx-webgpu-browser-inference.md)，
  [溢出 bug](https://clearpix.org/blob/main/blogs/13-2026-10-17-fp16-quantization-quality-loss-overflowed-tensor.md)
