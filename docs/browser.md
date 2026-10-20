# In the browser

Two separate questions:

1. [Can I run the converter in the browser?](#converting-in-the-browser) — yes,
   the core has no Node dependencies.
2. [Should I run the _converted model_ in the browser?](#running-fp16-models-in-the-browser)
   — it depends on the backend. WebGPU: yes. WASM: measure first.

---

## Converting in the browser

The converter is pure computation over a `Uint8Array`; there is no filesystem,
no WASM, no worker requirement.

```html
<input id="file" type="file" accept=".onnx" />
<script type="module">
  import { convertBytes, inspectModel, decodeModel } from 'onnx-fp16-guard';

  document.getElementById('file').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    const source = new Uint8Array(await file.arrayBuffer());

    // Report first: cheap, read-only, and it tells the user what will happen.
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

A complete drag-and-drop demo lives in
[`examples/browser`](../examples/browser) (no build step, no server needed for
`file://` in Chrome — see the note in that folder).

### Memory

The converter holds the decoded protobuf, the new payload and the encoded output
at the same time. Budget roughly **3× the model size** in peak RSS:

| Model size | Peak memory |
| ---------: | ----------: |
|      50 MB |     ~150 MB |
|     200 MB |     ~600 MB |

For models above ~100 MB, run the conversion in a Web Worker so the page stays
responsive, and call `convertBytes` once rather than per-tensor.

### Bundlers

The package is ESM and has no Node polyfills to configure. It depends on
`onnx-proto`, which is CommonJS; Vite, Rollup, webpack and esbuild all handle it.

```ts
// vite / rollup / webpack
import { convertBytes } from 'onnx-fp16-guard';
```

If you target a browser without `BigInt64Array` (very old), the CLI still works;
`verifyModels` needs Node.

---

## Running fp16 models in the browser

The converter only changes storage. Whether fp16 is _faster_ depends on the
execution provider.

### WebGPU: the reason to do this

onnxruntime-web's WebGPU backend has native fp16 kernels. Half the bytes means
half the upload and half the memory traffic, and it can use the hardware's fp16
ALUs.

```ts
import * as ort from 'onnxruntime-web/webgpu';

const session = await ort.InferenceSession.create('/models/model.fp16.onnx', {
  executionProviders: ['webgpu'],
  graphOptimizationLevel: 'all',
});
```

In our own benchmark (LaMa 512×512 tiles, headless Chrome on Apple silicon),
WebGPU ran **0.86 s/tile versus 4.06 s/tile on WASM** — 4.7× — after switching
to fp16.

### WASM: usually slower

The WASM backend has no native fp16 compute. An fp16 model gets loaded, cast to
fp32 internally, and executed as before — you pay the decode cost plus casts. In
our measurements the same model was ~1.5× _slower_ on WASM than its fp32 twin.

Practical rule:

```ts
const webgpu = typeof navigator !== 'undefined' && 'gpu' in navigator;
const model = webgpu ? 'model.fp16.onnx' : 'model.onnx'; // keep both
const providers = webgpu ? ['webgpu'] : ['wasm'];
```

Feature-detect; do not assume `navigator.gpu` means a usable adapter (iOS
Safari's WebGPU support is recent and partial). Always ship an fp32 fallback for
CPU-only clients.

### Session startup checklist

- `onnxruntime-web` version and ORT tool versions drift; verify with
  `verifyModels` before shipping.
- Call `ort.env.wasm.numThreads` deliberately: threaded WASM needs
  cross-origin isolation headers (`Cross-Origin-Opener-Policy`,
  `Cross-Origin-Embedder-Policy`).
- Serve `.onnx` with `Content-Type: application/octet-stream` and long cache
  headers; the file is immutable per build.
- Keep the fp32 model for `verify` and for CPU devices; the fp16 copy is a
  delivery optimisation, not a replacement.

### A note on ops without fp16 kernels

`Range`, `Resize`, `TopK`, `NonMaxSuppression`, `RoiAlign`, `CumSum` and
`Upsample` have no fp16 kernels in ONNX Runtime. The converter keeps them in
fp32 and inserts `Cast`s around them by default — this is why a converted graph
still loads. If your runtime _does_ support them, pass
`defaultBlockOps: false` (or `--no-default-blocks`) to shave off the casts.

## Further reading

- [How it works](how-it-works.md) — where casts are placed and why
- [Case study: LaMa](case-study-lama.md) — a 208 MB model shipped at 102 MB
- ClearPix engineering notes:
  [fp16 in the browser](https://github.com/SharksevenRo/clearpix/blob/main/blogs/03-2026-10-01-fp16-onnx-webgpu-browser-inference.md),
  [the overflow bug](https://github.com/SharksevenRo/clearpix/blob/main/blogs/13-2026-10-17-fp16-quantization-quality-loss-overflowed-tensor.md)
