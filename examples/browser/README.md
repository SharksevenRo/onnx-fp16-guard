# Browser example

A drag-and-drop converter: drop an `.onnx` file, see what is in it, convert it to
fp16, download the result. Everything happens locally — the model is never
uploaded.

## Run it

```bash
# from the repository root
npm run build
npx serve .          # or: python3 -m http.server 8000
```

Open <http://localhost:3000/examples/browser/>.

ES modules cannot be imported from `file://`, so a local server is required.
The page imports `../../dist/index.js`, which is why `npm run build` comes first.

## What it demonstrates

1. `decodeModel()` + `inspectModel()` — read-only analysis, including the
   projected fp16 size and every tensor outside the fp16 range.
2. `convertBytes()` — the full conversion, in the page, with no Worker.
3. A download link built from a `Blob` (no server round-trip).

## Production notes

- **Memory.** Peak usage is roughly 3× the model size. For models above ~100 MB,
  move the conversion into a Web Worker so the UI stays responsive:

  ```js
  // worker.js
  import { convertBytes } from 'onnx-fp16-guard';
  self.onmessage = ({ data }) => {
    const { bytes, report } = convertBytes(new Uint8Array(data));
    self.postMessage({ bytes, report }, [bytes.buffer]);
  };
  ```

- **Caching.** Store the converted bytes in the Cache API or IndexedDB keyed by
  the source model hash, so a reload does not re-convert.
- **Correctness.** Run `verifyModels()` once in CI (Node) on the same model; the
  browser demo intentionally does not depend on onnxruntime-web.

## Serving the converted model

If you then run the model with `onnxruntime-web`, read
[../../docs/browser.md](../../docs/browser.md) first: fp16 is a win on WebGPU and
a loss on WASM.
