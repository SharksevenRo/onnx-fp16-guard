import { defineConfig } from 'tsup';

export default defineConfig([
  {
    entry: {
      index: 'src/index.ts',
      cli: 'src/cli.ts',
    },
    format: ['esm'],
    platform: 'neutral',
    target: 'es2020',
    // `node:` built-ins are only used by the CLI entry; keeping them external is
    // what makes `dist/index.js` importable from a browser bundle.
    external: [/^node:/, 'onnxruntime-node'],
    dts: true,
    clean: true,
    splitting: false,
    sourcemap: true,
    treeshake: true,
  },
  {
    // Browser bundle with `onnx-proto` inlined, so the example page works as a
    // plain static site (GitHub Pages) — no bundler, no import map, no CDN.
    entry: {
      browser: 'src/index.ts',
    },
    format: ['esm'],
    platform: 'browser',
    target: 'es2020',
    noExternal: ['onnx-proto'],
    external: [/^node:/, 'onnxruntime-node'],
    dts: false,
    clean: false,
    splitting: false,
    sourcemap: true,
    treeshake: true,
  },
]);
