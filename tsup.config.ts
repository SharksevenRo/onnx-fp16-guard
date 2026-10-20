import { defineConfig } from 'tsup';

export default defineConfig({
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
});
