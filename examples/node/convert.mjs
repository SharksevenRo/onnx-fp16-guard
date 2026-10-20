#!/usr/bin/env node
/**
 * Convert every .onnx file in a directory — the "nightly job" example.
 *
 *   npm install onnx-fp16-guard
 *   node convert.mjs ./models [--out ./models-fp16] [--overflow rescale|block|error]
 *                             [--verify]
 *
 * Exit code: 0 all files converted · 1 a file failed · 2 a model overflows and
 * --overflow=error was requested.
 */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { convertBytes, OverflowError, verifyModels } from 'onnx-fp16-guard';

const args = process.argv.slice(2);
const inputDir = resolve(args.find((arg) => !arg.startsWith('--')) ?? '.');
const flag = (name) => args.includes(name);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};

const outputDir = resolve(option('out', inputDir));
const overflow = option('overflow', 'rescale');
const verify = flag('--verify');

if (!['rescale', 'block', 'error'].includes(overflow)) {
  console.error(`--overflow must be rescale, block or error (got "${overflow}")`);
  process.exit(1);
}

mkdirSync(outputDir, { recursive: true });

const files = readdirSync(inputDir)
  .filter((name) => name.endsWith('.onnx'))
  .sort();

if (files.length === 0) {
  console.error(`no .onnx files in ${inputDir}`);
  process.exit(1);
}

let failures = 0;
let overflowFailures = 0;
let savedBytes = 0;
let inputBytes = 0;

for (const name of files) {
  const source = readFileSync(join(inputDir, name));
  const target = join(outputDir, basename(name).replace(/\.onnx$/, '.fp16.onnx'));
  inputBytes += source.length;

  try {
    const started = performance.now();
    const { bytes, report } = convertBytes(source, { overflow });
    const elapsed = ((performance.now() - started) / 1000).toFixed(2);

    writeFileSync(target, bytes);
    savedBytes += report.bytesSaved;

    console.log(
      `${name.padEnd(40)} ${(source.length / 2 ** 20).toFixed(1).padStart(6)} MiB → ` +
        `${(bytes.length / 2 ** 20).toFixed(1).padStart(6)} MiB  ` +
        `(${((1 - report.ratio) * 100).toFixed(1)}% smaller, ${elapsed}s)`,
    );
    if (report.rescaled.length > 0) {
      const factors = [...new Set(report.rescaled.map((change) => change.k))].sort((a, b) => a - b);
      console.log(`  ${report.rescaled.length} BatchNorm layer(s) rescaled, K = ${factors.join(', ')}`);
    }
    if (report.blockedNodes.length > 0) {
      console.log(`  ${report.blockedNodes.length} node(s) kept in fp32`);
    }

    if (verify) {
      // onnxruntime-node is optional; this throws with an install hint if missing.
      const result = await verifyModels(source, bytes, { minPsnrDb: 30 });
      console.log(
        `  verify ${result.passed ? 'PASS' : 'FAIL'} · worst PSNR ${result.worstPsnrDb.toFixed(2)} dB`,
      );
      if (!result.passed) failures += 1;
    }
  } catch (error) {
    if (error instanceof OverflowError) {
      overflowFailures += 1;
      console.error(`${name}: ${error.tensors.length} tensor(s) exceed the fp16 range`);
      for (const tensor of error.tensors.slice(0, 5)) {
        console.error(`  ${tensor.name} max|value| = ${tensor.absMax.toExponential(3)}`);
      }
    } else {
      failures += 1;
      console.error(`${name}: ${error.message}`);
    }
  }
}

console.log(
  `\n${files.length} file(s) · ${(inputBytes / 2 ** 20).toFixed(1)} MiB in · ` +
    `${(savedBytes / 2 ** 20).toFixed(1)} MiB saved`,
);
if (failures > 0) process.exit(1);
if (overflowFailures > 0) process.exit(2);
