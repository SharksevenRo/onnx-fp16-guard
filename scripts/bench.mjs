#!/usr/bin/env node
/**
 * Benchmark / smoke-test harness.
 *
 *   node scripts/bench.mjs model.onnx [--out model.fp16.onnx] [--verify] [--json]
 *
 * Requires a build first (`npm run build`). Measures inspect + convert time,
 * prints before/after sizes, and optionally runs `verifyModels` (needs
 * onnxruntime-node). Not part of the published package.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { convertBytes, decodeModel, inspectModel, verifyModels } from '../dist/index.js';

const args = process.argv.slice(2);
const input = args.find((arg) => !arg.startsWith('--'));
if (!input) {
  console.error('usage: node scripts/bench.mjs <model.onnx> [--out <path>] [--verify] [--json]');
  process.exit(1);
}

const flag = (name) => args.includes(name);
const option = (name) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};

const source = readFileSync(input);

const t0 = performance.now();
const inspection = inspectModel(decodeModel(source));
const inspectMs = performance.now() - t0;

const t1 = performance.now();
const { bytes, report } = convertBytes(source);
const convertMs = performance.now() - t1;

const out = option('--out') ?? join(process.cwd(), basename(input).replace(/\.onnx$/, '') + '.fp16.onnx');
writeFileSync(out, bytes);

const summary = {
  input,
  output: out,
  inputBytes: source.length,
  outputBytes: bytes.length,
  savedPercent: Number((((source.length - bytes.length) / source.length) * 100).toFixed(1)),
  inspectMs: Number(inspectMs.toFixed(0)),
  convertMs: Number(convertMs.toFixed(0)),
  overflowFindings: inspection.overflow.findings.length,
  rescaled: report.rescaled.length,
  blockedNodes: report.blockedNodes.length,
  keptFloat32: report.keptFloat32.length,
  insertedCasts: report.insertedCasts,
};

if (flag('--verify')) {
  const t2 = performance.now();
  try {
    const verification = await verifyModels(source, bytes, { minPsnrDb: 30 });
    summary.verifyMs = Number((performance.now() - t2).toFixed(0));
    summary.worstPsnrDb = Number(verification.worstPsnrDb.toFixed(2));
    summary.verified = verification.passed;
  } catch (error) {
    summary.verified = false;
    summary.verifyError = error.message;
  }
}

if (flag('--json')) {
  console.log(JSON.stringify(summary, null, 2));
} else {
  const mib = (value) => (value / 2 ** 20).toFixed(1);
  console.log(`${basename(input)}`);
  console.log(
    `  ${mib(summary.inputBytes)} MiB → ${mib(summary.outputBytes)} MiB  (−${summary.savedPercent}%)`,
  );
  console.log(`  inspect ${summary.inspectMs} ms · convert ${summary.convertMs} ms`);
  console.log(
    `  ${summary.rescaled} BatchNorm rescale(s) · ${summary.blockedNodes} blocked node(s) · ` +
      `${summary.keptFloat32} fp32 tensor(s) · ${summary.insertedCasts} cast(s)`,
  );
  if (summary.verifyError) console.log(`  verify skipped: ${summary.verifyError.split('\n')[0]}`);
  if (summary.worstPsnrDb !== undefined) {
    console.log(`  verify ${summary.verified ? 'PASS' : 'FAIL'}: ${summary.worstPsnrDb} dB`);
  }
  console.log(`  wrote ${out}`);
}
