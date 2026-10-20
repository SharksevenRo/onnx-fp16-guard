import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { runCli } from '../src/cli.js';
import { buildBatchNormModel, serialize } from './helpers/models.js';

const directory = mkdtempSync(join(tmpdir(), 'onnx-fp16-guard-'));
const modelPath = join(directory, 'fixture.onnx');
writeFileSync(modelPath, serialize(buildBatchNormModel()));

afterAll(() => {
  rmSync(directory, { recursive: true, force: true });
});

async function run(args: string[]): Promise<{ code: number; out: string; err: string }> {
  const out: string[] = [];
  const err: string[] = [];
  const code = await runCli(args, {
    out: (text) => out.push(text),
    err: (text) => err.push(text),
    color: false,
  });
  return { code, out: out.join('\n'), err: err.join('\n') };
}

describe('cli', () => {
  it('prints usage', async () => {
    const { code, out } = await run(['--help']);
    expect(code).toBe(0);
    expect(out).toContain('onnx-fp16-guard');
    expect(out).toContain('Usage:');
  });

  it('inspects a model as text', async () => {
    const { code, out } = await run(['inspect', modelPath]);
    expect(code).toBe(0);
    expect(out).toMatch(/exceed the fp16 range/);
    expect(out).toContain('bn1.var');
    expect(out).toContain('rescale-batchnorm');
  });

  it('inspects a model as JSON', async () => {
    const { code, out } = await run(['inspect', modelPath, '--json']);
    expect(code).toBe(0);
    const report = JSON.parse(out) as {
      overflow: { findings: { tensor: string; k: number }[] };
      fileBytes: number;
      counts: { nodes: number };
    };
    expect(report.fileBytes).toBeGreaterThan(0);
    expect(report.overflow.findings[0].tensor).toBe('bn1.var');
    expect(report.overflow.findings[0].k).toBe(16);
  });

  it('fails a CI guard when a tensor overflows', async () => {
    const { code, err } = await run(['inspect', modelPath, '--fail-on-overflow']);
    expect(code).toBe(2);
    expect(err).toMatch(/exceed the fp16 range/);
  });

  it('converts a model and refuses to overwrite it', async () => {
    const output = join(directory, 'converted.onnx');
    const first = await run(['convert', modelPath, '-o', output, '--json']);
    expect(first.code).toBe(0);
    const report = JSON.parse(first.out) as {
      output: string;
      outputBytes: number;
      inputBytes: number;
      rescaled: { k: number }[];
    };
    expect(existsSync(output)).toBe(true);
    expect(report.outputBytes).toBe(readFileSync(output).byteLength);
    expect(report.rescaled).toHaveLength(1);
    expect(report.rescaled[0].k).toBe(16);

    const second = await run(['convert', modelPath, '-o', output]);
    expect(second.code).toBe(1);
    expect(second.err).toMatch(/refusing to overwrite/);

    const forced = await run(['convert', modelPath, '-o', output, '--force', '--quiet']);
    expect(forced.code).toBe(0);
    expect(forced.out.trim()).toBe(output);
  });

  it('writes a JSON report next to the model', async () => {
    const output = join(directory, 'reported.onnx');
    const reportPath = join(directory, 'report.json');
    const { code } = await run(['convert', modelPath, '-o', output, '--report', reportPath, '--quiet']);
    expect(code).toBe(0);
    const report = JSON.parse(readFileSync(reportPath, 'utf8')) as { insertedCasts: number; ratio: number };
    expect(report.insertedCasts).toBeGreaterThan(0);
    expect(report.ratio).toBeGreaterThan(0);
  });

  it('exits with 2 when overflow=error finds an overflow', async () => {
    const { code, err } = await run([
      'convert',
      modelPath,
      '-o',
      join(directory, 'nope.onnx'),
      '--overflow',
      'error',
    ]);
    expect(code).toBe(2);
    expect(err).toMatch(/exceed the fp16 range/);
    expect(existsSync(join(directory, 'nope.onnx'))).toBe(false);
  });

  it('blocks nodes by pattern', async () => {
    const output = join(directory, 'blocked.onnx');
    const { code, out } = await run([
      'convert',
      modelPath,
      '-o',
      output,
      '--no-keep-io',
      '--block-node',
      'bn*',
      '--json',
    ]);
    expect(code).toBe(0);
    const report = JSON.parse(out) as { blockedNodes: string[]; insertedCasts: number };
    expect(report.blockedNodes.sort()).toEqual(['bn1', 'bn2']);
    expect(report.insertedCasts).toBeGreaterThan(0);
  });

  it('rejects unknown commands and bad arguments', async () => {
    expect((await run(['frobnicate'])).code).toBe(1);
    expect((await run(['inspect'])).code).toBe(1);
    expect((await run(['inspect', modelPath, '--nope'])).code).toBe(1);
    expect((await run(['convert', modelPath, '--overflow', 'sometimes'])).code).toBe(1);
  });

  it('reports a missing file clearly', async () => {
    const { code, err } = await run(['inspect', join(directory, 'missing.onnx')]);
    expect(code).toBe(1);
    expect(err).toMatch(/cannot read/);
  });
});
