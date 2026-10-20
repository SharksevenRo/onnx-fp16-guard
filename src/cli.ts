#!/usr/bin/env node
/**
 * Command line interface.
 *
 * `runCli()` is exported and dependency-injected so tests can drive it without
 * spawning a process or building the package first.
 */
import { readFileSync, realpathSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { convertModel, OverflowError, type ConvertOptions } from './convert.js';
import { decodeModel, encodeModel } from './model.js';
import { inspectModel } from './analyze.js';
import { formatConversion, formatBytes, formatInspection } from './format.js';
import { verifyModels } from './verify.js';

export interface CliIo {
  out(text: string): void;
  err(text: string): void;
  /** Defaults to `process.stdout.isTTY`. */
  color?: boolean;
}

const USAGE = `onnx-fp16-guard — safe fp16 conversion for ONNX models

Usage:
  onnx-fp16-guard inspect <model.onnx> [options]
  onnx-fp16-guard convert <model.onnx> [options]
  onnx-fp16-guard verify <reference.onnx> <candidate.onnx> [options]

Inspect:
  --json                 print the full report as JSON
  --fail-on-overflow     exit with code 2 if any tensor exceeds the fp16 range
  --limit <n>            fp16 range limit (default 65504)
  --target-max <n>       target max |var| for BatchNorm rescale (default 40000)

Convert:
  -o, --out <path>       output file (default: <model>.fp16.onnx)
  --overflow <mode>      rescale | block | error (default: rescale)
  --no-keep-io           convert graph inputs/outputs to fp16 as well
  --block-op <op>        keep every node with this op type in fp32 (repeatable)
  --block-node <name>    keep matching node names in fp32 (glob or /regex/)
  --no-default-blocks    allow fp16 for Range/Resize/TopK (no fp16 kernels in ORT)
  --no-constants         keep Constant node tensors in fp32
  --no-toposort          skip the topological sort
  --no-validate          skip structural validation
  --report <path>        write the conversion report as JSON
  --json                 print the conversion report as JSON
  --force                overwrite an existing output file
  --quiet                suppress the summary

Verify (requires onnxruntime-node):
  --min-psnr <db>        pass threshold (default 30)
  --seed <n>             PRNG seed for generated inputs (default 42)
  --json                 print the report as JSON

Exit codes: 0 success · 1 error · 2 overflow found · 3 verification failed
`;

function colorEnabled(io: CliIo): boolean {
  if (io.color !== undefined) return io.color;
  if (process.env.NO_COLOR) return false;
  return Boolean(process.stdout.isTTY);
}

function defaultOutputPath(input: string): string {
  const extension = extname(input);
  const base = basename(input, extension);
  return join(dirname(input), `${base}.fp16${extension || '.onnx'}`);
}

function readModel(path: string): Uint8Array {
  let stats;
  try {
    stats = statSync(path);
  } catch {
    throw new Error(`cannot read "${path}": no such file`);
  }
  if (!stats.isFile()) throw new Error(`"${path}" is not a file`);
  return readFileSync(path);
}

function numberOption(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`expected a number, received "${value}"`);
  return parsed;
}

function parseIntegerOption(value: string | undefined, fallback: number): number {
  const parsed = Math.trunc(numberOption(value, fallback));
  return parsed;
}

export async function runCli(argv: string[], io: CliIo): Promise<number> {
  if (argv.length === 0 || argv.includes('--help') || argv.includes('-h') || argv[0] === 'help') {
    io.out(USAGE);
    return 0;
  }
  if (argv.includes('--version') || argv.includes('-v')) {
    io.out(readVersion());
    return 0;
  }

  const [command, ...rest] = argv;
  if (command !== 'inspect' && command !== 'convert' && command !== 'verify') {
    io.err(`Unknown command "${command}".\n\n${USAGE}`);
    return 1;
  }

  let parsed;
  try {
    parsed = parseArgs({
      args: rest,
      allowPositionals: true,
      strict: true,
      options: {
        out: { type: 'string', short: 'o' },
        json: { type: 'boolean', default: false },
        'fail-on-overflow': { type: 'boolean', default: false },
        overflow: { type: 'string' },
        'keep-io': { type: 'boolean' },
        'no-keep-io': { type: 'boolean', default: false },
        'block-op': { type: 'string', multiple: true },
        'block-node': { type: 'string', multiple: true },
        'no-default-blocks': { type: 'boolean', default: false },
        constants: { type: 'boolean' },
        'no-constants': { type: 'boolean', default: false },
        toposort: { type: 'boolean' },
        'no-toposort': { type: 'boolean', default: false },
        validate: { type: 'boolean' },
        'no-validate': { type: 'boolean', default: false },
        report: { type: 'string' },
        force: { type: 'boolean', default: false },
        quiet: { type: 'boolean', default: false },
        limit: { type: 'string' },
        'target-max': { type: 'string' },
        'min-psnr': { type: 'string' },
        seed: { type: 'string' },
        amplitude: { type: 'string' },
        help: { type: 'boolean', default: false },
        version: { type: 'boolean', default: false },
      },
    });
  } catch (error) {
    io.err(`${(error as Error).message}\n\nRun \`onnx-fp16-guard --help\` for usage.`);
    return 1;
  }

  const { values, positionals } = parsed;
  const color = colorEnabled(io);

  try {
    if (command === 'inspect') {
      const [modelPath] = positionals;
      if (!modelPath) {
        io.err('inspect requires a model path.\n\nUsage: onnx-fp16-guard inspect <model.onnx>');
        return 1;
      }
      const bytes = readModel(modelPath);
      const model = decodeModel(bytes);
      const report = inspectModel(model, {
        limit: numberOption(values.limit as string | undefined, 65504),
        targetMax: numberOption(values['target-max'] as string | undefined, 40_000),
      });
      report.fileBytes = bytes.length;

      if (values.json) {
        io.out(JSON.stringify({ ...report, file: resolve(modelPath) }, null, 2));
      } else {
        io.out(formatInspection(report, { color }));
        io.out(`\n  file ${basename(modelPath)} · ${formatBytes(bytes.length)}`);
      }

      if (values['fail-on-overflow'] && report.overflow.findings.length > 0) {
        io.err(
          `\n${report.overflow.findings.length} tensor(s) exceed the fp16 range; ` +
            `converting this model with a naive converter would corrupt it.`,
        );
        return 2;
      }
      return 0;
    }

    if (command === 'convert') {
      const [modelPath] = positionals;
      if (!modelPath) {
        io.err('convert requires a model path.\n\nUsage: onnx-fp16-guard convert <model.onnx> -o <out.onnx>');
        return 1;
      }

      const overflow = (values.overflow as string | undefined) ?? 'rescale';
      if (overflow !== 'rescale' && overflow !== 'block' && overflow !== 'error') {
        io.err(`--overflow must be rescale, block or error (received "${overflow}")`);
        return 1;
      }

      const options: ConvertOptions = {
        keepIoTypes: values['no-keep-io'] ? false : true,
        overflow,
        limit: numberOption(values.limit as string | undefined, 65504),
        targetMax: numberOption(values['target-max'] as string | undefined, 40_000),
        blockOps: (values['block-op'] as string[] | undefined) ?? [],
        blockNodes: (values['block-node'] as string[] | undefined) ?? [],
        defaultBlockOps: values['no-default-blocks'] ? false : true,
        convertConstants: values['no-constants'] ? false : true,
        toposort: values['no-toposort'] ? false : true,
        validate: values['no-validate'] ? false : true,
      };

      const bytes = readModel(modelPath);
      const outputPath = resolve((values.out as string | undefined) ?? defaultOutputPath(modelPath));

      if (!values.force) {
        try {
          statSync(outputPath);
          io.err(`refusing to overwrite "${outputPath}" (pass --force)`);
          return 1;
        } catch {
          // does not exist — good
        }
      }

      const started = Date.now();
      const { model, report } = convertModel(decodeModel(bytes), options);
      const encoded = encodeModel(model);
      const elapsed = Date.now() - started;
      writeFileSync(outputPath, encoded);

      const finalReport = {
        ...report,
        inputBytes: bytes.length,
        outputBytes: encoded.length,
        bytesSaved: bytes.length - encoded.length,
        ratio: encoded.length / bytes.length,
      };

      if (values.report)
        writeFileSync(resolve(values.report as string), JSON.stringify(finalReport, null, 2));

      if (values.json) {
        io.out(JSON.stringify({ ...finalReport, output: outputPath, elapsedMs: elapsed }, null, 2));
      } else if (!values.quiet) {
        io.out(formatConversion(finalReport, { color }));
        io.out(`\n  wrote ${outputPath} in ${(elapsed / 1000).toFixed(1)}s`);
      } else {
        io.out(outputPath);
      }
      return 0;
    }

    // verify
    const [referencePath, candidatePath] = positionals;
    if (!referencePath || !candidatePath) {
      io.err(
        'verify requires two model paths.\n\nUsage: onnx-fp16-guard verify <reference.onnx> <candidate.onnx>',
      );
      return 1;
    }

    const report = await verifyModels(readModel(referencePath), readModel(candidatePath), {
      minPsnrDb: numberOption(values['min-psnr'] as string | undefined, 30),
      seed: parseIntegerOption(values.seed as string | undefined, 42),
      amplitude: numberOption(values.amplitude as string | undefined, 1),
    });

    if (values.json) {
      io.out(JSON.stringify(report, null, 2));
    } else {
      io.out(`onnxruntime-node ${report.runtimeVersion}`);
      for (const output of report.outputs) {
        io.out(
          `  ${output.name}: PSNR ${output.psnrDb.toFixed(2)} dB · max|Δ| ${output.maxAbsDiff.toExponential(2)} · ` +
            `cosine ${output.cosineSimilarity.toFixed(6)}`,
        );
      }
      io.out(
        report.passed
          ? `\nPass: worst PSNR ${report.worstPsnrDb.toFixed(2)} dB ≥ ${report.thresholdDb} dB`
          : `\nFail: worst PSNR ${report.worstPsnrDb.toFixed(2)} dB < ${report.thresholdDb} dB`,
      );
    }
    return report.passed ? 0 : 3;
  } catch (error) {
    if (error instanceof OverflowError) {
      io.err(error.message);
      return 2;
    }
    io.err(`Error: ${(error as Error).message}`);
    return 1;
  }
}

function readVersion(): string {
  try {
    const packageJson = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as {
      version?: string;
    };
    return packageJson.version ?? '0.0.0';
  } catch {
    return '0.0.0';
  }
}

/**
 * Only run when this file is the process entry point. Tests import `runCli`,
 * and package managers invoke the bin through a symlink — compare *real* paths,
 * otherwise `npx onnx-fp16-guard` silently does nothing.
 */
function isEntryPoint(): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return realpathSync(entry) === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
}

if (isEntryPoint()) {
  runCli(process.argv.slice(2), {
    out: (text) => process.stdout.write(`${text}\n`),
    err: (text) => process.stderr.write(`${text}\n`),
  }).then(
    (code) => {
      process.exitCode = code;
    },
    (error: unknown) => {
      process.stderr.write(`Error: ${(error as Error).message}\n`);
      process.exitCode = 1;
    },
  );
}
