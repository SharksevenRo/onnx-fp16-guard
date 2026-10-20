/**
 * Human-readable rendering of reports for the CLI.
 *
 * Kept separate from the report types so that library consumers can format
 * things themselves, and so the CLI has zero dependencies.
 */
import type { ConvertReport } from './convert.js';
import type { InspectionReport } from './analyze.js';

export interface FormatOptions {
  color?: boolean;
  /** Show at most this many rows of tables (default 10). */
  maxRows?: number;
}

const ANSI = {
  reset: '\u001b[0m',
  bold: '\u001b[1m',
  dim: '\u001b[2m',
  red: '\u001b[31m',
  green: '\u001b[32m',
  yellow: '\u001b[33m',
  blue: '\u001b[34m',
  cyan: '\u001b[36m',
};

function paint(enabled: boolean, code: string, text: string): string {
  return enabled ? `${code}${text}${ANSI.reset}` : text;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = value >= 100 || unit === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

export function formatNumber(value: number): string {
  return value.toLocaleString('en-US');
}

function table(rows: string[][], aligns: ('left' | 'right')[], options: FormatOptions): string {
  if (rows.length === 0) return '';
  const widths = rows[0].map((_, column) => Math.max(...rows.map((row) => (row[column] ?? '').length)));
  const maxRows = options.maxRows ?? 12;
  const visible = rows.slice(0, maxRows);
  const lines = visible.map((row) =>
    row
      .map((cell, column) => {
        const padded =
          aligns[column] === 'right' ? cell.padStart(widths[column]) : cell.padEnd(widths[column]);
        return padded;
      })
      .join('  ')
      .trimEnd(),
  );
  if (rows.length > maxRows) lines.push(`… ${rows.length - maxRows} more`);
  return lines.join('\n');
}

const typeLabel = (report: InspectionReport) =>
  `${report.graphName || '<unnamed graph>'} · IR v${report.irVersion} · ${report.counts.nodes} nodes`;

export function formatInspection(report: InspectionReport, options: FormatOptions = {}): string {
  const color = options.color ?? false;
  const out: string[] = [];
  const bold = (text: string) => paint(color, ANSI.bold, text);

  out.push(bold('Model'));
  out.push(`  ${typeLabel(report)}`);
  if (report.opsets.length > 0) {
    out.push(`  opset    ${report.opsets.map((entry) => `${entry.domain}@${entry.version}`).join(', ')}`);
  }
  if (report.producer) out.push(`  producer ${report.producer}`);
  out.push(
    `  tensors  ${formatNumber(report.tensors.count)} (${formatNumber(report.counts.initializers)} initializers, ` +
      `${formatNumber(report.counts.constants)} Constant nodes)`,
  );

  out.push('');
  out.push(bold('I/O'));
  for (const input of report.io.inputs) {
    out.push(`  in   ${input.elemTypeName.padEnd(8)} ${input.name} [${input.dims.join(', ')}]`);
  }
  for (const output of report.io.outputs) {
    out.push(`  out  ${output.elemTypeName.padEnd(8)} ${output.name} [${output.dims.join(', ')}]`);
  }

  out.push('');
  out.push(bold('Storage'));
  const bucketRows = report.tensors.buckets.map((bucket) => [
    bucket.name,
    formatNumber(bucket.count),
    formatBytes(bucket.bytes),
    report.tensors.payloadBytes > 0
      ? `${((bucket.bytes / report.tensors.payloadBytes) * 100).toFixed(1)}%`
      : '—',
  ]);
  out.push(
    table(
      [['dtype', 'tensors', 'bytes', 'share'], ...bucketRows],
      ['left', 'right', 'right', 'right'],
      options,
    ),
  );
  out.push(
    `  payload ${formatBytes(report.tensors.payloadBytes)} → project fp16 ` +
      `${formatBytes(report.tensors.projectedPayloadBytes)} ` +
      `(${report.tensors.float32Count} float32 tensor(s) to convert)`,
  );

  out.push('');
  if (report.overflow.findings.length === 0) {
    out.push(paint(color, ANSI.green, `✔ no tensor exceeds the fp16 range (${report.overflow.limit})`));
  } else {
    out.push(
      paint(
        color,
        ANSI.red,
        `✖ ${report.overflow.findings.length} tensor(s) exceed the fp16 range (${report.overflow.limit})`,
      ),
    );
    const rows = report.overflow.findings.map((finding) => [
      finding.tensor,
      finding.absMax.toExponential(2),
      finding.kind,
      finding.fix,
    ]);
    out.push(
      table([['tensor', 'max|value|', 'kind', 'fix'], ...rows], ['left', 'right', 'left', 'left'], options),
    );
    if (report.overflow.batchNorm.overflowing > 0) {
      out.push(
        `  ${report.overflow.batchNorm.overflowing} of ${report.overflow.batchNorm.total} BatchNormalization ` +
          `layer(s) need attention`,
      );
    }
  }

  if (report.ops.length > 0) {
    out.push('');
    out.push(bold('Ops'));
    const rows = report.ops.map((entry) => [entry.opType, formatNumber(entry.count)]);
    out.push(
      table([['op', 'count'], ...rows], ['left', 'right'], { ...options, maxRows: options.maxRows ?? 8 }),
    );
  }

  if (report.warnings.length > 0) {
    out.push('');
    out.push(paint(color, ANSI.yellow, 'Warnings'));
    for (const warning of report.warnings) out.push(`  ! ${warning}`);
  }

  return out.join('\n');
}

export function formatConversion(report: ConvertReport, options: FormatOptions = {}): string {
  const color = options.color ?? false;
  const out: string[] = [];
  const bold = (text: string) => paint(color, ANSI.bold, text);
  const saved = report.bytesSaved;
  const percent = report.inputBytes > 0 ? (saved / report.inputBytes) * 100 : 0;

  out.push(bold('Converted'));
  out.push(
    `  ${formatBytes(report.inputBytes)} → ${formatBytes(report.outputBytes)} ` +
      paint(color, ANSI.green, `(−${formatBytes(saved)}, ${percent.toFixed(1)}%)`),
  );
  out.push(
    `  ${formatNumber(report.convertedTensors)} initializer(s) + ${formatNumber(report.constantsConverted)} ` +
      `Constant tensor(s) → float16, ${formatNumber(report.insertedCasts)} Cast node(s) inserted`,
  );
  out.push(`  ratio ${(report.ratio * 100).toFixed(1)}% of the original file`);

  if (report.rescaled.length > 0) {
    out.push('');
    out.push(bold('BatchNorm rescale'));
    const rows = report.rescaled.map((change) => [
      change.node,
      change.k.toExponential(0),
      change.maxVarianceBefore.toExponential(2),
      change.maxVarianceAfter.toExponential(2),
    ]);
    out.push(
      table(
        [['node', 'K', 'max|var| before', 'after'], ...rows],
        ['left', 'right', 'right', 'right'],
        options,
      ),
    );
  }

  if (report.blockedNodes.length > 0) {
    out.push('');
    out.push(bold(`${report.blockedNodes.length} node(s) kept in fp32`));
    for (const name of report.blockedNodes.slice(0, options.maxRows ?? 10)) out.push(`  · ${name}`);
    if (report.blockedNodes.length > (options.maxRows ?? 10)) {
      out.push(`  … ${report.blockedNodes.length - (options.maxRows ?? 10)} more`);
    }
  }

  if (report.keptFloat32.length > 0) {
    out.push('');
    out.push(bold(`${report.keptFloat32.length} tensor(s) kept in fp32`));
    for (const entry of report.keptFloat32.slice(0, options.maxRows ?? 10)) {
      out.push(`  · ${entry.name} (${entry.reason})`);
    }
  }

  if (report.warnings.length > 0) {
    out.push('');
    out.push(paint(color, ANSI.yellow, 'Warnings'));
    for (const warning of report.warnings) out.push(`  ! ${warning}`);
  }

  return out.join('\n');
}
