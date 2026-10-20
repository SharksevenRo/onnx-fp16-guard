/**
 * onnx-fp16-guard — safe fp16 conversion for ONNX models.
 *
 * ```ts
 * import { convertBytes, inspectModel, decodeModel } from 'onnx-fp16-guard';
 *
 * const report = inspectModel(decodeModel(bytes));
 * if (report.overflow.findings.length > 0) { … }
 *
 * const { bytes: fp16, report: conversion } = convertBytes(bytes);
 * ```
 *
 * The package is ESM, has no Node built-in imports in its core, and runs in the
 * browser as well as in Node — see `docs/browser.md`.
 */
export { FP16_MAX, DataType, AttrType, dataTypeName } from './onnx.js';
export type { ModelProto, GraphProto, NodeProto, TensorProto, ValueInfoProto, TypeProto } from './onnx.js';

export {
  float32ToFloat16,
  float32ToFloat16Bits,
  float16ToFloat32,
  float16BitsToFloat32,
  bytesPerElement,
} from './fp16.js';

export {
  decodeModel,
  encodeModel,
  tensorDims,
  tensorElementCount,
  tensorBytes,
  tensorAbsMax,
  readFloat32Tensor,
  readFloat16Tensor,
  writeFloat32Tensor,
  writeFloat16Bits,
  collectProducers,
  collectConsumers,
  renameNodeInputs,
  renameNodeOutputs,
  topologicalSort,
  validateGraph,
  elemTypeOf,
  setElemType,
  makeCast,
  initializerBytes,
  toNumber,
} from './model.js';

export { inspectModel, rescaleFactor } from './analyze.js';
export type {
  InspectionReport,
  InspectOptions,
  OverflowFinding,
  OverflowFix,
  DataTypeBucket,
  IoInfo,
  OpsetInfo,
} from './analyze.js';

export { DEFAULT_TARGET_MAX, findBatchNormOverflows, rescaleBatchNorms, assertNoOverflow } from './bn.js';
export type { BatchNormOverflow, RescaleChange } from './bn.js';

export { convertModel, convertBytes, matchesPattern, OverflowError, DEFAULT_FP32_OPS } from './convert.js';
export type {
  ConvertOptions,
  ConvertReport,
  ConvertResult,
  OverflowStrategy,
  OverflowErrorEntry,
} from './convert.js';

export { verifyModels, describeInputs } from './verify.js';
export type { VerifyOptions, VerifyReport, OutputComparison } from './verify.js';

export { formatInspection, formatConversion, formatBytes, formatNumber } from './format.js';
export type { FormatOptions } from './format.js';
