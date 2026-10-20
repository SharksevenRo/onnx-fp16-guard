/**
 * Typed, hand-written view of the ONNX protobuf schema.
 *
 * Why not use the generated `onnx-proto` interfaces directly? Because they mark
 * every field as optional *and* nullable, which turns a 400-line converter into
 * a sea of `!` and `?.` without buying any safety: protobufjs always initializes
 * repeated fields to `[]`. The interfaces below describe the subset this package
 * reads and writes, with the fields that are guaranteed after `decode()` marked
 * as required.
 *
 * The objects are the real decoded messages — these are *structural* types, so
 * every field we do not mention is still preserved on encode.
 */
import onnxProto from 'onnx-proto';

// eslint-disable-next-line @typescript-eslint/consistent-type-imports
type OnnxModule = typeof import('onnx-proto');

/** The generated ONNX namespace (`ModelProto.decode` / `.encode`). */
export const onnx = (onnxProto as unknown as OnnxModule).onnx;

/** Just enough of protobufjs' `Long` to convert it to a number. */
export interface LongLike {
  toNumber(): number;
  toString(): string;
}

export interface TensorProto {
  name: string;
  dataType: number;
  dims: (number | LongLike)[];
  rawData?: Uint8Array;
  floatData?: number[];
  doubleData?: number[];
  int32Data?: number[];
  int64Data?: number[];
  uint64Data?: number[];
  externalData?: { key?: string | null; value?: string | null }[];
}

export interface AttributeProto {
  name?: string;
  /** `onnx.AttributeProto.AttributeType` */
  type?: number;
  f?: number;
  i?: number | LongLike;
  s?: Uint8Array;
  t?: TensorProto;
  floats?: number[];
  ints?: (number | LongLike)[];
  strings?: Uint8Array[];
  tensors?: TensorProto[];
}

export interface NodeProto {
  name: string;
  opType: string;
  input: string[];
  output: string[];
  attribute: AttributeProto[];
  domain?: string;
}

export interface Dimension {
  dimValue?: number | LongLike | null;
  dimParam?: string | null;
}

export interface TensorTypeProto {
  elemType?: number;
  shape?: { dim?: Dimension[] } | null;
}

export interface TypeProto {
  tensorType?: TensorTypeProto | null;
}

export interface ValueInfoProto {
  name: string;
  type?: TypeProto;
}

export interface GraphProto {
  name: string;
  node: NodeProto[];
  initializer: TensorProto[];
  input: ValueInfoProto[];
  output: ValueInfoProto[];
  valueInfo: ValueInfoProto[];
  sparseInitializer?: unknown[];
}

export interface OpsetId {
  domain?: string;
  version?: number | LongLike;
}

export interface ModelProto {
  irVersion?: number | LongLike;
  opsetImport: OpsetId[];
  producerName?: string;
  producerVersion?: string;
  graph: GraphProto;
  metadataProps?: { key: string; value: string }[];
}

/** Serialized shape accepted by `onnx.ModelProto.encode`. */
// eslint-disable-next-line @typescript-eslint/consistent-type-imports
export type PbModelProto = import('onnx-proto').onnx.IModelProto;

/** Subset of `TensorProto.DataType` we care about. */
export const DataType = {
  FLOAT: 1,
  UINT8: 2,
  INT8: 3,
  INT32: 6,
  INT64: 7,
  FLOAT16: 10,
  DOUBLE: 11,
  BFLOAT16: 16,
} as const;

/** Subset of `AttributeProto.AttributeType`. */
export const AttrType = {
  FLOAT: 1,
  INT: 2,
  STRING: 3,
  TENSOR: 4,
  FLOATS: 6,
  INTS: 7,
  TENSORS: 10,
} as const;

/** Largest finite value representable in binary16. */
export const FP16_MAX = 65504;

export const DATA_TYPE_NAMES: Record<number, string> = {
  1: 'float32',
  2: 'uint8',
  3: 'int8',
  4: 'uint16',
  5: 'int16',
  6: 'int32',
  7: 'int64',
  8: 'string',
  9: 'bool',
  10: 'float16',
  11: 'float64',
  12: 'uint32',
  13: 'uint64',
  14: 'complex64',
  15: 'complex128',
  16: 'bfloat16',
};

export function dataTypeName(type: number): string {
  return DATA_TYPE_NAMES[type] ?? `unknown(${type})`;
}
