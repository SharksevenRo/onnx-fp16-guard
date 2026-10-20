# Security policy

## Scope

`onnx-fp16-guard` parses untrusted `.onnx` files. That is the attack surface:

- protobuf decoding,
- allocation based on declared tensor sizes,
- graph traversal and topological sorting.

The converter does not execute models, does not evaluate attributes, and does
not write files unless the CLI is told to (`convert -o …`).

## Reporting a vulnerability

Please **do not open a public issue** for anything that could crash, hang or
exhaust memory on a crafted model.

Use GitHub's private advisory flow:

<https://github.com/SharksevenRo/onnx-fp16-guard/security/advisories/new>

Include the model file (or a minimal generator), the command you ran, and the
observed behaviour. We will acknowledge within 7 days and aim to ship a fix or
mitigation within 30 days.

## Known limitations (not vulnerabilities)

- A model can legitimately declare enormous tensor sizes; the decoder will try
  to allocate them. Run the CLI with a memory limit
  (`node --max-old-space-size=4096`) when processing untrusted files, or inspect
  on a machine where that is acceptable.
- `verify` executes models with `onnxruntime-node`. ONNX Runtime itself is in
  scope of Microsoft's security process, not ours.
- The optional external-data check rejects models whose weights live in separate
  files; it does not follow those paths.

## Supported versions

Only the latest minor release receives security fixes.
