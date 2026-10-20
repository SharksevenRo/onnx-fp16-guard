## What this changes

<!-- One paragraph. Describe the failure mode, not just the diff. -->

## Why

<!-- Link the issue, or explain what breaks without this. -->

## Checklist

- [ ] `npm run typecheck` passes
- [ ] `npm run lint` passes
- [ ] `npm run format:check` passes
- [ ] `npm test` passes
- [ ] Added a test that fails without this change (or explained why not)
- [ ] Updated `CHANGELOG.md` under `## [Unreleased]`
- [ ] Updated docs if graph surgery, flags or the report schema changed

## Numerical impact

<!--
If this touches conversion, report the effect on a real model:
  onnx-fp16-guard inspect model.onnx --json
  onnx-fp16-guard convert model.onnx -o /tmp/out.onnx --json
  python scripts/psnr.py model.onnx /tmp/out.onnx
-->
