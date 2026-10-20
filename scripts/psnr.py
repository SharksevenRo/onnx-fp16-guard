#!/usr/bin/env python3
"""Compare two ONNX models numerically (PSNR / max error) with onnxruntime.

    python scripts/psnr.py reference.onnx candidate.onnx [--seed 0] [--min-psnr 30]

Feeds deterministic random data to every model input (fixed shapes only) and
prints per-output metrics. Useful when you already have a Python ML environment
and do not want to install onnxruntime-node.

Requires: numpy, onnxruntime.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import onnxruntime as ort


def build_feeds(session: ort.InferenceSession, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in item.shape]
        if item.type == "tensor(float)":
            feeds[item.name] = rng.random(shape, dtype=np.float32) * 2 - 1
        elif item.type == "tensor(float16)":
            feeds[item.name] = (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float16)
        elif item.type == "tensor(int64)":
            feeds[item.name] = np.zeros(shape, dtype=np.int64)
        elif item.type == "tensor(int32)":
            feeds[item.name] = np.zeros(shape, dtype=np.int32)
        elif item.type == "tensor(bool)":
            feeds[item.name] = np.zeros(shape, dtype=bool)
        else:
            raise SystemExit(f"unsupported input type {item.type} for {item.name}")
    return feeds


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    a = reference.astype(np.float64)
    b = candidate.astype(np.float64)
    diff = b - a
    mse = float(np.mean(diff**2))
    value_range = float(np.max(a) - np.min(a)) or 1.0
    psnr = 10 * np.log10(value_range**2 / mse) if mse > 0 else float("inf")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a.ravel(), b.ravel()) / denominator) if denominator else 1.0
    return {
        "psnr": psnr,
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
        "cosine": cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-psnr", type=float, default=30.0)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    reference = ort.InferenceSession(args.reference, sess_options=options, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(args.candidate, sess_options=options, providers=["CPUExecutionProvider"])

    feeds = build_feeds(reference, args.seed)
    ref_outputs = reference.run(None, feeds)
    cand_outputs = candidate.run(None, feeds)

    worst = float("inf")
    for index, (a, b) in enumerate(zip(ref_outputs, cand_outputs)):
        result = metrics(a, b)
        worst = min(worst, result["psnr"])
        print(
            f"output[{index}] PSNR {result['psnr']:.2f} dB · "
            f"max|Δ| {result['max_abs']:.3e} · RMSE {result['rmse']:.3e} · cosine {result['cosine']:.7f}"
        )

    passed = worst >= args.min_psnr
    print(f"\n{'PASS' if passed else 'FAIL'}: worst PSNR {worst:.2f} dB (threshold {args.min_psnr} dB)")
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
