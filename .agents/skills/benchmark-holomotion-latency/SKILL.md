---
name: benchmark-holomotion-latency
description: Benchmark HoloMotion inference with deployment-equivalent artifacts, runtime state, preprocessing, I/O, and hardware. Use for latency, throughput, cache, refinement, export, quantization, target-runtime, or framework-versus-deployment comparisons.
---

# Benchmark HoloMotion Latency

Treat development-framework timing as structural evidence, not a deployment
result. Resolve the current exporter, runtime caller, profile, and target
backend from source before building a benchmark.

## Match the deployed contract

1. Use the exact model, configuration, precision, batch, input shapes, dtypes,
   preprocessing, postprocessing, outputs, and recurrent state transitions.
2. Feed returned state into every subsequent step and reproduce reset rules.
3. Compare deployment outputs against the reference implementation before
   timing. Fail on unintended provider or device fallback.
4. Match production I/O, synchronization, and memory transfers. Measure engine
   construction and cold startup separately from steady-state execution.
5. Label random weights or synthetic inputs as structural smoke tests only.

## Measure

1. Prefer an otherwise idle target and record competing load.
2. Warm the complete steady-state path before collecting samples.
3. Use enough sequential samples to stabilize tail latency; report sample
   count, mean, P50, P95, and P99 when a deadline is relevant.
4. Time a clearly named boundary: graph execution, deployed model step, or
   complete control loop. Never compare different boundaries as one metric.
5. Repeat or interleave variants when device load can drift.

Keep artifact, inputs, backend, state, and measurement boundary fixed when
comparing refinements, precisions, or runtimes. Validate numerical and task
quality before accepting a speedup.

Target access or model execution on a robot computer requires explicit user
authorization. Keep controllers stopped, use isolated temporary artifacts, and
never overwrite deployed files or infer physical validation from a benchmark.

## Report

State the resolved artifact and hash, runtime/backend, precision, hardware,
measurement boundary, state handling, warmup, sample count, load, latency
percentiles, deadline relation, and numerical or task-quality delta.
