# SYSTEMDS-BENCH-GPT

Backend-agnostic benchmarking suite for LLM inference systems.

This repository implements the benchmarking framework to evaluate:

- local inference engines (e.g. vLLM; later SystemDS inference)
- hosted APIs (e.g. OpenAI GPT)

## Scope separation

- In scope (DIA): benchmarking harness, workloads, metrics, scripts, results.
- Out of scope (HiWi): implementing the SystemDS inference engine itself.

## Status

Initial skeleton (PR1 target: runnable harness + one local backend + one workload).
