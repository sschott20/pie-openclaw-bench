# PIE Modular KV Cache Benchmark

Benchmark infrastructure for measuring the performance impact of modular KV caching on LLM serving workloads using [PIE](https://pie-project.org/) (Programmable Inference Engine).

## Overview

Standard LLM serving systems (vLLM, API providers) cache contiguous token prefixes. When any early part of a prompt changes, everything after it must be recomputed. This benchmark measures whether **modular KV caching** — caching and reusing KV attention states at per-module granularity — reduces time-to-first-token (TTFT) for structured, repetitive prompts.

## Architecture

Three backends receive identical workloads:

- **vLLM + prefix caching**: Industry-standard baseline
- **PIE std/chat**: PIE overhead baseline (no application-level caching)
- **PIE modular-cache**: Custom Rust inferlet with per-module KV page export/import

## Workload Scenarios

1. **Tool loop** — Stable system prompt across sequential tool calls
2. **Multi-turn conversation** — Growing conversation history with static system prefix
3. **Module switch** — Mid-session module changes that break prefix caching
4. **Heartbeat** — Periodic requests with long inter-request intervals

## Setup

```bash
# Python harness
python -m venv venv && source venv/bin/activate
pip install -r harness/requirements.txt

# Rust inferlet
cd inferlet && cargo build --target wasm32-wasip2 --release
```

## Running

```bash
python harness/runner.py --config <experiment_config>
```

## Hardware

Benchmarked on NVIDIA B200 (192GB HBM3e). Model: Llama 3.x family.
