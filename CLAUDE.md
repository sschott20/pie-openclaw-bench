# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research benchmark measuring whether PIE (Programmable Inference Engine) with application-level modular KV caching outperforms vLLM prefix caching for agentic LLM workloads. Uses OpenClaw (Claude Code-like agent) as the representative workload. The full specification is in `DESIGN.md`.

**Current state**: Implementation complete — Python harness with all 4 workload generators, 3 backend drivers, metrics pipeline, and plotting. Rust inferlet with modular KV caching written. Awaiting RunPod deployment for actual experiments. Early PIE vs vLLM latency results are in `notes-pie-vs-vllm.md` (PIE 1.43x faster in initial test).

## Architecture (Three-Tier)

1. **Benchmark Harness** (`harness/`, Python) — Workload generators, backend drivers, metrics, trace replay, experiment runner
2. **Backend Services** — vLLM server (prefix caching baseline), PIE std/chat (overhead baseline), PIE openclaw-cache (our contribution)
3. **PIE Inferlet** (`inferlet/`, Rust) — Custom Rust inferlet implementing per-module KV page export/import with content-hash invalidation

The harness sends identical `ModularRequest`s (ordered list of `PromptModule`s with name, content, SHA-256 hash) to all three backends and collects TTFT, throughput, and cache metrics.

## Build & Run

```bash
# Python setup
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run tests (use -I to avoid ROS2/conda path pollution)
./venv/bin/python -I -m pytest tests/ -v

# Run a single test
./venv/bin/python -I -m pytest tests/test_workloads.py::TestSkillSwitchWorkload -v

# Run experiment (requires vLLM or PIE server running)
python harness/runner.py --workload skill_switch --backend vllm
python harness/runner.py --config experiment_config.json

# Rust inferlet build
cd inferlet && cargo build --target wasm32-wasip2 --release

# PIE commands
pie-cli submit <wasm_path> -- <args>

# vLLM baseline
# launched with: --enable-prefix-caching --max-model-len 131072 --gpu-memory-utilization 0.9
```

## Key Design Decisions

- **Per-module KV caching**: Each prompt module (core_instructions, tool_schemas, skills, memory, history) gets its own named KV page handle — independently cached, invalidated, and reused
- **Content-hash invalidation**: SHA-256 per module; unchanged modules import via memcpy, changed modules re-prefill and re-export
- **Cross-module attention limitation**: Modules prefilled independently (KV states attend within-module only). Acceptable for semantically independent modules; quality measured via BLEU/ROUGE against full-prefill baseline
- **Trace-based methodology**: Poisson-distributed program arrivals replaying real OpenClaw traces (following Continuum, Li et al.)

## Four Workload Scenarios

1. **ReAct tool loop** — stable prefix across tool steps; tests cache retention during tool pauses
2. **Multi-turn conversation** — growing history dominates; constant savings on system prompt
3. **Skill switch** — mid-module change breaks prefix caching; modular caching's strongest scenario
4. **Heartbeat** — periodic requests with long intervals; tests cache eviction immunity

## Hardware

NVIDIA B200 (192GB HBM3e) on RunPod. Model: Llama 3.x family (configurable 8B/70B).

## Implementation Build Order (from DESIGN.md §6.2)

1. Collect real OpenClaw prompt content → `harness/prompts/`
2. Build Rust inferlet → `inferlet/src/lib.rs`
3. Backend drivers → `harness/backends/{vllm,pie_std,pie_cache}.py`
4. Workload generators → `harness/workloads/{react,multiturn,skill_switch,heartbeat}.py`
5. Metrics collection → `harness/metrics/{collector,aggregator,plots}.py`
6. Runner + RunPod scripts → `harness/runner.py`, `scripts/`
7. Analysis and figures → `results/`

## RunPod Deployment

Setup scripts are in `pod_setup/pod_setup/`. Run `./pod_setup.sh` from your local machine — it takes the RunPod SSH command, copies scripts to the pod, and runs them.

### Prerequisites
- RunPod account (group account, invite link from Zhiyao)
- SSH public key added to RunPod pod config at creation time
- Pod uses B200 GPU with enough disk for model weights (200GB+)

### What the setup script does
1. `pod_root_setup.sh` — installs packages, creates user `alexs`
2. `pod_user_setup.sh` — installs Rust, uv, oh-my-zsh, clones PIE + benchmark repos, builds everything

### After setup, on the pod
```bash
ssh pod

# Start PIE server (terminal 1)
cd ~/work/pie/pie
uv run pie serve

# Run benchmark (terminal 2)
cd ~/work/pie_openclaw
source .venv/bin/activate
python harness/runner.py --backend vllm --workload skill_switch --model Qwen/Qwen3-8B --num-programs 2 --num-repetitions 1

# Start vLLM baseline server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --port 8000

# Smoke test PIE inferlet
cd ~/work/pie/pie
uv run pie run ~/work/pie_openclaw/inferlet/target/wasm32-wasip2/release/modular_kv_cache.wasm -- '{"mode": "warmup"}'
```

### Important RunPod notes
- Only `/workspace` persists across pod stop/start — everything else is lost
- Terminating (deleting) a pod loses everything including `/workspace`
- Git push frequently — pods can be accidentally deleted by others
- PIE model must be downloaded after first `pie config init` (interactive prompt)
- The setup script uses `~/.ssh/id_rsa` for local SSH auth (hardcoded by Zhiyao)

### Known issues
- **PIE cross-instance KV export/import**: `export_kv_pages`/`import_kv_pages` may not persist across instance lifetimes. The one-shot cache architecture depends on this. Needs confirmation from PIE team (Zhiyao). The `store_get`/`store_set` key-value store DOES persist across instances.
- **PIE flush() bug**: `flush()` corrupts host state after first use in a looping inferlet. See `context/notes-one-shot-architecture.md`.

## Notes Files

- `context/notes-pie-setup.md` — PIE installation commands, PyTorch CUDA setup
- `context/notes-pie-thor.md` — Docker-based PIE (Thor) setup, inferlet build commands
- `context/notes-pie-vs-vllm.md` — Initial benchmark results (PIE 1.43x faster than vLLM)
- `context/notes-one-shot-architecture.md` — Flush bug analysis and one-shot design
- `context/DESIGN.md` — Full benchmark specification
