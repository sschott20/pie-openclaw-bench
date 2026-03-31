#!/usr/bin/env python3
"""Cache strategy benchmark — compare caching approaches on the same prompt.

Runs a single modular prompt through multiple caching strategies (cold, hot, hot)
and reports timing breakdowns. Designed for quick local iteration with small models.

Usage:
    # PIE strategies (start pie serve first):
    python scripts/cache_strategy_bench.py --strategies pie_auto,pie_prefix,pie_modular

    # vLLM (start vLLM server first, stop PIE to free GPU):
    python scripts/cache_strategy_bench.py --strategies vllm_prefix

    # Real Claude Code prompts:
    python scripts/cache_strategy_bench.py --strategies pie_auto,pie_prefix,pie_modular --prompt-source real

    # Larger prompt (~24k tokens):
    python scripts/cache_strategy_bench.py --prompt-scale 3 --prompt-source real

    # Local testing with small model:
    python scripts/cache_strategy_bench.py --model Qwen/Qwen3-0.6B

    # RunPod (B200, Qwen3-32B — default):
    python scripts/cache_strategy_bench.py --prompt-source real --prompt-scale 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field

import aiohttp


# ── Prompt generation ────────────────────────────────────────────────────────

CORE_TEMPLATE = """\
You are an advanced AI coding assistant. Your primary purpose is to help \
software engineers write, debug, refactor, and understand code. You have \
access to tools that let you read files, write files, execute shell commands, \
search codebases, and browse the web.

## Core Principles
- Always read existing code before suggesting modifications
- Prefer editing existing files over creating new ones
- Keep solutions simple and focused on the task at hand
- Write safe, secure code that avoids common vulnerabilities
- Use appropriate error handling at system boundaries

## Response Format
- Be concise and direct in your responses
- Lead with the answer or action, not reasoning
- Use code blocks with language annotations
- Reference specific file paths and line numbers when discussing code
- Break down complex changes into clear steps
"""

TOOL_TEMPLATE = """\
## Available Tools

### bash
Execute shell commands in a sandboxed environment.
Parameters: command (string, required), timeout (integer), working_dir (string)

### read
Read file contents from the filesystem. Supports text, images, PDFs, notebooks.
Parameters: file_path (string, required), offset (integer), limit (integer)

### write
Write content to a file. Creates if not exists, overwrites if it does.
Parameters: file_path (string, required), content (string, required)

### edit
Perform exact string replacements in files. old_string must be unique.
Parameters: file_path (string, required), old_string (string, required), new_string (string, required)

### glob
Find files matching glob patterns. Returns paths sorted by modification time.
Parameters: pattern (string, required), path (string)

### grep
Search file contents using regular expressions with context lines.
Parameters: pattern (string, required), path (string), glob (string), context (integer)

### web_search
Search the web for current information. Returns results with snippets.
Parameters: query (string, required), num_results (integer)

### web_fetch
Fetch and read content from a URL. Returns text content of the page.
Parameters: url (string, required)
"""

SKILL_TEMPLATE = """\
# Code Review Skill
When reviewing code changes, follow this structured approach:
1. Check for correctness: Does the code do what it claims?
2. Check for security: Are there injection risks, auth bypasses, or data leaks?
3. Check for performance: Are there N+1 queries, unnecessary allocations?
4. Check for maintainability: Is the code readable and well-structured?
5. Check test coverage: Are edge cases tested? Are tests meaningful?
Provide specific, actionable feedback with line references.
"""

MEMORY_TEMPLATE = """\
## Agent Memory
- User prefers TypeScript over JavaScript for new projects.
- Project uses React 19 with server components.
- Database is PostgreSQL 16 with pgvector extension.
- CI/CD runs on GitHub Actions with required checks.
- User uses pytest with pytest-asyncio for Python tests.
- Preferred code style: functional where possible, classes for stateful components.
- Project deployment target is Kubernetes on AWS EKS.
- Last successful deployment was March 8, 2026.
- Auth system uses JWT with 15-minute access tokens.
- Rate limiting is configured at 100 req/min per user.
"""

HISTORY_TEMPLATE = """\
## Conversation History

**user**: Can you help me optimize this database query? It's taking 3 seconds.
**assistant**: I'll look at the query. Let me read the relevant code first.
**user**: It's in src/db/queries.py, the get_user_activity function.
**assistant**: I see the issue — there's a missing index on activity.user_id, and \
the query is doing a sequential scan. Let me add the index and rewrite the query.
**user**: That brought it down to 50ms. Can you also add caching?
**assistant**: I'll add a Redis cache with a 5-minute TTL for the aggregated results.
"""

USER_MESSAGE = "Now please review the changes I made to the auth middleware and suggest improvements."

CHARS_PER_TOKEN = 4


def pad_text(text: str, target_tokens: int) -> str:
    """Pad text to target token count by repeating content."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    if len(text) >= target_chars:
        return text[:target_chars]
    padding = "\nAdditional context continues below.\n" + text[:200]
    reps = (target_chars - len(text)) // len(padding) + 1
    return (text + padding * reps)[:target_chars]


@dataclass
class PromptConfig:
    """Configurable prompt module sizes (in tokens)."""
    core_tokens: int = 1500
    tool_tokens: int = 3000
    skill_tokens: int = 1500
    memory_tokens: int = 1000
    history_tokens: int = 1000
    max_response_tokens: int = 100

    @property
    def total_tokens(self) -> int:
        return (self.core_tokens + self.tool_tokens + self.skill_tokens +
                self.memory_tokens + self.history_tokens + 50)  # +50 for user msg


@dataclass
class Module:
    name: str
    content: str

    @property
    def est_tokens(self) -> int:
        return max(1, len(self.content) // CHARS_PER_TOKEN)


def build_modules(config: PromptConfig, source: str = "synthetic") -> list[Module]:
    """Build the prompt modules at configured sizes."""
    if source == "real":
        from harness.prompts import get_module
        p = get_module("real")
        pm_core = p.make_core_instructions(config.core_tokens)
        pm_tools = p.make_tool_schemas(config.tool_tokens)
        pm_skill = p.make_skill("code_review", config.skill_tokens)
        pm_memory = p.make_memory(config.memory_tokens)
        pm_history = p.make_conversation_history(4, config.history_tokens // 4)
        pm_user = p.make_user_message(
            "Now please review the changes I made to the auth middleware and suggest improvements."
        )
        return [
            Module(pm_core.name, pm_core.content),
            Module(pm_tools.name, pm_tools.content),
            Module(pm_skill.name, pm_skill.content),
            Module(pm_memory.name, pm_memory.content),
            Module(pm_history.name, pm_history.content),
            Module(pm_user.name, pm_user.content),
        ]
    return [
        Module("core_instructions", pad_text(CORE_TEMPLATE, config.core_tokens)),
        Module("tool_schemas", pad_text(TOOL_TEMPLATE, config.tool_tokens)),
        Module("skill_active", pad_text(SKILL_TEMPLATE, config.skill_tokens)),
        Module("memory", pad_text(MEMORY_TEMPLATE, config.memory_tokens)),
        Module("history", pad_text(HISTORY_TEMPLATE, config.history_tokens)),
        Module("user_message", USER_MESSAGE),
    ]


# ── Timing result ────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    strategy: str
    run_label: str  # "cold", "hot1", "hot2"
    cache_build_ms: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_generated: int = 0
    generated_text: str = ""


# ── vLLM strategies ──────────────────────────────────────────────────────────

async def run_vllm(
    modules: list[Module],
    config: PromptConfig,
    model: str,
    base_url: str,
    strategy_name: str,
    run_label: str,
) -> RunResult:
    """Run a single vLLM request and measure timing."""
    prompt = "\n".join(m.content for m in modules)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_response_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    result = RunResult(strategy=strategy_name, run_label=run_label)
    t_start = time.monotonic()
    first_token = False

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                text = line.decode().strip()
                if not text.startswith("data: "):
                    continue
                data = text[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if not first_token:
                        result.ttft_ms = (time.monotonic() - t_start) * 1000
                        first_token = True
                    result.generated_text += content
                    result.tokens_generated += 1

    result.total_ms = (time.monotonic() - t_start) * 1000
    return result


# ── PIE strategies ───────────────────────────────────────────────────────────

async def run_pie_auto(
    modules: list[Module],
    config: PromptConfig,
    pie_uri: str,
    run_label: str,
) -> RunResult:
    """PIE autoregressive — fresh context, full prefill, no caching."""
    from pie_client import PieClient
    from pathlib import Path

    wasm = str(Path(__file__).parent.parent / "inferlet-baseline" / "target" /
               "wasm32-wasip2" / "release" / "modular_kv_baseline.wasm")
    manifest = str(Path(__file__).parent.parent / "inferlet-baseline" / "Pie.toml")

    result = RunResult(strategy="pie_auto", run_label=run_label)

    async with PieClient(pie_uri) as client:
        await client.authenticate("benchmark")
        await client.install_program(wasm, manifest)
        inst = await client.launch_instance("modular-kv-baseline@0.3.0", arguments=[])

        # Drain startup
        try:
            while True:
                await asyncio.wait_for(inst.recv(), timeout=2)
        except asyncio.TimeoutError:
            pass

        # Send request
        msg = json.dumps({
            "program_id": "bench",
            "turn_index": 0,
            "modules": [{"name": m.name, "content": m.content, "hash": ""} for m in modules],
            "max_tokens": config.max_response_tokens,
        })

        t_start = time.monotonic()
        await inst.send(msg)

        first_token = False
        while True:
            event, data = await asyncio.wait_for(inst.recv(), timeout=120)
            ename = event.name if hasattr(event, "name") else str(event)
            text = data if isinstance(data, str) else data.decode() if data else ""
            if ename == "Message":
                if text.startswith("__DONE__"):
                    break
                elif text.startswith("__ERROR__"):
                    raise RuntimeError(f"PIE error: {text}")
                else:
                    if not first_token:
                        result.ttft_ms = (time.monotonic() - t_start) * 1000
                        first_token = True
                    result.generated_text += text
                    result.tokens_generated += 1
            elif ename in ("Exception", "ServerError", "Completed"):
                break

        result.total_ms = (time.monotonic() - t_start) * 1000

        # Terminate instance
        try:
            await inst.send("__SHUTDOWN__")
            await asyncio.wait_for(inst.recv(), timeout=5)
        except Exception:
            pass
        try:
            await inst.terminate()
        except Exception:
            pass

    return result


async def run_pie_prefix(
    modules: list[Module],
    config: PromptConfig,
    pie_uri: str,
    run_label: str,
    instance_state: dict,
) -> RunResult:
    """PIE prefix cache — persistent inferlet with sequential prefix caching.

    instance_state holds the PieClient and instance across runs so the cache persists.
    """
    import hashlib
    from pathlib import Path

    result = RunResult(strategy="pie_prefix", run_label=run_label)

    # Reuse or create persistent instance
    if "client" not in instance_state:
        from pie_client import PieClient

        wasm = str(Path(__file__).parent.parent / "inferlet" / "target" /
                    "wasm32-wasip2" / "release" / "modular_kv_cache.wasm")
        manifest = str(Path(__file__).parent.parent / "inferlet" / "Pie.toml")

        client = PieClient(pie_uri)
        await client.__aenter__()
        await client.authenticate("benchmark")
        await client.install_program(wasm, manifest)
        inst = await client.launch_instance("modular-kv-cache@0.9.0", arguments=[])

        # Drain startup
        try:
            while True:
                await asyncio.wait_for(inst.recv(), timeout=2)
        except asyncio.TimeoutError:
            pass

        # Warmup
        await inst.send(json.dumps({"mode": "warmup"}))
        while True:
            event, data = await asyncio.wait_for(inst.recv(), timeout=30)
            text = data if isinstance(data, str) else data.decode() if data else ""
            if "__WARMUP_DONE__" in text:
                break

        instance_state["client"] = client
        instance_state["instance"] = inst
        instance_state["cache_built"] = set()

    inst = instance_state["instance"]
    cache_built = instance_state["cache_built"]

    # Compute module hashes
    hashes = []
    for m in modules:
        h = hashlib.sha256(m.content.encode()).hexdigest()[:16]
        hashes.append(h)

    # Build cache layers (timed separately)
    t_cache_start = time.monotonic()
    for i in range(len(modules)):
        key = "kv_" + "_".join(hashes[:i + 1])
        if key in cache_built:
            continue
        import_key = "kv_" + "_".join(hashes[:i]) if i > 0 else None
        await inst.send(json.dumps({
            "mode": "cache_build",
            "import_key": import_key,
            "module_content": modules[i].content,
            "export_key": key,
            "is_first_module": (i == 0),
        }))
        while True:
            event, data = await asyncio.wait_for(inst.recv(), timeout=120)
            text = data if isinstance(data, str) else data.decode() if data else ""
            if "__CACHE_BUILT__" in text:
                cache_built.add(key)
                break
            elif "Error" in (event.name if hasattr(event, "name") else ""):
                raise RuntimeError(f"Cache build error: {text}")
    result.cache_build_ms = (time.monotonic() - t_cache_start) * 1000

    # Find deepest cached prefix
    cache_hits = 0
    import_key = None
    for depth in range(len(hashes), 0, -1):
        key = "kv_" + "_".join(hashes[:depth])
        if key in cache_built:
            cache_hits = depth
            import_key = key
            break

    remaining = modules[cache_hits:]

    # Generate (timed as TTFT/total)
    msg = json.dumps({
        "mode": "generate",
        "import_key": import_key,
        "cache_hits": cache_hits,
        "modules": [{"content": m.content} for m in remaining],
        "first_module_is_system": (cache_hits == 0),
        "user_prompt": "Continue.",
        "max_tokens": config.max_response_tokens,
        "program_id": "bench",
        "turn_index": 0,
        "total_modules": len(modules),
    })

    t_start = time.monotonic()
    await inst.send(msg)

    first_token = False
    while True:
        event, data = await asyncio.wait_for(inst.recv(), timeout=120)
        ename = event.name if hasattr(event, "name") else str(event)
        text = data if isinstance(data, str) else data.decode() if data else ""
        if ename == "Message":
            if text.startswith("__DONE__"):
                break
            elif text.startswith("__ERROR__"):
                raise RuntimeError(f"PIE error: {text}")
            else:
                if not first_token:
                    result.ttft_ms = (time.monotonic() - t_start) * 1000
                    first_token = True
                result.generated_text += text
                result.tokens_generated += 1
        elif ename in ("Exception", "ServerError", "Completed"):
            break

    result.total_ms = (time.monotonic() - t_start) * 1000
    return result


async def run_pie_modular(
    modules: list[Module],
    config: PromptConfig,
    pie_uri: str,
    run_label: str,
    instance_state: dict,
) -> RunResult:
    """PIE modular cache — independent per-module KV caching (PromptCache-style)."""
    import hashlib
    from pathlib import Path

    result = RunResult(strategy="pie_modular", run_label=run_label)

    # Reuse or create persistent instance (same inferlet as prefix)
    if "client" not in instance_state:
        from pie_client import PieClient
        wasm = str(Path(__file__).parent.parent / "inferlet" / "target" /
                    "wasm32-wasip2" / "release" / "modular_kv_cache.wasm")
        manifest = str(Path(__file__).parent.parent / "inferlet" / "Pie.toml")
        client = PieClient(pie_uri)
        await client.__aenter__()
        await client.authenticate("benchmark")
        await client.install_program(wasm, manifest)
        inst = await client.launch_instance("modular-kv-cache@0.9.0", arguments=[])
        try:
            while True:
                await asyncio.wait_for(inst.recv(), timeout=2)
        except asyncio.TimeoutError:
            pass
        await inst.send(json.dumps({"mode": "warmup"}))
        while True:
            event, data = await asyncio.wait_for(inst.recv(), timeout=30)
            text = data if isinstance(data, str) else data.decode() if data else ""
            if "__WARMUP_DONE__" in text:
                break
        instance_state["client"] = client
        instance_state["instance"] = inst
        instance_state["cache_built"] = set()

    inst = instance_state["instance"]
    cache_built = instance_state["cache_built"]

    # Cache each module independently (keyed by content hash alone)
    hashes = []
    for m in modules:
        h = hashlib.sha256(m.content.encode()).hexdigest()[:16]
        hashes.append(h)

    t_cache_start = time.monotonic()
    for i, m in enumerate(modules):
        key = f"mod_{hashes[i]}"
        if key in cache_built:
            continue
        await inst.send(json.dumps({
            "mode": "modular_cache_build",
            "module_content": m.content,
            "export_key": key,
            "is_first_module": (i == 0),
        }))
        while True:
            event, data = await asyncio.wait_for(inst.recv(), timeout=120)
            text = data if isinstance(data, str) else data.decode() if data else ""
            if "__MODULAR_CACHE_BUILT__" in text:
                cache_built.add(key)
                break
            elif text.startswith("__ERROR__"):
                raise RuntimeError(f"Modular cache build error: {text}")
    result.cache_build_ms = (time.monotonic() - t_cache_start) * 1000

    # Generate by concatenating all cached modules
    module_keys = [f"mod_{h}" for h in hashes]

    msg = json.dumps({
        "mode": "modular_generate",
        "cache_keys": module_keys,
        "user_prompt": "Continue.",
        "max_tokens": config.max_response_tokens,
        "program_id": "bench",
        "turn_index": 0,
    })

    t_start = time.monotonic()
    await inst.send(msg)

    first_token = False
    while True:
        event, data = await asyncio.wait_for(inst.recv(), timeout=120)
        ename = event.name if hasattr(event, "name") else str(event)
        text = data if isinstance(data, str) else data.decode() if data else ""
        if ename == "Message":
            if text.startswith("__DONE__"):
                break
            elif text.startswith("__ERROR__"):
                raise RuntimeError(f"PIE error: {text}")
            else:
                if not first_token:
                    result.ttft_ms = (time.monotonic() - t_start) * 1000
                    first_token = True
                result.generated_text += text
                result.tokens_generated += 1
        elif ename in ("Exception", "ServerError", "Completed"):
            break

    result.total_ms = (time.monotonic() - t_start) * 1000
    return result


# ── Runner ───────────────────────────────────────────────────────────────────

ALL_STRATEGIES = ["vllm_nocache", "vllm_prefix", "pie_auto", "pie_prefix", "pie_modular"]


async def run_benchmark(args: argparse.Namespace) -> list[RunResult]:
    """Run the benchmark across all requested strategies."""
    config = PromptConfig(
        core_tokens=args.core_tokens,
        tool_tokens=args.tool_tokens,
        skill_tokens=args.skill_tokens,
        memory_tokens=args.memory_tokens,
        history_tokens=args.history_tokens,
        max_response_tokens=args.max_tokens,
    )
    modules = build_modules(config, source=args.prompt_source)

    total_est = sum(m.est_tokens for m in modules)
    print(f"\nPrompt: {len(modules)} modules, ~{total_est} tokens")
    for m in modules:
        print(f"  {m.name}: ~{m.est_tokens} tokens")
    print(f"  max_response_tokens: {config.max_response_tokens}")

    strategies = args.strategies.split(",")
    if "all" in strategies:
        strategies = ALL_STRATEGIES

    run_labels = ["cold", "hot1", "hot2"]
    results: list[RunResult] = []

    # vLLM strategies
    for strat in strategies:
        if not strat.startswith("vllm"):
            continue
        print(f"\n{'='*60}")
        print(f"Strategy: {strat}")
        print(f"{'='*60}")

        for label in run_labels:
            print(f"  Run: {label}...", end=" ", flush=True)
            try:
                r = await run_vllm(
                    modules, config, args.model, args.vllm_url, strat, label,
                )
                results.append(r)
                print(f"TTFT={r.ttft_ms:.1f}ms total={r.total_ms:.1f}ms "
                      f"tokens={r.tokens_generated}")
            except Exception as e:
                print(f"FAILED: {e}")

    # PIE strategies
    pie_prefix_state: dict = {}
    pie_modular_state: dict = {}
    for strat in strategies:
        if not strat.startswith("pie"):
            continue
        print(f"\n{'='*60}")
        print(f"Strategy: {strat}")
        print(f"{'='*60}")

        for label in run_labels:
            print(f"  Run: {label}...", end=" ", flush=True)
            try:
                if strat == "pie_auto":
                    r = await run_pie_auto(
                        modules, config, args.pie_uri, label,
                    )
                elif strat == "pie_prefix":
                    r = await run_pie_prefix(
                        modules, config, args.pie_uri, label, pie_prefix_state,
                    )
                elif strat == "pie_modular":
                    r = await run_pie_modular(
                        modules, config, args.pie_uri, label, pie_modular_state,
                    )
                else:
                    print(f"SKIPPED (not implemented)")
                    continue
                results.append(r)
                cache_str = f"cache_build={r.cache_build_ms:.1f}ms " if r.cache_build_ms else ""
                print(f"{cache_str}TTFT={r.ttft_ms:.1f}ms total={r.total_ms:.1f}ms "
                      f"tokens={r.tokens_generated}")
            except Exception as e:
                print(f"FAILED: {e}")

    # Cleanup PIE state
    for state in [pie_prefix_state, pie_modular_state]:
        if "client" in state:
            try:
                inst = state["instance"]
                await inst.send(json.dumps({"mode": "shutdown"}))
                await asyncio.wait_for(inst.recv(), timeout=5)
            except Exception:
                pass
            try:
                await state["client"].__aexit__(None, None, None)
            except Exception:
                pass

    return results


def print_results_table(results: list[RunResult]) -> None:
    """Print a formatted results table."""
    if not results:
        return

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"{'Strategy':<16} {'Run':<6} {'Cache(ms)':>10} {'TTFT(ms)':>10} "
          f"{'Total(ms)':>10} {'Tokens':>7}")
    print(f"{'-'*16} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")

    for r in results:
        cache = f"{r.cache_build_ms:.1f}" if r.cache_build_ms else "-"
        print(f"{r.strategy:<16} {r.run_label:<6} {cache:>10} {r.ttft_ms:>10.1f} "
              f"{r.total_ms:>10.1f} {r.tokens_generated:>7}")

    # Print text comparison (first 100 chars)
    print(f"\n{'='*80}")
    print("OUTPUT COMPARISON (first 100 chars)")
    print(f"{'='*80}")
    seen = set()
    for r in results:
        key = (r.strategy, r.run_label)
        if key not in seen:
            seen.add(key)
            text_preview = r.generated_text[:100].replace("\n", "\\n")
            print(f"  {r.strategy}/{r.run_label}: {text_preview}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache strategy benchmark")
    p.add_argument("--strategies", default="all",
                   help="Comma-separated: vllm_nocache,vllm_prefix,pie_auto,pie_prefix,pie_modular,all")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--pie-uri", default="ws://localhost:8080")
    p.add_argument("--max-tokens", type=int, default=100)

    # Prompt size controls
    p.add_argument("--prompt-scale", type=float, default=1.0,
                   help="Multiply all module sizes by this factor (e.g., 3 for ~24k tokens)")
    p.add_argument("--core-tokens", type=int, default=None)
    p.add_argument("--tool-tokens", type=int, default=None)
    p.add_argument("--skill-tokens", type=int, default=None)
    p.add_argument("--memory-tokens", type=int, default=None)
    p.add_argument("--history-tokens", type=int, default=None)
    p.add_argument("--prompt-source", choices=["synthetic", "real"], default="synthetic",
                   help="Use 'real' for actual Claude Code prompt content")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply scale factor to defaults, then override with explicit values
    scale = args.prompt_scale
    defaults = PromptConfig()
    if args.core_tokens is None:
        args.core_tokens = int(defaults.core_tokens * scale)
    if args.tool_tokens is None:
        args.tool_tokens = int(defaults.tool_tokens * scale)
    if args.skill_tokens is None:
        args.skill_tokens = int(defaults.skill_tokens * scale)
    if args.memory_tokens is None:
        args.memory_tokens = int(defaults.memory_tokens * scale)
    if args.history_tokens is None:
        args.history_tokens = int(defaults.history_tokens * scale)

    results = asyncio.run(run_benchmark(args))
    print_results_table(results)

    # Save results JSON
    from pathlib import Path
    out = Path("results") / "cache_strategy_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([
        {
            "strategy": r.strategy,
            "run": r.run_label,
            "cache_build_ms": r.cache_build_ms,
            "ttft_ms": r.ttft_ms,
            "total_ms": r.total_ms,
            "tokens_generated": r.tokens_generated,
            "generated_text": r.generated_text,
        }
        for r in results
    ], indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
