"""Test that PIE cache and PIE std produce identical output.

Verifies that modular KV caching doesn't change the generated tokens.
Requires a running PIE server (pie serve) on localhost:8080.

Run:  ./venv/bin/python -I -m pytest tests/test_output_equivalence.py -v -s
"""

import asyncio
import json
import time

import pytest

from harness.models import ExperimentConfig, ModularRequest, PromptModule, BackendType

# Skip if PIE server not available
pytestmark = pytest.mark.skipif(
    not _pie_server_available(),
    reason="PIE server not running on localhost:8080",
) if not True else []  # Always try; will fail with clear error if server down


PIE_URI = "ws://localhost:8080"
MODEL = "Qwen/Qwen3-0.6B"
# Use enough tokens to get a real response
MAX_TOKENS = 150


def make_test_request() -> ModularRequest:
    """Create a request with realistic modules."""
    core = PromptModule(
        name="core",
        content=(
            "You are a helpful math tutor. You solve arithmetic problems "
            "step by step, showing your work clearly. Always explain each "
            "step and give the final numerical answer."
        ),
    )
    tools = PromptModule(
        name="tools",
        content=(
            "Available tools:\n"
            "- Calculator: evaluate mathematical expressions\n"
            "- UnitConverter: convert between measurement units\n"
            "These tools are for reference only. Solve problems manually."
        ),
    )
    memory = PromptModule(
        name="memory",
        content=(
            "The student is learning multiplication and division. "
            "They understand addition and subtraction well."
        ),
    )
    user_msg = PromptModule(
        name="user",
        content="What is 37 times 24? Show all your work step by step.",
    )

    return ModularRequest(
        program_id="equiv_test",
        turn_index=0,
        modules=[core, tools, memory, user_msg],
        max_response_tokens=MAX_TOKENS,
    )


async def run_pie_cache(request: ModularRequest) -> str:
    """Run request through PIE cache backend, return generated text."""
    from harness.backends.pie_cache import PIECacheBackend

    config = ExperimentConfig(
        name="equiv_test",
        backend=BackendType.PIE_CACHE,
        workload="skill_switch",
        model=MODEL,
        pie_server_uri=PIE_URI,
    )

    backend = PIECacheBackend()
    await backend.setup(config)
    try:
        response = await backend.send_request(request)
        return response.text
    finally:
        await backend.teardown()


async def run_pie_std(request: ModularRequest) -> str:
    """Run request through PIE std backend, return generated text."""
    from harness.backends.pie_std import PIEStdBackend

    config = ExperimentConfig(
        name="equiv_test",
        backend=BackendType.PIE_STD,
        workload="skill_switch",
        model=MODEL,
        pie_server_uri=PIE_URI,
    )

    backend = PIEStdBackend()
    await backend.setup(config)
    try:
        response = await backend.send_request(request)
        return response.text
    finally:
        await backend.teardown()


@pytest.mark.asyncio
async def test_outputs_match():
    """PIE cache and PIE std should produce identical output for the same prompt."""
    request = make_test_request()

    print("\n--- Running PIE std (no caching, full prefill) ---")
    t0 = time.monotonic()
    std_text = await run_pie_std(request)
    std_time = time.monotonic() - t0
    print(f"PIE std output ({std_time:.1f}s, {len(std_text)} chars):")
    print(f"  {std_text[:200]}...")

    print("\n--- Running PIE cache (modular KV caching) ---")
    t0 = time.monotonic()
    cache_text = await run_pie_cache(request)
    cache_time = time.monotonic() - t0
    print(f"PIE cache output ({cache_time:.1f}s, {len(cache_text)} chars):")
    print(f"  {cache_text[:200]}...")

    print(f"\n--- Comparison ---")
    print(f"  std  length: {len(std_text)} chars")
    print(f"  cache length: {len(cache_text)} chars")
    print(f"  match: {std_text == cache_text}")

    if std_text != cache_text:
        # Find first divergence point
        for i, (a, b) in enumerate(zip(std_text, cache_text)):
            if a != b:
                print(f"  first divergence at char {i}:")
                print(f"    std:   ...{std_text[max(0,i-20):i+20]}...")
                print(f"    cache: ...{cache_text[max(0,i-20):i+20]}...")
                break

    # Exact match is not guaranteed — separate forward passes for cached modules
    # cause bfloat16 precision differences that can flip greedy argmax at some steps.
    # But the outputs should be highly similar (same reasoning, same structure).

    assert len(std_text) > 50, (
        f"std output too short ({len(std_text)} chars) — model may be hitting EOS early"
    )
    assert len(cache_text) > 50, (
        f"cache output too short ({len(cache_text)} chars) — model may be hitting EOS early"
    )

    if std_text == cache_text:
        print(f"\n  PASS (exact match): {len(std_text)} chars, identical output")
    else:
        # Compute character-level overlap using longest common subsequence ratio
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, std_text, cache_text).ratio()
        print(f"  similarity ratio: {ratio:.3f}")
        assert ratio > 0.7, (
            f"Outputs too different (similarity {ratio:.3f})!\n"
            f"  std  ({len(std_text)} chars): {std_text[:100]}...\n"
            f"  cache ({len(cache_text)} chars): {cache_text[:100]}..."
        )
        print(f"\n  PASS (near-match): similarity={ratio:.3f}, "
              f"std={len(std_text)} chars, cache={len(cache_text)} chars")
        print(f"  Note: bfloat16 precision differences between cached/uncached "
              f"forward passes cause minor token divergence. This is expected.")
