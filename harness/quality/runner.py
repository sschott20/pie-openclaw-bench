"""Quality comparison runner: runs identical workloads against two backends and evaluates output quality."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from harness.models import (
    BackendType,
    ExperimentConfig,
    ModularRequest,
    StreamingResponse,
)
from harness.prompts.synthetic import PromptSizes
from harness.quality.evaluator import QualityEvaluator
from harness.quality.models import QualityReport
from harness.runner import create_backend, create_workload


async def _collect_responses(
    config: ExperimentConfig,
    requests: list[ModularRequest],
) -> list[str]:
    """Run a list of requests against a backend and collect response texts."""
    backend = create_backend(config.backend)
    await backend.setup(config)

    responses: list[str] = []
    try:
        for request in requests:
            resp: StreamingResponse = await backend.send_request(request)
            responses.append(resp.text)
    finally:
        await backend.teardown()

    return responses


async def run_quality_comparison(
    reference_config: ExperimentConfig,
    candidate_config: ExperimentConfig,
) -> QualityReport:
    """Run identical requests against two backends and compare output quality.

    Args:
        reference_config: Config for the full-prefill reference backend (e.g. vLLM).
        candidate_config: Config for the modular-cache candidate backend (e.g. PIE cache).

    Returns:
        QualityReport with per-pair and aggregate metrics.
    """
    # Both configs must use the same workload
    if reference_config.workload != candidate_config.workload:
        raise ValueError(
            f"Workload mismatch: {reference_config.workload} vs {candidate_config.workload}"
        )

    # Build workload from reference config (same for both)
    sizes = PromptSizes(**{
        k: v
        for k, v in reference_config.workload_params.items()
        if k in PromptSizes.__dataclass_fields__
    })
    workload_params = {
        k: v
        for k, v in reference_config.workload_params.items()
        if k not in PromptSizes.__dataclass_fields__
    }

    workload = create_workload(reference_config.workload, workload_params, sizes)
    requests = workload.generate_program("quality_eval_prog_0")

    # Collect responses from both backends
    reference_responses = await _collect_responses(reference_config, requests)
    candidate_responses = await _collect_responses(candidate_config, requests)

    # Evaluate quality
    evaluator = QualityEvaluator()
    report = evaluator.compare_outputs(reference_responses, candidate_responses)

    # Fill in metadata
    report.workload = reference_config.workload
    report.reference_backend = reference_config.backend.value
    report.candidate_backend = candidate_config.backend.value

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality comparison: reference backend vs candidate backend"
    )
    parser.add_argument(
        "--workload",
        choices=["react", "multiturn", "skill_switch", "heartbeat"],
        default="skill_switch",
        help="Workload scenario (default: skill_switch)",
    )
    parser.add_argument(
        "--reference",
        choices=["vllm", "pie_std", "pie_cache"],
        default="vllm",
        help="Reference backend (default: vllm)",
    )
    parser.add_argument(
        "--candidate",
        choices=["vllm", "pie_std", "pie_cache"],
        default="pie_cache",
        help="Candidate backend (default: pie_cache)",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
    )
    parser.add_argument(
        "--pie-uri",
        default="ws://localhost:8080",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON report (default: stdout)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ref_config = ExperimentConfig(
        name=f"quality_ref_{args.reference}",
        backend=BackendType(args.reference),
        workload=args.workload,
        model=args.model,
        num_programs=1,
        num_repetitions=1,
        warmup_programs=0,
        temperature=0.0,
        vllm_url=args.vllm_url,
        pie_server_uri=args.pie_uri,
    )
    cand_config = ExperimentConfig(
        name=f"quality_cand_{args.candidate}",
        backend=BackendType(args.candidate),
        workload=args.workload,
        model=args.model,
        num_programs=1,
        num_repetitions=1,
        warmup_programs=0,
        temperature=0.0,
        vllm_url=args.vllm_url,
        pie_server_uri=args.pie_uri,
    )

    report = asyncio.run(run_quality_comparison(ref_config, cand_config))

    if args.output:
        report.save_json(args.output)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report.to_dict(), indent=2))

    # Print summary
    print(f"\n--- Quality Summary ({args.workload}) ---")
    print(f"Reference: {args.reference}  Candidate: {args.candidate}")
    print(f"Pairs evaluated: {report.num_pairs}")
    print(f"Exact match rate: {report.exact_match_rate:.1%}")
    print(f"Mean BLEU:        {report.mean_bleu:.4f}")
    print(f"Mean ROUGE-L F1:  {report.mean_rouge_l:.4f}")
    print(f"Mean edit dist:   {report.mean_edit_distance:.4f}")


if __name__ == "__main__":
    main()
