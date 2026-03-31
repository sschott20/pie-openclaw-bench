"""Experiment runner — orchestrates (workload, backend, config) combinations."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from harness.backends.base import Backend
from harness.backends.vllm import VLLMBackend
from harness.backends.pie_std import PIEStdBackend
from harness.backends.pie_cache import PIECacheBackend
from harness.metrics.aggregator import (
    aggregate_program_metrics,
    compute_experiment_summary,
    save_metrics_csv,
)
from harness.metrics.collector import collect_request_metrics
from harness.models import (
    BackendType,
    ExperimentConfig,
    ModularRequest,
    RequestMetrics,
)
from harness.workloads.base import WorkloadGenerator
from harness.workloads.react import ReactWorkload
from harness.workloads.multiturn import MultiTurnWorkload
from harness.workloads.skill_switch import SkillSwitchWorkload
from harness.workloads.heartbeat import HeartbeatWorkload
from harness.prompts import PromptSizes


def create_backend(backend_type: BackendType) -> Backend:
    """Factory for backend instances."""
    match backend_type:
        case BackendType.VLLM:
            return VLLMBackend()
        case BackendType.PIE_STD:
            return PIEStdBackend()
        case BackendType.PIE_CACHE:
            return PIECacheBackend()


def create_workload(
    name: str,
    params: dict,
    sizes: PromptSizes,
) -> WorkloadGenerator:
    """Factory for workload generators."""
    match name:
        case "react":
            return ReactWorkload(
                num_turns=params.get("num_turns", 10),
                sizes=sizes,
            )
        case "multiturn":
            return MultiTurnWorkload(
                num_turns=params.get("num_turns", 20),
                sizes=sizes,
            )
        case "skill_switch":
            return SkillSwitchWorkload(
                num_turns=params.get("num_turns", 15),
                sizes=sizes,
            )
        case "heartbeat":
            return HeartbeatWorkload(
                num_beats=params.get("num_beats", 10),
                memory_update_every=params.get("memory_update_every", 10),
                sizes=sizes,
            )
        case _:
            raise ValueError(f"Unknown workload: {name}")


async def run_program(
    backend: Backend,
    requests: list[ModularRequest],
    backend_type: BackendType,
    inter_turn_delay: float = 0.0,
) -> list[RequestMetrics]:
    """Run a single program (all turns) against a backend."""
    metrics = []
    for request in requests:
        response = await backend.send_request(request)
        rm = collect_request_metrics(request, response, backend_type)
        metrics.append(rm)

        if inter_turn_delay > 0:
            await asyncio.sleep(inter_turn_delay)

    return metrics


async def run_experiment(config: ExperimentConfig) -> list[RequestMetrics]:
    """Run a single experiment: one workload × one backend × one config.

    Programs arrive according to Poisson distribution at config.arrival_rate.
    """
    sizes = PromptSizes(**{
        k: v for k, v in config.workload_params.items()
        if k in PromptSizes.__dataclass_fields__
    })
    workload_params = {
        k: v for k, v in config.workload_params.items()
        if k not in PromptSizes.__dataclass_fields__
    }

    workload = create_workload(config.workload, workload_params, sizes)
    backend = create_backend(config.backend)

    await backend.setup(config)

    all_metrics: list[RequestMetrics] = []

    try:
        for rep in range(config.num_repetitions):
            await backend.reset_state()
            rep_metrics: list[RequestMetrics] = []

            # Generate programs
            programs = []
            for i in range(config.num_programs + config.warmup_programs):
                pid = f"{config.name}_rep{rep}_prog{i}"
                programs.append(workload.generate_program(pid))

            # Schedule with Poisson arrivals
            for prog_idx, program_requests in enumerate(programs):
                is_warmup = prog_idx < config.warmup_programs

                # Run the program
                metrics = await run_program(
                    backend,
                    program_requests,
                    config.backend,
                )

                if not is_warmup:
                    rep_metrics.extend(metrics)

                # Poisson inter-arrival delay
                if config.arrival_rate > 0 and prog_idx < len(programs) - 1:
                    delay = random.expovariate(config.arrival_rate)
                    await asyncio.sleep(delay)

            all_metrics.extend(rep_metrics)
            print(f"  Repetition {rep + 1}/{config.num_repetitions}: "
                  f"{len(rep_metrics)} requests collected")

    finally:
        await backend.teardown()

    return all_metrics


async def run_experiment_suite(configs: list[ExperimentConfig], output_dir: Path) -> None:
    """Run multiple experiments and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Experiment: {config.name}")
        print(f"  Backend: {config.backend.value}")
        print(f"  Workload: {config.workload}")
        print(f"  Programs: {config.num_programs} (+ {config.warmup_programs} warmup)")
        print(f"  Repetitions: {config.num_repetitions}")
        print(f"  Arrival rate: {config.arrival_rate} prog/sec")
        print(f"{'='*60}")

        metrics = await run_experiment(config)

        # Save raw metrics
        csv_path = output_dir / f"{config.name}.csv"
        save_metrics_csv(metrics, csv_path)
        print(f"  Saved {len(metrics)} metrics to {csv_path}")

        # Print summary
        programs = aggregate_program_metrics(metrics)
        summary = compute_experiment_summary(programs)
        summary_path = output_dir / f"{config.name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  Summary: mean TTFT = {summary.get('mean_ttft_ms', 0):.1f}ms, "
              f"p95 TTFT = {summary.get('p95_ttft_ms', 0):.1f}ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIE Modular KV Cache Benchmark")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to experiment config JSON file",
    )
    parser.add_argument(
        "--workload",
        choices=["react", "multiturn", "skill_switch", "heartbeat"],
        default="skill_switch",
        help="Workload scenario to run (default: skill_switch)",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "pie_std", "pie_cache"],
        default="vllm",
        help="Backend to benchmark (default: vllm)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--num-programs",
        type=int,
        default=10,
        help="Number of programs per experiment (default: 10)",
    )
    parser.add_argument(
        "--num-repetitions",
        type=int,
        default=3,
        help="Repetitions per experiment (default: 3)",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=0.0,
        help="Poisson arrival rate in programs/sec (0 = sequential, default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
    )
    parser.add_argument(
        "--pie-uri",
        type=str,
        default="ws://localhost:8080",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        # Load from config file
        config_data = json.loads(args.config.read_text())
        if isinstance(config_data, list):
            configs = [ExperimentConfig(**c) for c in config_data]
        else:
            configs = [ExperimentConfig(**config_data)]
    else:
        # Build from CLI args
        configs = [ExperimentConfig(
            name=f"{args.workload}_{args.backend}",
            backend=BackendType(args.backend),
            workload=args.workload,
            model=args.model,
            num_programs=args.num_programs,
            arrival_rate=args.arrival_rate,
            num_repetitions=args.num_repetitions,
            vllm_url=args.vllm_url,
            pie_server_uri=args.pie_uri,
        )]

    asyncio.run(run_experiment_suite(configs, args.output_dir))


if __name__ == "__main__":
    main()
