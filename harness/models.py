"""Core data models shared across all harness components."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class BackendType(str, Enum):
    VLLM = "vllm"
    PIE_STD = "pie_std"
    PIE_CACHE = "pie_cache"


@dataclass(frozen=True)
class PromptModule:
    """A single module within a modular prompt."""

    name: str
    content: str
    content_hash: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.content_hash:
            h = hashlib.sha256(self.content.encode()).hexdigest()
            object.__setattr__(self, "content_hash", h)

    @property
    def token_count_estimate(self) -> int:
        """Rough estimate: ~4 chars per token for English text."""
        return max(1, len(self.content) // 4)


@dataclass
class ModularRequest:
    """A single LLM request composed of ordered modules."""

    program_id: str
    turn_index: int
    modules: list[PromptModule]
    max_response_tokens: int = 200

    @property
    def total_prompt_tokens_estimate(self) -> int:
        return sum(m.token_count_estimate for m in self.modules)

    def flat_prompt(self) -> str:
        """Concatenate all modules into a single prompt string."""
        return "\n".join(m.content for m in self.modules)


@dataclass
class RequestMetrics:
    """Metrics collected for a single LLM request."""

    program_id: str
    turn_index: int
    backend: BackendType
    ttft_ms: float
    total_latency_ms: float
    tokens_generated: int
    generated_text: str = ""
    # PIE-cache specific (0 for other backends)
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_saved: int = 0
    tokens_computed: int = 0

    @property
    def tpot_ms(self) -> float:
        """Time per output token."""
        if self.tokens_generated <= 1:
            return 0.0
        return (self.total_latency_ms - self.ttft_ms) / (self.tokens_generated - 1)

    @property
    def prefill_ratio(self) -> float:
        """Fraction of prefill tokens served from cache."""
        total = self.tokens_saved + self.tokens_computed
        if total == 0:
            return 0.0
        return self.tokens_saved / total


@dataclass
class ProgramMetrics:
    """Aggregated metrics for a complete agent session."""

    program_id: str
    backend: BackendType
    request_metrics: list[RequestMetrics] = field(default_factory=list)

    @property
    def job_completion_time_ms(self) -> float:
        """Wall-clock time from first request to last completion."""
        if not self.request_metrics:
            return 0.0
        return sum(m.total_latency_ms for m in self.request_metrics)

    @property
    def total_llm_time_ms(self) -> float:
        return sum(m.total_latency_ms for m in self.request_metrics)

    @property
    def mean_ttft_ms(self) -> float:
        if not self.request_metrics:
            return 0.0
        return sum(m.ttft_ms for m in self.request_metrics) / len(self.request_metrics)

    @property
    def cache_hit_rate(self) -> float:
        total_hits = sum(m.cache_hits for m in self.request_metrics)
        total_modules = sum(m.cache_hits + m.cache_misses for m in self.request_metrics)
        if total_modules == 0:
            return 0.0
        return total_hits / total_modules

    @property
    def total_prefill_savings_tokens(self) -> int:
        return sum(m.tokens_saved for m in self.request_metrics)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str
    backend: BackendType
    workload: str  # scenario name
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    num_programs: int = 20
    arrival_rate: float = 0.1  # lambda for Poisson (programs/sec)
    warmup_programs: int = 5
    num_repetitions: int = 10
    temperature: float = 0.0  # greedy decoding
    max_response_tokens: int = 200
    # Backend-specific
    vllm_url: str = "http://localhost:8000"
    pie_server_uri: str = "ws://localhost:8080"
    # Workload-specific params
    workload_params: dict = field(default_factory=dict)


@dataclass
class StreamingResponse:
    """Response from a backend, accumulated during streaming."""

    tokens: list[str] = field(default_factory=list)
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    # PIE-cache specific
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_saved: int = 0
    tokens_computed: int = 0

    @property
    def text(self) -> str:
        return "".join(self.tokens)

    @property
    def tokens_generated(self) -> int:
        return len(self.tokens)
