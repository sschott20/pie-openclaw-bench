"""Prompt content modules for benchmark workloads.

Dispatches to ``real`` (actual Claude Code content) or ``synthetic``
(placeholder content) based on ``PromptSizes.prompt_source``.

Workloads call ``get_module(source)`` to get the right content module,
then call its factory functions directly.
"""

from __future__ import annotations

import types

# Re-export PromptSizes for convenience
from harness.prompts.synthetic import PromptSizes  # noqa: F401


def get_module(source: str = "real") -> types.ModuleType:
    """Return the prompt content module for *source* ('real' or 'synthetic')."""
    if source == "real":
        from harness.prompts import real as m
    else:
        from harness.prompts import synthetic as m
    return m
