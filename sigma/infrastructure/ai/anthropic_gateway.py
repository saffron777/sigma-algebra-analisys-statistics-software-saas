"""Deprecated: the Anthropic code moved to :mod:`sigma.infrastructure.ai.adapters` (multi-provider AI).

This module only re-exports the old names so an installation that still imports them keeps working.
"""

from __future__ import annotations

from sigma.infrastructure.ai.adapters import AnthropicAdapter
from sigma.infrastructure.ai.providers import (
    ANTHROPIC_FALLBACK_BETA as FALLBACK_BETA,
    ANTHROPIC_FALLBACK_MODEL as FALLBACK_MODEL,
    TEST_SYSTEM,
    TEST_USER,
)
from sigma.infrastructure.ai.http import TIMEOUT_SECONDS
from sigma.domain.ai import DEFAULT_MAX_TOKENS as MAX_TOKENS

__all__ = [
    "AnthropicAdapter",
    "FALLBACK_BETA",
    "FALLBACK_MODEL",
    "MAX_TOKENS",
    "TEST_SYSTEM",
    "TEST_USER",
    "TIMEOUT_SECONDS",
]
