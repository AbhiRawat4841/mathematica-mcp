"""Shared constants for mathematica-mcp.

Centralised to prevent string typos and ensure consistent naming
across server, routing_memory, transport_classification, and tests.
"""

from __future__ import annotations

import enum


class ExecutionPath:
    """Execution path labels recorded in routing telemetry."""

    ADDON_NOTEBOOK = "addon_notebook"
    ADDON_CLI = "addon_cli"
    KERNEL_FALLBACK = "kernel_fallback"
    KERNEL_DIRECT_ROUTING_SKIP = "kernel_direct_routing_skip"


class AttemptOutcome(enum.Enum):
    """Typed outcome of a single transport attempt.

    Used by the circuit breaker and attempt-level telemetry.
    Only INFRA_ERROR trips the breaker; TIMEOUT and SEMANTIC_ERROR do not.
    """

    OK = "ok"
    INFRA_ERROR = "infra_error"
    TIMEOUT = "timeout"
    SEMANTIC_ERROR = "semantic_error"
