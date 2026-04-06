"""Pydantic schema for the Mathematica MCP corpus test manifest.

Defines two item kinds:
- CorpusCase: single tool invocation with an oracle assertion
- CorpusWorkflow: multi-step stateful scenario with setup, steps, and cleanup
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# -- Constrained Literal types ------------------------------------------------

Capability = Literal[
    "wolfram_runtime",
    "live_addon",
    "frontend",
    "network",
    "resource",
    "subkernels",
]

Backend = Literal[
    "server_tool",
    "offline_notebook_file",
    "live_frontend",
    "alias_codegen",
]

Tier = Literal["smoke", "core", "extended", "probe"]

Profile = Literal["math", "notebook", "full"]

OracleType = Literal[
    "field_equals",
    "field_contains",
    "exact_text",
    "symbolic_equiv",
    "numeric_tol",
    "boolean",
    "structural_fields",
    "artifact_exists",
    "warning_tag",
    "raw_contains",
    "workflow_assert",
]


# -- Sub-models ---------------------------------------------------------------


class PollCondition(BaseModel):
    """Structured polling condition for async workflow steps."""

    path: str
    op: Literal["==", "!=", "in", "not_in"]
    value: Any


class Oracle(BaseModel):
    """Describes what success looks like for a test case or workflow step."""

    type: OracleType
    path: str | None = None
    value: Any | None = None
    tolerance: float = 1e-5
    contains: list[str] | None = None
    checks: list[str] | None = None


class SaveSpec(BaseModel):
    """Describes a value to extract from a step result into workflow state."""

    state_key: str
    path: str  # dot-path into NormalizedResult: "parsed.id"


class WorkflowStep(BaseModel):
    """A single step within a workflow."""

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    params_from_state: dict[str, str] = Field(default_factory=dict)
    save: list[SaveSpec] = Field(default_factory=list)
    oracle: Oracle | None = None
    backend: Backend | None = None  # per-step override
    timeout_s: int = 30
    poll: PollCondition | None = None
    max_attempts: int = 1
    delay_ms: int = 500


# -- Top-level items ----------------------------------------------------------


class CorpusCase(BaseModel):
    """An atomic single-tool test case."""

    id: str
    title: str
    kind: Literal["case"] = "case"
    tier: Tier
    section: str
    min_profile: Profile
    backend: Backend
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    oracle: Oracle
    required_capabilities: list[Capability] = Field(default_factory=list)
    timeout_s: int = 30
    seed: int | None = None


class CorpusWorkflow(BaseModel):
    """A multi-step stateful workflow with setup, steps, and cleanup."""

    id: str
    title: str
    kind: Literal["workflow"] = "workflow"
    tier: Tier
    section: str
    min_profile: Profile
    backend: Backend
    required_capabilities: list[Capability] = Field(default_factory=list)
    steps: list[WorkflowStep] = Field(default_factory=list)
    final_oracle: Oracle
    cleanup: list[WorkflowStep] = Field(default_factory=list)
    timeout_s: int = 60


CorpusItem = CorpusCase | CorpusWorkflow
