"""Core data contracts.

Every layer of the application speaks this language:
    use case  ->  RunResult (KPIs, charts, tables, recommendation, story)
    API       ->  RunRequest / RunResult / LogEvent (JSON over SSE)
    reporting ->  RunResult -> HTML / XLSX / CSV / PDF

Nothing here imports numpy, polars or any model code. It is pure shape.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Use-case self description (drives the whole UI: no hard-coded screens)
# ---------------------------------------------------------------------------
class ParamSpec(BaseModel):
    """One knob the user can turn. Rendered automatically by the front-end."""
    key: str
    label: str
    kind: str = "number"          # number | integer | choice | bool
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: list[dict[str, Any]] = Field(default_factory=list)  # [{value,label}]
    unit: Optional[str] = None
    group: str = "Assumptions"
    help: str = ""
    # QA TM-16: params sharing a non-empty `sum_group` must add up to 100. The UI
    # shows the running total; the use case validates and normalises them.
    sum_group: Optional[str] = None
    sum_target: float = 100.0


class RoleSpec(BaseModel):
    """A column *role* the use case needs.

    Roles (not column names) are what make the tool dimension-agnostic: the
    user maps whatever their file calls it onto a role, or we auto-map by alias,
    or we fall back to a derived default.
    """
    key: str
    label: str
    dtype: str = "numeric"        # numeric | categorical | datetime | boolean | identifier
    required: bool = True
    aliases: list[str] = Field(default_factory=list)
    fallback: Any = None          # value (or {"derive": "..."} hint) if column absent
    help: str = ""


class UseCaseMeta(BaseModel):
    id: str
    title: str
    subtitle: str
    category: str                 # e.g. "Deal Desk", "Margin", "Pipeline"
    icon: str = ""                # emoji-ish token, optional
    models: list[str] = Field(default_factory=list)   # engine keys used
    business_question: str        # one plain sentence
    plain_summary: str            # 2-3 sentences a non-analyst understands
    story_steps: list[str] = Field(default_factory=list)  # what happens, in order
    data_granularity: str         # deal | line_item | transition | cost | portfolio | event
    input_roles: list[RoleSpec] = Field(default_factory=list)
    parameters: list[ParamSpec] = Field(default_factory=list)
    sample_dataset: Optional[str] = None  # filename in data/samples


# ---------------------------------------------------------------------------
# Dataset side
# ---------------------------------------------------------------------------
class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_pct: float = 0.0
    unique: int = 0
    sample_values: list[Any] = Field(default_factory=list)
    numeric_stats: Optional[dict[str, float]] = None


class DatasetProfile(BaseModel):
    dataset_id: str
    source_name: str
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    warnings: list[str] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list)  # first ~8 rows
    # QA NEW-05: when the app uses fewer rows than the file holds it must say so
    # everywhere the number appears - in the run log and in the result - never
    # only in a panel the user may never open.
    sampled: bool = False
    #: how many rows the file actually had (0 when we could not count them)
    n_rows_in_file: int = 0
    #: one plain sentence naming both numbers, ready to show the user
    sample_note: str = ""


class RoleMapping(BaseModel):
    """role_key -> column name (or {"const": value} / {"derive": "..."})."""
    mapping: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run request / result
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    use_case_id: str
    dataset_id: Optional[str] = None
    rows: Optional[list[dict[str, Any]]] = None   # paste-table / manual form path
    manual_inputs: dict[str, Any] = Field(default_factory=dict)
    role_mapping: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    # QA DF-10 / UX-04: the parameter keys the user actually typed at Step 4.
    # Distinct from `parameters`, which is merged with defaults, so a use case
    # can tell "the user asked for this" from "this is just the default".
    # A key present here must win over a value learned from the uploaded file.
    explicit_params: list[str] = Field(default_factory=list)
    seed: int = 42
    output_formats: list[str] = Field(default_factory=lambda: ["html"])


class LogEvent(BaseModel):
    t: float = 0.0                 # seconds since run start
    phase: str = "run"             # read|validate|map|prepare|simulate|summarize|report
    level: str = "info"            # info|warn|error|success
    message: str                   # PLAIN LANGUAGE, one sentence, no jargon
    detail: Optional[str] = None   # optional technical detail, shown in "more"
    progress: Optional[float] = None  # 0..1


class KPI(BaseModel):
    label: str
    value: str                     # pre-formatted for humans, e.g. "$176,400"
    unit: Optional[str] = None
    tone: str = "neutral"          # good | bad | warn | neutral
    plain: str = ""                # what this number means in one sentence
    comparison: Optional[str] = None  # e.g. "+41k vs 18% discount"


class ChartSpec(BaseModel):
    id: str
    type: str                      # line|bar|grouped_bar|hist|scatter|heatmap|funnel|stacked_area|waterfall
    title: str
    subtitle: str = ""
    x_label: str = ""
    y_label: str = ""
    x: list[Any] = Field(default_factory=list)
    series: list[dict[str, Any]] = Field(default_factory=list)  # [{name, values:[...], color?}]
    annotations: list[dict[str, Any]] = Field(default_factory=list)  # [{x, y, text}]
    plain_meaning: str = ""


class ResultTable(BaseModel):
    name: str
    description: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    highlight_rows: list[int] = Field(default_factory=list)
    highlight_note: str = ""


class Recommendation(BaseModel):
    headline: str
    rationale: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    risk_score: Optional[float] = None     # 0-100
    risk_label: Optional[str] = None       # Low / Moderate / High / Critical
    confidence: str = "Medium"             # Low / Medium / High
    tradeoff: str = ""


class StoryStep(BaseModel):
    title: str
    detail: str                    # plain language
    evidence: Optional[str] = None # the number that backs it


class RunResult(BaseModel):
    run_id: str
    use_case_id: str
    use_case_title: str = ""
    seed: int = 42
    status: str = "succeeded"
    headline: str = ""             # the one-line answer
    method_plain: str = ""         # how the model works, in layman terms
    model_used: str = ""
    n_simulations: int = 0
    runtime_ms: int = 0
    story: list[StoryStep] = Field(default_factory=list)
    kpis: list[KPI] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    tables: list[ResultTable] = Field(default_factory=list)
    recommendation: Optional[Recommendation] = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    log: list[LogEvent] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)  # format -> filename


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
