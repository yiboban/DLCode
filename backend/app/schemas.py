from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiError(BaseModel):
    message: str
    detail: str | None = None


class ProblemListItem(BaseModel):
    id: int
    slug: str
    title: str
    difficulty: str
    categories: list[str]
    company_tags: list[str]
    source_note: str
    acceptance_rate: float
    completed: bool
    submit_count: int
    pass_count: int


class FormulaItem(BaseModel):
    latex: str
    label: str | None = None


class ProblemPresentation(BaseModel):
    formulas: list[FormulaItem] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


class ProblemDetail(BaseModel):
    id: int
    slug: str
    title: str
    difficulty: str
    categories: list[str]
    company_tags: list[str]
    source_note: str
    description: str
    function_name: str
    function_signature: str
    starter_code: str
    explanation: str
    presentation: ProblemPresentation
    constraints: list[str]
    examples: list[dict[str, Any]]
    public_tests: list[dict[str, Any]]
    time_limit: float
    memory_limit: int
    acceptance_rate: float
    completed: bool


class ProblemsResponse(BaseModel):
    items: list[ProblemListItem]
    total: int
    categories: list[str]
    companies: list[str]


class DraftRequest(BaseModel):
    code: str = Field(min_length=0)


class DraftResponse(BaseModel):
    problem_id: int
    code: str
    updated_at: datetime | None = None


class RunRequest(BaseModel):
    problem_id: int
    code: str
    custom_tests: list[Any] | None = None
    language: str = "Python 3"


class SubmitRequest(BaseModel):
    problem_id: int
    code: str
    language: str = "Python 3"


class TestCaseResult(BaseModel):
    passed: bool
    input: Any = None
    expected_output: Any = None
    actual_output: Any = None
    stdout: str = ""
    stderr: str = ""
    runtime_ms: float = 0
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None


class JudgeResponse(BaseModel):
    status: str
    status_code: str
    passed_tests: int
    total_tests: int
    runtime_ms: float
    results: list[TestCaseResult]
    first_error: dict[str, Any] | None = None


class SubmissionItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    problem_id: int
    problem_slug: str
    problem_title: str
    status: str
    language: str
    runtime_ms: float
    passed_tests: int
    total_tests: int
    created_at: datetime


class SubmissionDetail(SubmissionItem):
    code: str
    error_sample: dict[str, Any] | None = None
    result_detail: dict[str, Any] | None = None


class DifficultyStat(BaseModel):
    total: int
    completed: int


class StatisticsResponse(BaseModel):
    total_problems: int
    completed_problems: int
    by_difficulty: dict[str, DifficultyStat]
    recent_submissions: list[SubmissionItem]
    categories: list[str]
