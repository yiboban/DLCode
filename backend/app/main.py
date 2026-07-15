from __future__ import annotations

import ast
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from .database import Base, SessionLocal, engine, get_db
from .judge import judge_code, normalize_custom_tests
from .models import Draft, Problem, ProblemProgress, Submission
from .problem_bank import get_seed_problems
from .schemas import (
    DraftRequest,
    DraftResponse,
    JudgeResponse,
    ProblemDetail,
    ProblemListItem,
    ProblemsResponse,
    StatisticsResponse,
    SubmissionDetail,
    SubmissionItem,
    SubmitRequest,
    RunRequest,
)


def acceptance_rate(problem: Problem) -> float:
    if problem.submit_count == 0:
        return 0.0
    return round(problem.pass_count / problem.submit_count * 100, 2)


def is_untouched_placeholder(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            continue
        return False
    return True


def seed_database() -> None:
    Base.metadata.create_all(bind=engine)
    columns = {column["name"] for column in inspect(engine).get_columns("problems")}
    if "presentation" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE problems ADD COLUMN presentation JSON"))
    problems = get_seed_problems()
    with SessionLocal() as db:
        for data in problems:
            problem = db.scalar(select(Problem).where(Problem.slug == data["slug"]))
            if problem is None:
                db.add(Problem(**data))
                continue
            for key, value in data.items():
                if key in {"pass_count", "submit_count"}:
                    continue
                setattr(problem, key, value)
        db.flush()
        for draft in db.scalars(select(Draft)).all():
            if is_untouched_placeholder(draft.code):
                problem = db.get(Problem, draft.problem_id)
                if problem is not None:
                    draft.code = problem.starter_code
        db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_database()
    yield


app = FastAPI(
    title="DLCode API",
    description="DLCode：深度学习手撕题库本地判题服务",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DbDep = Annotated[Session, Depends(get_db)]


def get_problem_or_404(db: Session, problem_id: int) -> Problem:
    problem = db.get(Problem, problem_id)
    if problem is None:
        raise HTTPException(status_code=404, detail={"message": "题目不存在"})
    return problem


def get_problem_by_slug_or_404(db: Session, slug: str) -> Problem:
    problem = db.scalar(select(Problem).where(Problem.slug == slug))
    if problem is None:
        raise HTTPException(status_code=404, detail={"message": "题目不存在"})
    return problem


def completed_map(db: Session) -> dict[int, bool]:
    rows = db.scalars(select(ProblemProgress)).all()
    return {row.problem_id: row.completed for row in rows}


def to_problem_item(problem: Problem, completed: bool) -> ProblemListItem:
    return ProblemListItem(
        id=problem.id,
        slug=problem.slug,
        title=problem.title,
        difficulty=problem.difficulty,
        categories=problem.categories,
        company_tags=problem.company_tags,
        source_note=problem.source_note,
        acceptance_rate=acceptance_rate(problem),
        completed=completed,
        submit_count=problem.submit_count,
        pass_count=problem.pass_count,
    )


def to_problem_detail(problem: Problem, completed: bool) -> ProblemDetail:
    return ProblemDetail(
        id=problem.id,
        slug=problem.slug,
        title=problem.title,
        difficulty=problem.difficulty,
        categories=problem.categories,
        company_tags=problem.company_tags,
        source_note=problem.source_note,
        description=problem.description,
        function_name=problem.function_name,
        function_signature=problem.function_signature,
        starter_code=problem.starter_code,
        explanation=problem.explanation,
        presentation=problem.presentation or {"formulas": [], "symbols": [], "steps": [problem.explanation]},
        constraints=problem.constraints,
        examples=problem.examples,
        public_tests=problem.public_tests,
        time_limit=problem.time_limit,
        memory_limit=problem.memory_limit,
        acceptance_rate=acceptance_rate(problem),
        completed=completed,
    )


def complete_expected_outputs(problem: Problem, tests: list[dict]) -> list[dict]:
    missing_indices = [index for index, test in enumerate(tests) if test.get("expected") is None]
    if not missing_indices:
        return tests

    solution_result = judge_code(problem, problem.solution_code, tests, timeout_multiplier=4.0)
    if solution_result.get("status_code") != "ACCEPTED":
        raise HTTPException(
            status_code=500,
            detail={"message": "参考答案无法生成自定义测试的预期输出，请检查输入格式。"},
        )

    results = solution_result.get("results", [])
    enriched: list[dict] = []
    for index, test in enumerate(tests):
        next_test = dict(test)
        if index in missing_indices:
            if index >= len(results):
                raise HTTPException(status_code=500, detail={"message": "参考答案生成的测试结果数量不一致。"})
            next_test["expected"] = results[index].get("actual_output")
        enriched.append(next_test)
    return enriched


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/problems", response_model=ProblemsResponse)
def list_problems(
    db: DbDep,
    search: str | None = Query(default=None),
    category: str | None = Query(default=None),
    difficulty: str | None = Query(default=None),
    company: str | None = Query(default=None),
    status: str | None = Query(default=None),
    sort_by: str = Query(default="id", pattern="^(id|difficulty|acceptance_rate)$"),
    order: str = Query(default="asc", pattern="^(asc|desc)$"),
) -> ProblemsResponse:
    problems = list(db.scalars(select(Problem)).all())
    progress = completed_map(db)

    def match(problem: Problem) -> bool:
        if search and search.strip().lower() not in problem.title.lower() and search.strip() not in str(problem.id):
            return False
        if category and category not in problem.categories:
            return False
        if difficulty and difficulty != problem.difficulty:
            return False
        if company and company not in problem.company_tags:
            return False
        if status == "已完成" and not progress.get(problem.id, False):
            return False
        if status == "未完成" and progress.get(problem.id, False):
            return False
        return True

    filtered = [problem for problem in problems if match(problem)]
    difficulty_rank = {"简单": 0, "中等": 1, "困难": 2}
    if sort_by == "difficulty":
        filtered.sort(key=lambda item: (difficulty_rank.get(item.difficulty, 99), item.id))
    elif sort_by == "acceptance_rate":
        filtered.sort(key=lambda item: (acceptance_rate(item), item.id))
    else:
        filtered.sort(key=lambda item: item.id)
    if order == "desc":
        filtered.reverse()

    all_categories = sorted({cat for problem in problems for cat in problem.categories})
    all_companies = sorted({tag for problem in problems for tag in problem.company_tags})
    return ProblemsResponse(
        items=[to_problem_item(problem, progress.get(problem.id, False)) for problem in filtered],
        total=len(filtered),
        categories=all_categories,
        companies=all_companies,
    )


@app.get("/api/problems/{slug}", response_model=ProblemDetail)
def get_problem(slug: str, db: DbDep) -> ProblemDetail:
    problem = get_problem_by_slug_or_404(db, slug)
    progress = completed_map(db)
    return to_problem_detail(problem, progress.get(problem.id, False))


@app.post("/api/run", response_model=JudgeResponse)
def run_code(payload: RunRequest, db: DbDep) -> JudgeResponse:
    problem = get_problem_or_404(db, payload.problem_id)
    custom_tests = normalize_custom_tests(payload.custom_tests)
    tests = complete_expected_outputs(problem, custom_tests) if custom_tests else problem.public_tests
    result = judge_code(problem, payload.code, tests, timeout_multiplier=1.0)
    return JudgeResponse(**result)


@app.post("/api/submit", response_model=JudgeResponse)
def submit_code(payload: SubmitRequest, db: DbDep) -> JudgeResponse:
    problem = get_problem_or_404(db, payload.problem_id)
    tests = problem.public_tests + problem.hidden_tests
    result = judge_code(problem, payload.code, tests, timeout_multiplier=1.0)

    problem.submit_count += 1
    if result["status_code"] == "ACCEPTED":
        problem.pass_count += 1

    progress = db.scalar(select(ProblemProgress).where(ProblemProgress.problem_id == problem.id))
    if progress is None:
        progress = ProblemProgress(problem_id=problem.id, completed=False)
        db.add(progress)
    progress.last_status = result["status"]
    if result["status_code"] == "ACCEPTED":
        progress.completed = True

    submission = Submission(
        problem_id=problem.id,
        problem_slug=problem.slug,
        problem_title=problem.title,
        status=result["status"],
        code=payload.code,
        language=payload.language,
        runtime_ms=result["runtime_ms"],
        passed_tests=result["passed_tests"],
        total_tests=result["total_tests"],
        error_sample=result.get("first_error"),
        result_detail=result,
    )
    db.add(submission)
    db.commit()
    return JudgeResponse(**result)


@app.get("/api/submissions", response_model=list[SubmissionItem])
def list_submissions(db: DbDep, limit: int = Query(default=50, ge=1, le=200)) -> list[SubmissionItem]:
    rows = db.scalars(select(Submission).order_by(Submission.created_at.desc()).limit(limit)).all()
    return [SubmissionItem.model_validate(row) for row in rows]


@app.get("/api/submissions/{submission_id}", response_model=SubmissionDetail)
def get_submission(submission_id: int, db: DbDep) -> SubmissionDetail:
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail={"message": "提交记录不存在"})
    return SubmissionDetail.model_validate(submission)


@app.put("/api/drafts/{problem_id}", response_model=DraftResponse)
def save_draft(problem_id: int, payload: DraftRequest, db: DbDep) -> DraftResponse:
    get_problem_or_404(db, problem_id)
    draft = db.scalar(select(Draft).where(Draft.problem_id == problem_id))
    if draft is None:
        draft = Draft(problem_id=problem_id, code=payload.code)
        db.add(draft)
    else:
        draft.code = payload.code
    db.commit()
    db.refresh(draft)
    return DraftResponse(problem_id=problem_id, code=draft.code, updated_at=draft.updated_at)


@app.get("/api/drafts/{problem_id}", response_model=DraftResponse)
def get_draft(problem_id: int, db: DbDep) -> DraftResponse:
    problem = get_problem_or_404(db, problem_id)
    draft = db.scalar(select(Draft).where(Draft.problem_id == problem_id))
    if draft is None:
        return DraftResponse(problem_id=problem_id, code=problem.starter_code, updated_at=None)
    return DraftResponse(problem_id=problem_id, code=draft.code, updated_at=draft.updated_at)


@app.get("/api/statistics", response_model=StatisticsResponse)
def statistics(db: DbDep) -> StatisticsResponse:
    total = db.scalar(select(func.count(Problem.id))) or 0
    progress = completed_map(db)
    completed = sum(1 for done in progress.values() if done)
    problems = list(db.scalars(select(Problem)).all())
    by_difficulty = {}
    for difficulty in ["简单", "中等", "困难"]:
        subset = [problem for problem in problems if problem.difficulty == difficulty]
        by_difficulty[difficulty] = {
            "total": len(subset),
            "completed": sum(1 for problem in subset if progress.get(problem.id, False)),
        }
    recent = db.scalars(select(Submission).order_by(Submission.created_at.desc()).limit(8)).all()
    categories = sorted({cat for problem in problems for cat in problem.categories})
    return StatisticsResponse(
        total_problems=total,
        completed_problems=completed,
        by_difficulty=by_difficulty,
        recent_submissions=[SubmissionItem.model_validate(row) for row in recent],
        categories=categories,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    from fastapi.responses import JSONResponse

    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=detail)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=500, content={"message": "服务内部错误", "detail": str(exc)[:300]})
