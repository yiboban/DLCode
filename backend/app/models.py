from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from .database import Base


class Problem(Base):
    __tablename__ = "problems"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    difficulty: Mapped[str] = mapped_column(String(20), index=True)
    categories: Mapped[list[str]] = mapped_column(JSON)
    company_tags: Mapped[list[str]] = mapped_column(JSON)
    source_note: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    function_name: Mapped[str] = mapped_column(String(120))
    function_signature: Mapped[str] = mapped_column(String(240))
    starter_code: Mapped[str] = mapped_column(Text)
    solution_code: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    constraints: Mapped[list[str]] = mapped_column(JSON)
    examples: Mapped[list[dict]] = mapped_column(JSON)
    public_tests: Mapped[list[dict]] = mapped_column(JSON)
    hidden_tests: Mapped[list[dict]] = mapped_column(JSON)
    time_limit: Mapped[float] = mapped_column(Float, default=2.0)
    memory_limit: Mapped[int] = mapped_column(Integer, default=256)
    pass_count: Mapped[int] = mapped_column(Integer, default=0)
    submit_count: Mapped[int] = mapped_column(Integer, default=0)

    drafts: Mapped[list["Draft"]] = relationship(back_populates="problem")
    submissions: Mapped[list["Submission"]] = relationship(back_populates="problem")


class Draft(Base):
    __tablename__ = "drafts"
    __table_args__ = (UniqueConstraint("problem_id", name="uq_drafts_problem_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    problem_id: Mapped[int] = mapped_column(ForeignKey("problems.id"), index=True)
    code: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    problem: Mapped[Problem] = relationship(back_populates="drafts")


class ProblemProgress(Base):
    __tablename__ = "problem_progress"
    __table_args__ = (UniqueConstraint("problem_id", name="uq_progress_problem_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    problem_id: Mapped[int] = mapped_column(ForeignKey("problems.id"), index=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    last_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    problem_id: Mapped[int] = mapped_column(ForeignKey("problems.id"), index=True)
    problem_slug: Mapped[str] = mapped_column(String(120), index=True)
    problem_title: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(40), index=True)
    code: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(40), default="Python 3")
    runtime_ms: Mapped[float] = mapped_column(Float, default=0)
    passed_tests: Mapped[int] = mapped_column(Integer, default=0)
    total_tests: Mapped[int] = mapped_column(Integer, default=0)
    error_sample: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    problem: Mapped[Problem] = relationship(back_populates="submissions")
