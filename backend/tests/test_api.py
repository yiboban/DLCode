from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.database import BASE_DIR, SessionLocal
from backend.app.main import app
from backend.app.models import Problem
from backend.app.problem_bank import get_seed_problems


@pytest.fixture(scope="session")
def client() -> TestClient:
    db_path = Path(BASE_DIR) / "dlcode.db"
    if db_path.exists():
        db_path.unlink()
    with TestClient(app) as test_client:
        yield test_client


def seed_problem(slug: str) -> dict:
    return next(problem for problem in get_seed_problems() if problem["slug"] == slug)


def problem_id(client: TestClient, slug: str) -> int:
    response = client.get(f"/api/problems/{slug}")
    assert response.status_code == 200
    return response.json()["id"]


def test_problem_list_endpoint(client: TestClient) -> None:
    response = client.get("/api/problems")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 60
    assert "Python 与 NumPy 基础" in data["categories"]
    assert any(item["title"] == "矩阵转置" for item in data["items"])


def test_problem_detail_hides_private_fields(client: TestClient) -> None:
    response = client.get("/api/problems/matrix-transpose")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "矩阵转置"
    assert "hidden_tests" not in data
    assert "solution_code" not in data
    assert len(data["public_tests"]) == 3


def test_correct_code_submit_and_submission_saved(client: TestClient) -> None:
    problem = seed_problem("matrix-transpose")
    response = client.post(
        "/api/submit",
        json={"problem_id": problem["id"], "code": problem["solution_code"], "language": "Python 3"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "通过"
    assert result["passed_tests"] == 8

    submissions = client.get("/api/submissions").json()
    assert submissions
    assert submissions[0]["problem_title"]


def test_wrong_answer_returns_error_sample(client: TestClient) -> None:
    pid = problem_id(client, "matrix-transpose")
    code = "def matrix_transpose(matrix):\n    return []\n"
    response = client.post("/api/run", json={"problem_id": pid, "code": code})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "答案错误"
    assert result["first_error"]["input"] is not None
    assert result["first_error"]["expected_output"] is not None


def test_syntax_error(client: TestClient) -> None:
    pid = problem_id(client, "matrix-transpose")
    response = client.post("/api/run", json={"problem_id": pid, "code": "def broken(:\n    pass\n"})
    assert response.status_code == 200
    assert response.json()["status"] == "语法错误"


def test_runtime_error(client: TestClient) -> None:
    pid = problem_id(client, "matrix-transpose")
    code = "def matrix_transpose(matrix):\n    return 1 / 0\n"
    response = client.post("/api/run", json={"problem_id": pid, "code": code})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "运行错误"
    assert "ZeroDivisionError" in data["first_error"]["traceback"]


def test_infinite_loop_timeout(client: TestClient) -> None:
    pid = problem_id(client, "matrix-transpose")
    with SessionLocal() as db:
        problem = db.get(Problem, pid)
        assert problem is not None
        problem.time_limit = 0.2
        db.commit()
    code = "def matrix_transpose(matrix):\n    while True:\n        pass\n"
    response = client.post("/api/run", json={"problem_id": pid, "code": code})
    assert response.status_code == 200
    assert response.json()["status"] == "超出时间限制"


def test_float_tolerance_comparison(client: TestClient) -> None:
    problem = seed_problem("l2-normalize-vector")
    code = """
import math

def l2_normalize(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [round(v / norm, 7) for v in vector]
"""
    response = client.post("/api/submit", json={"problem_id": problem["id"], "code": code})
    assert response.status_code == 200
    assert response.json()["status"] == "通过"


def test_numpy_array_comparison(client: TestClient) -> None:
    problem = seed_problem("numpy-broadcast-add")
    response = client.post("/api/submit", json={"problem_id": problem["id"], "code": problem["solution_code"]})
    assert response.status_code == 200
    assert response.json()["status"] == "通过"


def test_torch_tensor_comparison(client: TestClient) -> None:
    problem = seed_problem("torch-add-relu")
    response = client.post("/api/submit", json={"problem_id": problem["id"], "code": problem["solution_code"]})
    assert response.status_code == 200
    assert response.json()["status"] == "通过"


def test_draft_save_and_restore(client: TestClient) -> None:
    pid = problem_id(client, "matrix-transpose")
    code = "def matrix_transpose(matrix):\n    return matrix\n"
    save = client.put(f"/api/drafts/{pid}", json={"code": code})
    assert save.status_code == 200
    restored = client.get(f"/api/drafts/{pid}")
    assert restored.status_code == 200
    assert restored.json()["code"] == code
