from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest
from fastapi.testclient import TestClient

TEST_DB_PATH = Path(tempfile.gettempdir()) / f"dlcode-test-{os.getpid()}.db"
os.environ["DLCODE_DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"

from backend.app.database import SessionLocal, engine
from backend.app.main import app, is_untouched_placeholder
from backend.app.models import Problem
from backend.app.problem_bank import get_seed_problems


@pytest.fixture(scope="session")
def client() -> TestClient:
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()
    TEST_DB_PATH.unlink(missing_ok=True)


def seed_problem(slug: str) -> dict:
    return next(problem for problem in get_seed_problems() if problem["slug"] == slug)


def problem_id(client: TestClient, slug: str) -> int:
    response = client.get(f"/api/problems/{slug}")
    assert response.status_code == 200
    return response.json()["id"]


def test_placeholder_draft_detection() -> None:
    assert is_untouched_placeholder("import numpy as np\n\ndef f(x) -> np.ndarray:\n    pass\n")
    assert not is_untouched_placeholder("def f(x):\n    return x\n")
    assert not is_untouched_placeholder("def broken(:\n    pass\n")


def test_problem_list_endpoint(client: TestClient) -> None:
    response = client.get("/api/problems")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 80
    assert "Python 与 PyTorch 基础" in data["categories"]
    assert "大模型核心组件" in data["categories"]
    assert any(item["title"] == "矩阵转置" for item in data["items"])


def test_localhost_fallback_port_is_allowed_by_cors(client: TestClient) -> None:
    response = client.options(
        "/api/problems",
        headers={
            "Origin": "http://127.0.0.1:5174",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"


def test_problem_detail_hides_private_fields(client: TestClient) -> None:
    response = client.get("/api/problems/matrix-transpose")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "矩阵转置"
    assert "hidden_tests" not in data
    assert "solution_code" not in data
    assert len(data["public_tests"]) == 3
    assert data["examples"][0]["input"] == "[[1, 2], [3, 4], [5, 6]]"

    vector = client.get("/api/problems/l2-normalize-vector").json()
    assert vector["examples"][0]["input"] == "[3.0, 4.0]"

    positional = client.get("/api/problems/sinusoidal-positional-encoding").json()
    assert len(positional["presentation"]["formulas"]) == 2
    assert "10000" in positional["presentation"]["formulas"][0]["latex"]
    assert "torch.Tensor" in positional["function_signature"]
    assert "numpy" not in positional["starter_code"]


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


def test_run_without_custom_tests_uses_public_cases(client: TestClient) -> None:
    pid = problem_id(client, "matrix-transpose")
    code = "def matrix_transpose(matrix):\n    return matrix\n"
    response = client.post("/api/run", json={"problem_id": pid, "code": code})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "答案错误"
    assert result["total_tests"] == 3
    assert result["first_error"]["actual_output"] is not None


def test_raw_custom_input_is_wrapped_as_single_argument(client: TestClient) -> None:
    pid = problem_id(client, "matrix-transpose")
    code = "def matrix_transpose(matrix):\n    return [list(col) for col in zip(*matrix)] if matrix else []\n"
    response = client.post(
        "/api/run",
        json={"problem_id": pid, "code": code, "custom_tests": [[[1, 2], [3, 4], [5, 6]]]},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "通过"
    assert result["total_tests"] == 1
    assert result["results"][0]["actual_output"] == [[1, 3, 5], [2, 4, 6]]


def test_custom_input_without_expected_uses_solution_output(client: TestClient) -> None:
    pid = problem_id(client, "matrix-transpose")
    code = "def matrix_transpose(matrix):\n    return matrix\n"
    response = client.post(
        "/api/run",
        json={"problem_id": pid, "code": code, "custom_tests": [[[1, 2], [3, 4], [5, 6]]]},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "答案错误"
    assert result["total_tests"] == 1
    assert result["first_error"]["expected_output"] == [[1, 3, 5], [2, 4, 6]]
    assert result["first_error"]["actual_output"] == [[1, 2], [3, 4], [5, 6]]


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
    code = "import numpy as np\n\ndef broadcast_add(a, b):\n    return np.asarray(a) + np.asarray(b)\n"
    response = client.post("/api/submit", json={"problem_id": problem["id"], "code": code})
    assert response.status_code == 200
    assert response.json()["status"] == "通过"


def test_torch_tensor_comparison(client: TestClient) -> None:
    problem = seed_problem("torch-add-relu")
    response = client.post("/api/submit", json={"problem_id": problem["id"], "code": problem["solution_code"]})
    assert response.status_code == 200
    assert response.json()["status"] == "通过"


def test_empty_tensor_shape_round_trip(client: TestClient) -> None:
    problem = seed_problem("sinusoidal-positional-encoding")
    response = client.post("/api/submit", json={"problem_id": problem["id"], "code": problem["solution_code"]})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "通过"
    assert result["passed_tests"] == 8


def test_draft_save_and_restore(client: TestClient) -> None:
    pid = problem_id(client, "matrix-transpose")
    code = "def matrix_transpose(matrix):\n    return matrix\n"
    save = client.put(f"/api/drafts/{pid}", json={"code": code})
    assert save.status_code == 200
    restored = client.get(f"/api/drafts/{pid}")
    assert restored.status_code == 200
    assert restored.json()["code"] == code
