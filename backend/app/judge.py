from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from .models import Problem


STATUS_TEXT = {
    "ACCEPTED": "通过",
    "WRONG_ANSWER": "答案错误",
    "RUNTIME_ERROR": "运行错误",
    "SYNTAX_ERROR": "语法错误",
    "TIME_LIMIT_EXCEEDED": "超出时间限制",
    "MEMORY_LIMIT_EXCEEDED": "内存超出限制",
    "OUTPUT_FORMAT_ERROR": "输出格式错误",
    "SYSTEM_ERROR": "系统错误",
}


RUNNER_CODE = r"""
import contextlib
import io
import json
import math
import numbers
import os
import sys
import time
import traceback
from typing import Any

ATOL = 1e-6
RTOL = 1e-5
MAX_OUTPUT = 12000


class LimitedWriter(io.StringIO):
    def __init__(self, limit: int = MAX_OUTPUT):
        super().__init__()
        self.limit = limit
        self.truncated = False

    def write(self, s: str) -> int:
        current = self.tell()
        remaining = self.limit - current
        if remaining <= 0:
            self.truncated = True
            return len(s)
        if len(s) > remaining:
            super().write(s[:remaining])
            super().write("\n... 输出过长，已截断 ...")
            self.truncated = True
            return len(s)
        return super().write(s)


def get_np():
    import numpy as np

    return np


def get_torch():
    import torch

    return torch


def loaded_np():
    return sys.modules.get("numpy")


def loaded_torch():
    return sys.modules.get("torch")


def is_numpy_array(value: Any) -> bool:
    np = loaded_np()
    return np is not None and isinstance(value, np.ndarray)


def is_torch_tensor(value: Any) -> bool:
    torch = loaded_torch()
    return torch is not None and isinstance(value, torch.Tensor)


def materialize(value: Any) -> Any:
    if isinstance(value, dict) and "__type__" in value:
        if value["__type__"] == "ndarray":
            np = get_np()
            dtype = float if value.get("dtype") == "float" else int
            array = np.array(value["data"], dtype=dtype)
            return array.reshape(value["shape"]) if "shape" in value else array
        if value["__type__"] == "tensor":
            torch = get_torch()
            dtype = torch.float32 if value.get("dtype") == "float" else torch.long
            tensor = torch.tensor(value["data"], dtype=dtype)
            return tensor.reshape(value["shape"]) if "shape" in value else tensor
        if value["__type__"] == "nan":
            return float("nan")
    if isinstance(value, list):
        return [materialize(item) for item in value]
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    return value


def serialize(value: Any) -> Any:
    np = loaded_np()
    torch = loaded_torch()
    if np is not None and isinstance(value, np.ndarray):
        dtype = "float" if value.dtype.kind == "f" else "int"
        return {"__type__": "ndarray", "dtype": dtype, "shape": list(value.shape), "data": value.tolist()}
    if torch is not None and isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        dtype = "float" if detached.dtype.is_floating_point else "int"
        return {"__type__": "tensor", "dtype": dtype, "shape": list(detached.shape), "data": detached.tolist()}
    if np is not None and isinstance(value, (np.floating,)):
        return float(value)
    if np is not None and isinstance(value, (np.integer,)):
        return int(value)
    if np is not None and isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return [serialize(item) for item in value]
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def as_array(value: Any) -> Any:
    if is_torch_tensor(value):
        return value.detach().cpu().numpy()
    return get_np().asarray(value)


def is_array_expected(value: Any) -> bool:
    return is_numpy_array(value) or is_torch_tensor(value)


def compare_numbers(actual: Any, expected: Any) -> bool:
    try:
        a = float(actual)
        e = float(expected)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) or math.isnan(e):
        return math.isnan(a) and math.isnan(e)
    if math.isinf(a) or math.isinf(e):
        return a == e
    return math.isclose(a, e, rel_tol=RTOL, abs_tol=ATOL)


def compare(actual: Any, expected: Any) -> bool:
    if is_array_expected(expected) or is_numpy_array(actual) or is_torch_tensor(actual):
        try:
            np = get_np()
            a = as_array(actual)
            e = as_array(expected)
            if a.shape != e.shape:
                return False
            if a.dtype.kind in {"U", "S", "O"} or e.dtype.kind in {"U", "S", "O"}:
                return a.tolist() == e.tolist()
            return bool(np.allclose(a.astype(float), e.astype(float), rtol=RTOL, atol=ATOL, equal_nan=True))
        except Exception:
            return False
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(map(str, actual.keys())) != set(map(str, expected.keys())):
            return False
        return all(compare(actual[key], expected[key]) for key in expected)
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(compare(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, numbers.Number):
        return compare_numbers(actual, expected)
    return actual == expected


def filtered_traceback(exc_type, exc, tb) -> str:
    raw = traceback.format_exception(exc_type, exc, tb)
    filtered = []
    for line in raw:
        if "runner.py" in line and "user_code.py" not in line:
            continue
        filtered.append(line.replace(os.getcwd() + os.sep, ""))
    return "".join(filtered).strip()


def result_payload(status_code: str, status: str, results: list[dict[str, Any]], first_error=None, total_tests=None) -> dict[str, Any]:
    return {
        "status_code": status_code,
        "status": status,
        "results": results,
        "passed_tests": sum(1 for item in results if item.get("passed")),
        "total_tests": total_tests if total_tests is not None else len(results),
        "runtime_ms": round(sum(item.get("runtime_ms", 0) for item in results), 3),
        "first_error": first_error,
    }


def main() -> None:
    payload_path = sys.argv[1]
    output_path = sys.argv[2]
    payload = json.loads(open(payload_path, "r", encoding="utf-8").read())
    code = payload["code"]
    function_name = payload["function_name"]
    tests = payload["tests"]
    status_text = payload["status_text"]
    try:
        import resource

        memory_limit = int(payload.get("memory_limit", 256)) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    except Exception:
        pass

    setup_stdout = LimitedWriter()
    setup_stderr = LimitedWriter()
    namespace: dict[str, Any] = {}
    try:
        compiled = compile(code, "user_code.py", "exec")
    except SyntaxError as exc:
        item = {
            "passed": False,
            "input": None,
            "expected_output": None,
            "actual_output": None,
            "stdout": "",
            "stderr": "",
            "runtime_ms": 0,
            "error_type": "SyntaxError",
            "error_message": str(exc),
            "traceback": filtered_traceback(type(exc), exc, exc.__traceback__),
        }
        payload = result_payload("SYNTAX_ERROR", status_text["SYNTAX_ERROR"], [item], item, len(tests))
        open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, allow_nan=True))
        return
    except Exception as exc:
        item = {
            "passed": False,
            "input": None,
            "expected_output": None,
            "actual_output": None,
            "stdout": "",
            "stderr": "",
            "runtime_ms": 0,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": filtered_traceback(type(exc), exc, exc.__traceback__),
        }
        payload = result_payload("SYSTEM_ERROR", status_text["SYSTEM_ERROR"], [item], item, len(tests))
        open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, allow_nan=True))
        return

    try:
        with contextlib.redirect_stdout(setup_stdout), contextlib.redirect_stderr(setup_stderr):
            exec(compiled, namespace)
        target = namespace.get(function_name)
        if target is None:
            raise NameError(f"未找到函数或入口：{function_name}")
    except MemoryError as exc:
        item = {
            "passed": False,
            "input": None,
            "expected_output": None,
            "actual_output": None,
            "stdout": setup_stdout.getvalue(),
            "stderr": setup_stderr.getvalue(),
            "runtime_ms": 0,
            "error_type": "MemoryError",
            "error_message": str(exc),
            "traceback": filtered_traceback(type(exc), exc, exc.__traceback__),
        }
        payload = result_payload("MEMORY_LIMIT_EXCEEDED", status_text["MEMORY_LIMIT_EXCEEDED"], [item], item, len(tests))
        open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, allow_nan=True))
        return
    except Exception as exc:
        item = {
            "passed": False,
            "input": None,
            "expected_output": None,
            "actual_output": None,
            "stdout": setup_stdout.getvalue(),
            "stderr": setup_stderr.getvalue(),
            "runtime_ms": 0,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": filtered_traceback(type(exc), exc, exc.__traceback__),
        }
        payload = result_payload("RUNTIME_ERROR", status_text["RUNTIME_ERROR"], [item], item, len(tests))
        open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, allow_nan=True))
        return

    results = []
    for test in tests:
        args = [materialize(arg) for arg in test.get("args", [])]
        kwargs = {key: materialize(value) for key, value in test.get("kwargs", {}).items()}
        expected_present = "expected" in test and test.get("expected") is not None
        expected = materialize(test.get("expected")) if expected_present else None
        out = LimitedWriter()
        err = LimitedWriter()
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                actual = target(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            passed = True if not expected_present else compare(actual, expected)
            item = {
                "passed": passed,
                "input": serialize({"args": args, "kwargs": kwargs}),
                "expected_output": serialize(expected) if expected_present else None,
                "actual_output": serialize(actual),
                "stdout": (setup_stdout.getvalue() + out.getvalue())[:MAX_OUTPUT],
                "stderr": (setup_stderr.getvalue() + err.getvalue())[:MAX_OUTPUT],
                "runtime_ms": round(elapsed, 3),
                "error_type": None,
                "error_message": None,
                "traceback": None,
            }
            results.append(item)
            if not passed:
                payload = result_payload("WRONG_ANSWER", status_text["WRONG_ANSWER"], results, item, len(tests))
                open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, allow_nan=True))
                return
        except MemoryError as exc:
            item = {
                "passed": False,
                "input": serialize({"args": args, "kwargs": kwargs}),
                "expected_output": serialize(expected) if expected_present else None,
                "actual_output": None,
                "stdout": (setup_stdout.getvalue() + out.getvalue())[:MAX_OUTPUT],
                "stderr": (setup_stderr.getvalue() + err.getvalue())[:MAX_OUTPUT],
                "runtime_ms": round((time.perf_counter() - start) * 1000, 3),
                "error_type": "MemoryError",
                "error_message": str(exc),
                "traceback": filtered_traceback(type(exc), exc, exc.__traceback__),
            }
            results.append(item)
            payload = result_payload("MEMORY_LIMIT_EXCEEDED", status_text["MEMORY_LIMIT_EXCEEDED"], results, item, len(tests))
            open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, allow_nan=True))
            return
        except Exception as exc:
            item = {
                "passed": False,
                "input": serialize({"args": args, "kwargs": kwargs}),
                "expected_output": serialize(expected) if expected_present else None,
                "actual_output": None,
                "stdout": (setup_stdout.getvalue() + out.getvalue())[:MAX_OUTPUT],
                "stderr": (setup_stderr.getvalue() + err.getvalue())[:MAX_OUTPUT],
                "runtime_ms": round((time.perf_counter() - start) * 1000, 3),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": filtered_traceback(type(exc), exc, exc.__traceback__),
            }
            results.append(item)
            payload = result_payload("RUNTIME_ERROR", status_text["RUNTIME_ERROR"], results, item, len(tests))
            open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, allow_nan=True))
            return

    payload = result_payload("ACCEPTED", status_text["ACCEPTED"], results, None, len(tests))
    open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
"""


def normalize_custom_tests(custom_tests: list[Any] | None) -> list[dict[str, Any]]:
    if not custom_tests:
        return []
    normalized = []
    for item in custom_tests[:20]:
        if not isinstance(item, dict):
            normalized.append({"args": [item], "kwargs": {}})
            continue
        if "args" not in item:
            if "input" in item:
                normalized.append(
                    {
                        "args": [item["input"]],
                        "kwargs": item.get("kwargs", {}),
                        **({"expected": item["expected"]} if "expected" in item else {}),
                    }
                )
            else:
                normalized.append({"args": [item], "kwargs": {}})
            continue
        args = item.get("args", [])
        if not isinstance(args, list):
            args = [args]
        normalized.append(
            {
                "args": args,
                "kwargs": item.get("kwargs", {}),
                **({"expected": item["expected"]} if "expected" in item else {}),
            }
        )
    return normalized


def judge_code(problem: Problem, code: str, tests: list[dict[str, Any]], timeout_multiplier: float = 1.0) -> dict[str, Any]:
    if not tests:
        return {
            "status": STATUS_TEXT["SYSTEM_ERROR"],
            "status_code": "SYSTEM_ERROR",
            "passed_tests": 0,
            "total_tests": 0,
            "runtime_ms": 0,
            "results": [],
            "first_error": {"error_message": "没有可执行的测试用例"},
        }

    total_timeout = min(max(problem.time_limit * timeout_multiplier * max(len(tests), 1), 1.0), 8.0)
    with tempfile.TemporaryDirectory(prefix="dlcode_judge_") as temp_dir:
        temp = Path(temp_dir)
        runner_path = temp / "runner.py"
        payload_path = temp / "payload.json"
        output_path = temp / "result.json"
        runner_path.write_text(textwrap.dedent(RUNNER_CODE), encoding="utf-8")
        payload = {
            "code": code,
            "function_name": problem.function_name,
            "tests": tests[:80],
            "status_text": STATUS_TEXT,
            "memory_limit": problem.memory_limit,
        }
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=True), encoding="utf-8")

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), str(payload_path), str(output_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=total_timeout,
            )
        except subprocess.TimeoutExpired:
            runtime_ms = round((time.perf_counter() - started) * 1000, 3)
            item = {
                "passed": False,
                "input": None,
                "expected_output": None,
                "actual_output": None,
                "stdout": "",
                "stderr": "",
                "runtime_ms": runtime_ms,
                "error_type": "TimeoutError",
                "error_message": f"代码运行超过 {total_timeout:.1f} 秒，已终止。",
                "traceback": "用户代码执行超时，可能存在无限循环或计算量过大。",
            }
            return {
                "status": STATUS_TEXT["TIME_LIMIT_EXCEEDED"],
                "status_code": "TIME_LIMIT_EXCEEDED",
                "passed_tests": 0,
                "total_tests": len(tests),
                "runtime_ms": runtime_ms,
                "results": [item],
                "first_error": item,
            }

        if output_path.exists():
            try:
                return json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return _system_error(f"判题结果解析失败：{exc}", completed.stdout, completed.stderr, len(tests))

        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "判题子进程异常退出"
            return _system_error(message[:2000], completed.stdout, completed.stderr, len(tests))
        return _system_error("判题子进程没有生成结果", completed.stdout, completed.stderr, len(tests))


def _system_error(message: str, stdout: str, stderr: str, total_tests: int) -> dict[str, Any]:
    item = {
        "passed": False,
        "input": None,
        "expected_output": None,
        "actual_output": None,
        "stdout": stdout[:12000],
        "stderr": stderr[:12000],
        "runtime_ms": 0,
        "error_type": "SystemError",
        "error_message": message,
        "traceback": None,
    }
    return {
        "status": STATUS_TEXT["SYSTEM_ERROR"],
        "status_code": "SYSTEM_ERROR",
        "passed_tests": 0,
        "total_tests": total_tests,
        "runtime_ms": 0,
        "results": [item],
        "first_error": item,
    }
