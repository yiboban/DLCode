from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Callable

import numpy as np
import torch


COMPANY_TAGS = [
    "字节跳动",
    "阿里巴巴",
    "腾讯",
    "百度",
    "美团",
    "快手",
    "京东",
    "华为",
    "小红书",
    "拼多多",
    "滴滴",
    "Google",
    "Meta",
    "Amazon",
    "Microsoft",
    "NVIDIA",
    "OpenAI",
]


def ndarray(data: Any, dtype: str = "float") -> dict[str, Any]:
    return {"__type__": "ndarray", "dtype": dtype, "data": data}


def tensor(data: Any, dtype: str = "float") -> dict[str, Any]:
    return {"__type__": "tensor", "dtype": dtype, "data": data}


def materialize(value: Any) -> Any:
    if isinstance(value, dict) and "__type__" in value:
        if value["__type__"] == "ndarray":
            dtype = float if value.get("dtype") == "float" else int
            return np.array(value["data"], dtype=dtype)
        if value["__type__"] == "tensor":
            dtype = torch.float32 if value.get("dtype") == "float" else torch.long
            return torch.tensor(value["data"], dtype=dtype)
        if value["__type__"] == "nan":
            return float("nan")
    if isinstance(value, list):
        return [materialize(item) for item in value]
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ndarray(value.tolist(), "float" if value.dtype.kind == "f" else "int")
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        dtype = "float" if detached.dtype.is_floating_point else "int"
        return tensor(detached.tolist(), dtype)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def case(args: list[Any], expected: Any | None = None, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"args": args, "kwargs": kwargs or {}, "expected": expected}


def build_cases(reference: Callable[..., Any], raw_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    built = []
    for raw in raw_cases:
        args = raw["args"]
        kwargs = raw.get("kwargs", {})
        if raw.get("expected") is None:
            expected = reference(*[materialize(arg) for arg in args], **{k: materialize(v) for k, v in kwargs.items()})
        else:
            expected = raw["expected"]
        built.append(
            {
                "args": to_jsonable(args),
                "kwargs": to_jsonable(kwargs),
                "expected": to_jsonable(expected),
            }
        )
    return built


def starter(signature: str, imports: str = "", body: str = "    pass") -> str:
    prefix = imports.rstrip() + "\n\n" if imports else ""
    return f"{prefix}{signature}:\n{body}\n"


def format_value(value: Any) -> str:
    value = to_jsonable(value)
    if isinstance(value, dict) and value.get("__type__") in {"ndarray", "tensor"}:
        return f"{value['__type__']}({value['data']})"
    return repr(value)


def make_problem(
    *,
    pid: int,
    slug: str,
    title: str,
    difficulty: str,
    category: str,
    function_name: str,
    signature: str,
    description: str,
    reference: Callable[..., Any],
    raw_cases: list[dict[str, Any]],
    solution_code: str,
    explanation: str,
    constraints: list[str],
    imports: str = "",
    starter_code: str | None = None,
    time_limit: float = 2.0,
    memory_limit: int = 256,
) -> dict[str, Any]:
    tests = build_cases(reference, raw_cases)
    public_tests = tests[:3]
    hidden_tests = tests[3:]
    examples = [
        {
            "input": format_value(raw_cases[i]["args"]),
            "output": format_value(tests[i]["expected"]),
            "explanation": "覆盖常见输入或边界条件。",
        }
        for i in range(min(2, len(public_tests)))
    ]
    return {
        "id": pid,
        "slug": slug,
        "title": title,
        "difficulty": difficulty,
        "categories": [category],
        "company_tags": [COMPANY_TAGS[pid % len(COMPANY_TAGS)], COMPANY_TAGS[(pid * 3) % len(COMPANY_TAGS)]],
        "source_note": "常见公开面试知识点改编题，题面与测试数据为 DLCode 原创整理；公司标签仅表示相似高频方向。",
        "description": description,
        "function_name": function_name,
        "function_signature": signature,
        "starter_code": starter_code or starter(signature, imports),
        "solution_code": solution_code,
        "explanation": explanation,
        "constraints": constraints,
        "examples": examples,
        "public_tests": public_tests,
        "hidden_tests": hidden_tests,
        "time_limit": time_limit,
        "memory_limit": memory_limit,
    }


def py_imports() -> str:
    return "from typing import Any\nimport math\nimport numpy as np"


def torch_imports() -> str:
    return "from typing import Any\nimport math\nimport numpy as np\nimport torch\nfrom torch import nn"


def get_seed_problems() -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        problems.append(make_problem(pid=len(problems) + 1, **kwargs))

    add(
        slug="matrix-transpose",
        title="矩阵转置",
        difficulty="简单",
        category="Python 与 NumPy 基础",
        function_name="matrix_transpose",
        signature="def matrix_transpose(matrix: list[list[float]]) -> list[list[float]]",
        description="给定一个二维列表表示的矩阵，返回它的转置矩阵。输入矩阵可能只有一行或一列。",
        reference=lambda matrix: [list(col) for col in zip(*matrix)] if matrix else [],
        raw_cases=[
            case([[[1, 2], [3, 4], [5, 6]]]),
            case([[[1, 2, 3]]]),
            case([[[]]]),
            case([[[7], [8], [9]]]),
            case([[[1.5, -2.0], [0.0, 3.5]]]),
            case([[]]),
            case([[[1, 2, 3], [4, 5, 6]]]),
            case([[[0]]]),
        ],
        imports=py_imports(),
        solution_code="from typing import Any\n\ndef matrix_transpose(matrix):\n    return [list(col) for col in zip(*matrix)] if matrix else []\n",
        explanation="转置就是把原矩阵的列变成新矩阵的行。Python 的 zip(*matrix) 可以自然完成按列聚合。",
        constraints=["0 <= 行数 <= 200", "每行列数一致", "元素为整数或浮点数"],
    )

    add(
        slug="l2-normalize-vector",
        title="向量 L2 归一化",
        difficulty="简单",
        category="Python 与 NumPy 基础",
        function_name="l2_normalize",
        signature="def l2_normalize(vector: list[float]) -> list[float]",
        description="实现向量的 L2 归一化。若向量范数为 0，返回与输入等长的全 0 向量。",
        reference=lambda vector: [0.0 for _ in vector]
        if math.sqrt(sum(v * v for v in vector)) == 0
        else [v / math.sqrt(sum(x * x for x in vector)) for v in vector],
        raw_cases=[
            case([[3.0, 4.0]]),
            case([[0.0, 0.0, 0.0]]),
            case([[-1.0, 1.0]]),
            case([[1.0]]),
            case([[2.0, -2.0, 1.0]]),
            case([[10.0, 0.0, 0.0]]),
            case([[0.5, 0.5, 0.5, 0.5]]),
            case([[]]),
        ],
        imports=py_imports(),
        solution_code="import math\n\ndef l2_normalize(vector):\n    norm = math.sqrt(sum(v * v for v in vector))\n    return [0.0 for _ in vector] if norm == 0 else [v / norm for v in vector]\n",
        explanation="先计算平方和的平方根，再逐元素除以范数；零向量单独处理，避免除零。",
        constraints=["0 <= len(vector) <= 10000", "允许负数和浮点数", "误差容忍 1e-6"],
    )

    add(
        slug="numpy-broadcast-add",
        title="广播加法",
        difficulty="简单",
        category="Python 与 NumPy 基础",
        function_name="broadcast_add",
        signature="def broadcast_add(a: Any, b: Any) -> np.ndarray",
        description="使用 NumPy 广播机制返回 a + b 的结果。输入可以是标量、列表或二维列表。",
        reference=lambda a, b: np.asarray(a) + np.asarray(b),
        raw_cases=[
            case([[[1, 2, 3], [4, 5, 6]], [10, 20, 30]]),
            case([[1, 2, 3], 5]),
            case([[[1], [2], [3]], [10, 20]]),
            case([0, [[1, 2], [3, 4]]]),
            case([[[1.5, 2.5]], [[1.0], [2.0]]]),
            case([[[-1, -2, -3]], [1, 1, 1]]),
            case([[1], [2, 3, 4]]),
            case([[[0, 0]], 0]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef broadcast_add(a, b):\n    return np.asarray(a) + np.asarray(b)\n",
        explanation="把输入转换成 ndarray 后直接使用 +，NumPy 会按照广播规则扩展维度。",
        constraints=["输入满足 NumPy 可广播条件", "返回值必须是 np.ndarray"],
    )

    def ref_cosine(a: Any, b: Any) -> np.ndarray:
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        denom = np.linalg.norm(a_arr, axis=1) * np.linalg.norm(b_arr, axis=1)
        out = np.zeros(a_arr.shape[0], dtype=float)
        mask = denom != 0
        out[mask] = np.sum(a_arr[mask] * b_arr[mask], axis=1) / denom[mask]
        return out

    add(
        slug="batch-cosine-similarity",
        title="批量余弦相似度",
        difficulty="中等",
        category="Python 与 NumPy 基础",
        function_name="batch_cosine_similarity",
        signature="def batch_cosine_similarity(a: Any, b: Any) -> np.ndarray",
        description="给定两个形状相同的二维数组，返回每一行之间的余弦相似度。若某一行存在零向量，该行结果记为 0。",
        reference=ref_cosine,
        raw_cases=[
            case([[[1, 0], [1, 1]], [[1, 0], [1, -1]]]),
            case([[[0, 0], [2, 0]], [[1, 1], [0, 3]]]),
            case([[[1, 2, 3]], [[4, 5, 6]]]),
            case([[[-1, 0], [0, -2]], [[1, 0], [0, 2]]]),
            case([[[3, 4], [5, 12]], [[6, 8], [0, 0]]]),
            case([[[1.5, 2.5], [3.5, 4.5]], [[2, 1], [4, 3]]]),
            case([[[1, 1, 1], [1, 0, 0]], [[1, 1, 1], [0, 1, 0]]]),
            case([[[0, 1]], [[0, -1]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef batch_cosine_similarity(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)\n    out = np.zeros(a.shape[0], dtype=float)\n    mask = denom != 0\n    out[mask] = np.sum(a[mask] * b[mask], axis=1) / denom[mask]\n    return out\n",
        explanation="分子是逐行点积，分母是两组向量范数乘积；零范数行不能直接相除。",
        constraints=["a 与 b 形状一致", "二维数组行数至少为 1", "误差容忍 1e-6"],
    )

    add(
        slug="top-k-values",
        title="Top-K 最大值",
        difficulty="简单",
        category="Python 与 NumPy 基础",
        function_name="top_k",
        signature="def top_k(values: list[float], k: int) -> list[float]",
        description="返回列表中最大的 k 个数，按从大到小排列。若 k 大于列表长度，返回全部元素。",
        reference=lambda values, k: sorted(values, reverse=True)[: max(0, min(k, len(values)))],
        raw_cases=[
            case([[3, 1, 5, 2], 2]),
            case([[1, 1, 1], 5]),
            case([[-1, -3, 2, 0], 3]),
            case([[10], 1]),
            case([[4, 4, 2, 9, 9], 2]),
            case([[], 3]),
            case([[7, 6, 5], 0]),
            case([[0.1, 0.3, 0.2], 2]),
        ],
        imports=py_imports(),
        solution_code="def top_k(values, k):\n    return sorted(values, reverse=True)[:max(0, min(k, len(values)))]\n",
        explanation="排序后取前 k 个元素即可。面试中也可以进一步讨论堆实现以优化大规模数据。",
        constraints=["0 <= len(values) <= 10000", "k 可以为 0 或超过数组长度"],
    )

    def ref_softmax(logits: list[float]) -> list[float]:
        if not logits:
            return []
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        return [x / s for x in exps]

    add(
        slug="stable-softmax",
        title="稳定版 Softmax",
        difficulty="中等",
        category="Python 与 NumPy 基础",
        function_name="stable_softmax",
        signature="def stable_softmax(logits: list[float]) -> list[float]",
        description="实现数值稳定的一维 Softmax。需要通过减去最大值避免 exp 溢出。",
        reference=ref_softmax,
        raw_cases=[
            case([[1.0, 2.0, 3.0]]),
            case([[1000.0, 1000.0]]),
            case([[-1000.0, -999.0]]),
            case([[0.0]]),
            case([[2.0, -1.0, 0.0, 4.0]]),
            case([[10.0, 0.0, -10.0]]),
            case([[5.5, 5.5, 5.5]]),
            case([[]]),
        ],
        imports=py_imports(),
        solution_code="import math\n\ndef stable_softmax(logits):\n    if not logits:\n        return []\n    m = max(logits)\n    exps = [math.exp(x - m) for x in logits]\n    total = sum(exps)\n    return [x / total for x in exps]\n",
        explanation="Softmax 对整体平移不敏感，减去最大值可以避免大正数带来的指数溢出。",
        constraints=["0 <= len(logits) <= 10000", "误差容忍 1e-6"],
    )

    def ref_lse(values: list[float]) -> float:
        m = max(values)
        return m + math.log(sum(math.exp(v - m) for v in values))

    add(
        slug="log-sum-exp",
        title="LogSumExp",
        difficulty="中等",
        category="Python 与 NumPy 基础",
        function_name="log_sum_exp",
        signature="def log_sum_exp(values: list[float]) -> float",
        description="计算 log(sum(exp(values)))，要求使用数值稳定写法。",
        reference=ref_lse,
        raw_cases=[
            case([[1.0, 2.0, 3.0]]),
            case([[1000.0, 1001.0]]),
            case([[-1000.0, -1002.0]]),
            case([[0.0]]),
            case([[2.0, -1.0, 4.0, 8.0]]),
            case([[5.5, 5.5, 5.5]]),
            case([[-3.0, -2.0, -1.0]]),
            case([[20.0, 0.0, -20.0]]),
        ],
        imports=py_imports(),
        solution_code="import math\n\ndef log_sum_exp(values):\n    m = max(values)\n    return m + math.log(sum(math.exp(v - m) for v in values))\n",
        explanation="把最大值提出到 log 外部，可以避免 exp(1000) 这类溢出。",
        constraints=["len(values) >= 1", "误差容忍 1e-6"],
    )

    add(
        slug="one-hot-encoding",
        title="独热编码",
        difficulty="简单",
        category="Python 与 NumPy 基础",
        function_name="one_hot",
        signature="def one_hot(indices: list[int], num_classes: int) -> np.ndarray",
        description="把类别下标列表转换为独热编码矩阵。下标均在合法范围内。",
        reference=lambda indices, num_classes: np.eye(num_classes, dtype=int)[indices],
        raw_cases=[
            case([[0, 2, 1], 3]),
            case([[1], 4]),
            case([[], 3]),
            case([[2, 2, 0], 3]),
            case([[0, 1, 2, 3], 4]),
            case([[3, 0], 5]),
            case([[0], 1]),
            case([[4, 1, 4], 5]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef one_hot(indices, num_classes):\n    return np.eye(num_classes, dtype=int)[indices]\n",
        explanation="构造单位矩阵后按类别下标取行，是独热编码最直接的 NumPy 写法。",
        constraints=["0 <= index < num_classes", "num_classes >= 1", "返回 np.ndarray"],
    )

    add(
        slug="batch-gather",
        title="批量索引",
        difficulty="中等",
        category="Python 与 NumPy 基础",
        function_name="batch_gather",
        signature="def batch_gather(matrix: list[list[float]], indices: list[int]) -> list[float]",
        description="给定二维矩阵和与行数相同的列下标列表，返回每一行对应列的元素。",
        reference=lambda matrix, indices: [row[idx] for row, idx in zip(matrix, indices)],
        raw_cases=[
            case([[[1, 2, 3], [4, 5, 6]], [0, 2]]),
            case([[[7, 8], [9, 10], [11, 12]], [1, 0, 1]]),
            case([[[1]], [0]]),
            case([[[-1, -2, -3]], [2]]),
            case([[[0.5, 1.5], [2.5, 3.5]], [0, 1]]),
            case([[[3, 4, 5], [6, 7, 8], [9, 10, 11]], [2, 1, 0]]),
            case([[], []]),
            case([[[5, 6, 7]], [1]]),
        ],
        imports=py_imports(),
        solution_code="def batch_gather(matrix, indices):\n    return [row[idx] for row, idx in zip(matrix, indices)]\n",
        explanation="每一行使用自己的列下标，遍历行和下标即可。",
        constraints=["len(matrix) == len(indices)", "每个下标在对应行范围内"],
    )

    add(
        slug="sliding-window-sum",
        title="滑动窗口求和",
        difficulty="简单",
        category="Python 与 NumPy 基础",
        function_name="sliding_window_sum",
        signature="def sliding_window_sum(values: list[float], window: int) -> list[float]",
        description="返回长度为 window 的连续窗口和。若 window 非法或大于数组长度，返回空列表。",
        reference=lambda values, window: []
        if window <= 0 or window > len(values)
        else [sum(values[i : i + window]) for i in range(len(values) - window + 1)],
        raw_cases=[
            case([[1, 2, 3, 4], 2]),
            case([[1, 2, 3], 3]),
            case([[1, 2], 3]),
            case([[5], 1]),
            case([[-1, 1, -1, 1], 2]),
            case([[0, 0, 0], 1]),
            case([[1, 2, 3], 0]),
            case([[0.5, 1.5, 2.0], 2]),
        ],
        imports=py_imports(),
        solution_code="def sliding_window_sum(values, window):\n    if window <= 0 or window > len(values):\n        return []\n    return [sum(values[i:i + window]) for i in range(len(values) - window + 1)]\n",
        explanation="直接枚举每个窗口即可；可进一步用前缀和或滚动和优化。",
        constraints=["0 <= len(values) <= 10000", "window 可以非法"],
    )

    # 传统机器学习
    add(
        slug="linear-regression-predict",
        title="线性回归预测",
        difficulty="简单",
        category="传统机器学习",
        function_name="linear_regression_predict",
        signature="def linear_regression_predict(features: list[list[float]], weights: list[float], bias: float) -> list[float]",
        description="实现线性回归的预测 y = Xw + b，返回每个样本的预测值。",
        reference=lambda features, weights, bias: [sum(x * w for x, w in zip(row, weights)) + bias for row in features],
        raw_cases=[
            case([[[1, 2], [3, 4]], [0.5, 1.0], 0.0]),
            case([[[0, 0]], [1, 2], 3.0]),
            case([[[1]], [2], -1.0]),
            case([[[-1, 2], [2, -3]], [1.5, -0.5], 0.5]),
            case([[], [1, 2], 0.0]),
            case([[[5, 6, 7]], [1, 0, -1], 2.0]),
            case([[[0.1, 0.2]], [10, 20], 1.0]),
            case([[[2, 2], [1, 1]], [2, 2], -2.0]),
        ],
        imports=py_imports(),
        solution_code="def linear_regression_predict(features, weights, bias):\n    return [sum(x * w for x, w in zip(row, weights)) + bias for row in features]\n",
        explanation="逐样本做点积并加上偏置即可。",
        constraints=["特征维度与权重长度一致", "允许空样本列表"],
    )

    add(
        slug="sigmoid-probabilities",
        title="逻辑回归 Sigmoid 概率",
        difficulty="简单",
        category="传统机器学习",
        function_name="sigmoid_probs",
        signature="def sigmoid_probs(logits: list[float]) -> list[float]",
        description="把逻辑回归的 logit 转换为概率。需要兼顾较大的正负输入。",
        reference=lambda logits: [1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x)) for x in logits],
        raw_cases=[
            case([[0.0, 1.0, -1.0]]),
            case([[20.0, -20.0]]),
            case([[]]),
            case([[2.5, -3.5, 0.5]]),
            case([[100.0, -100.0]]),
            case([[10.0]]),
            case([[-0.25, 0.25]]),
            case([[5.0, 0.0, -5.0]]),
        ],
        imports=py_imports(),
        solution_code="import math\n\ndef sigmoid_probs(logits):\n    out = []\n    for x in logits:\n        if x >= 0:\n            out.append(1 / (1 + math.exp(-x)))\n        else:\n            e = math.exp(x)\n            out.append(e / (1 + e))\n    return out\n",
        explanation="负数分支使用 exp(x)/(1+exp(x))，避免 exp(-x) 对大负数溢出。",
        constraints=["0 <= len(logits) <= 10000", "误差容忍 1e-6"],
    )

    def ref_knn(train_x: list[list[float]], train_y: list[int], query: list[float], k: int) -> int:
        distances = []
        for x, y in zip(train_x, train_y):
            distances.append((sum((a - b) ** 2 for a, b in zip(x, query)), y))
        votes = Counter(y for _, y in sorted(distances)[:k])
        best_count = max(votes.values())
        return min(label for label, count in votes.items() if count == best_count)

    add(
        slug="knn-majority-vote",
        title="KNN 多数投票",
        difficulty="中等",
        category="传统机器学习",
        function_name="knn_predict",
        signature="def knn_predict(train_x: list[list[float]], train_y: list[int], query: list[float], k: int) -> int",
        description="使用欧氏距离实现 KNN 分类。若票数相同，返回标签值更小的类别。",
        reference=ref_knn,
        raw_cases=[
            case([[[0, 0], [1, 1], [5, 5]], [0, 0, 1], [0.2, 0.1], 2]),
            case([[[0], [2], [4]], [1, 2, 2], [3], 2]),
            case([[[0, 0], [0, 1]], [2, 1], [0, 0.4], 2]),
            case([[[1, 1], [2, 2], [9, 9]], [3, 3, 4], [1.5, 1.5], 1]),
            case([[[0], [10], [11], [12]], [0, 1, 1, 2], [10.5], 3]),
            case([[[0, 0]], [7], [1, 1], 1]),
            case([[[0, 0], [2, 0], [0, 2]], [1, 2, 2], [1, 1], 3]),
            case([[[1, 2], [3, 4], [5, 6]], [1, 2, 3], [4, 5], 2]),
        ],
        imports=py_imports(),
        solution_code="from collections import Counter\n\ndef knn_predict(train_x, train_y, query, k):\n    distances = []\n    for x, y in zip(train_x, train_y):\n        distances.append((sum((a - b) ** 2 for a, b in zip(x, query)), y))\n    votes = Counter(y for _, y in sorted(distances)[:k])\n    best = max(votes.values())\n    return min(label for label, count in votes.items() if count == best)\n",
        explanation="计算查询点到训练样本的距离，取最近 k 个标签投票；平票时按题意选择较小标签。",
        constraints=["1 <= k <= len(train_x)", "train_x 与 train_y 长度一致"],
    )

    add(
        slug="kmeans-assign",
        title="K-Means 簇分配",
        difficulty="中等",
        category="传统机器学习",
        function_name="kmeans_assign",
        signature="def kmeans_assign(points: list[list[float]], centers: list[list[float]]) -> list[int]",
        description="给定样本点和聚类中心，返回每个样本最近中心的下标。距离相同时选择较小下标。",
        reference=lambda points, centers: [
            min(range(len(centers)), key=lambda i: (sum((a - b) ** 2 for a, b in zip(point, centers[i])), i))
            for point in points
        ],
        raw_cases=[
            case([[[0, 0], [10, 10]], [[0, 1], [9, 9]]]),
            case([[[1], [5], [9]], [[0], [10]]]),
            case([[[0, 0]], [[1, 0], [0, 1]]]),
            case([[[2, 2], [3, 3]], [[0, 0], [4, 4]]]),
            case([[], [[0, 0]]]),
            case([[[1.5, 2.5]], [[1, 2], [3, 4]]]),
            case([[[-1, -1], [1, 1]], [[-2, -2], [2, 2]]]),
            case([[[0, 0], [0, 2], [2, 0]], [[0, 0], [2, 2]]]),
        ],
        imports=py_imports(),
        solution_code="def kmeans_assign(points, centers):\n    return [min(range(len(centers)), key=lambda i: (sum((a - b) ** 2 for a, b in zip(point, centers[i])), i)) for point in points]\n",
        explanation="K-Means 的分配步骤就是为每个点寻找最近中心。",
        constraints=["centers 非空", "点和中心维度一致"],
    )

    def ref_entropy(labels: list[Any]) -> float:
        total = len(labels)
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log2(c / total) for c in Counter(labels).values())

    add(
        slug="information-entropy",
        title="信息熵",
        difficulty="简单",
        category="传统机器学习",
        function_name="entropy",
        signature="def entropy(labels: list[int]) -> float",
        description="计算类别标签的信息熵，log 以 2 为底。空列表熵定义为 0。",
        reference=ref_entropy,
        raw_cases=[
            case([[0, 0, 1, 1]]),
            case([[1, 1, 1]]),
            case([[]]),
            case([[0, 1, 2, 3]]),
            case([[0, 0, 0, 1]]),
            case([[1, 2, 2, 2, 3, 3]]),
            case([[5]]),
            case([[0, 1, 1, 1, 1]]),
        ],
        imports=py_imports(),
        solution_code="import math\nfrom collections import Counter\n\ndef entropy(labels):\n    total = len(labels)\n    if total == 0:\n        return 0.0\n    return -sum((c / total) * math.log2(c / total) for c in Counter(labels).values())\n",
        explanation="统计每个类别概率后套用 -sum(p log2 p)。",
        constraints=["标签可以是整数", "误差容忍 1e-6"],
    )

    def ref_gini(labels: list[Any]) -> float:
        total = len(labels)
        if total == 0:
            return 0.0
        return 1 - sum((c / total) ** 2 for c in Counter(labels).values())

    add(
        slug="gini-index",
        title="基尼系数",
        difficulty="简单",
        category="传统机器学习",
        function_name="gini",
        signature="def gini(labels: list[int]) -> float",
        description="计算分类标签的 Gini impurity。空列表基尼系数定义为 0。",
        reference=ref_gini,
        raw_cases=[
            case([[0, 0, 1, 1]]),
            case([[1, 1, 1]]),
            case([[]]),
            case([[0, 1, 2, 3]]),
            case([[0, 0, 0, 1]]),
            case([[1, 2, 2, 2, 3, 3]]),
            case([[5]]),
            case([[0, 1, 1, 1, 1]]),
        ],
        imports=py_imports(),
        solution_code="from collections import Counter\n\ndef gini(labels):\n    total = len(labels)\n    if total == 0:\n        return 0.0\n    return 1 - sum((c / total) ** 2 for c in Counter(labels).values())\n",
        explanation="Gini impurity 为 1 减去各类别概率平方和。",
        constraints=["标签可以是整数", "误差容忍 1e-6"],
    )

    def ref_standardize(values: list[float]) -> list[float]:
        if not values:
            return []
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        return [0.0 for _ in values] if std == 0 else [(v - mean) / std for v in values]

    add(
        slug="standardize-feature",
        title="特征标准化",
        difficulty="简单",
        category="传统机器学习",
        function_name="standardize_column",
        signature="def standardize_column(values: list[float]) -> list[float]",
        description="对一列特征做 z-score 标准化。若标准差为 0，返回全 0。",
        reference=ref_standardize,
        raw_cases=[
            case([[1, 2, 3]]),
            case([[5, 5, 5]]),
            case([[]]),
            case([[-1, 0, 1]]),
            case([[10, 20, 30, 40]]),
            case([[0.5, 1.5]]),
            case([[100]]),
            case([[-3, -3, 0, 6]]),
        ],
        imports=py_imports(),
        solution_code="import math\n\ndef standardize_column(values):\n    if not values:\n        return []\n    mean = sum(values) / len(values)\n    var = sum((v - mean) ** 2 for v in values) / len(values)\n    std = math.sqrt(var)\n    return [0.0 for _ in values] if std == 0 else [(v - mean) / std for v in values]\n",
        explanation="标准化使用总体标准差；常量列不能除以 0。",
        constraints=["0 <= len(values) <= 10000", "误差容忍 1e-6"],
    )

    add(
        slug="binary-confusion-matrix",
        title="二分类混淆矩阵",
        difficulty="简单",
        category="传统机器学习",
        function_name="confusion_matrix_binary",
        signature="def confusion_matrix_binary(y_true: list[int], y_pred: list[int]) -> dict[str, int]",
        description="统计二分类任务中的 TP、TN、FP、FN，正类标签为 1。",
        reference=lambda y_true, y_pred: {
            "TP": sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1),
            "TN": sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0),
            "FP": sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1),
            "FN": sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0),
        },
        raw_cases=[
            case([[1, 0, 1, 0], [1, 0, 0, 1]]),
            case([[1, 1], [1, 1]]),
            case([[0, 0], [1, 1]]),
            case([[], []]),
            case([[1, 0, 0, 1, 1], [0, 0, 0, 1, 1]]),
            case([[0], [0]]),
            case([[1], [0]]),
            case([[0, 1, 0, 1], [0, 1, 0, 1]]),
        ],
        imports=py_imports(),
        solution_code="def confusion_matrix_binary(y_true, y_pred):\n    return {\n        'TP': sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1),\n        'TN': sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0),\n        'FP': sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1),\n        'FN': sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0),\n    }\n",
        explanation="按真实标签和预测标签四种组合分别计数。",
        constraints=["y_true 与 y_pred 长度一致", "标签只包含 0 和 1"],
    )

    def ref_prf(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}

    add(
        slug="precision-recall-f1",
        title="Precision、Recall 和 F1",
        difficulty="中等",
        category="传统机器学习",
        function_name="precision_recall_f1",
        signature="def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> dict[str, float]",
        description="计算二分类的 precision、recall 和 F1。分母为 0 时对应指标记为 0。",
        reference=ref_prf,
        raw_cases=[
            case([[1, 0, 1, 0], [1, 0, 0, 1]]),
            case([[1, 1], [1, 1]]),
            case([[0, 0], [0, 0]]),
            case([[1, 1], [0, 0]]),
            case([[0, 1, 1, 1], [1, 1, 0, 1]]),
            case([[], []]),
            case([[1], [1]]),
            case([[0], [1]]),
        ],
        imports=py_imports(),
        solution_code="def precision_recall_f1(y_true, y_pred):\n    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)\n    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)\n    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)\n    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)\n    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)\n    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)\n    return {'precision': precision, 'recall': recall, 'f1': f1}\n",
        explanation="先统计 TP、FP、FN，再按定义计算三个指标。",
        constraints=["标签只包含 0 和 1", "误差容忍 1e-6"],
    )

    def ref_gd(w: float, b: float, x: list[float], y: list[float], lr: float) -> dict[str, float]:
        n = len(x)
        preds = [w * xi + b for xi in x]
        grad_w = sum(2 * (p - yi) * xi for p, yi, xi in zip(preds, y, x)) / n
        grad_b = sum(2 * (p - yi) for p, yi in zip(preds, y)) / n
        return {"w": w - lr * grad_w, "b": b - lr * grad_b}

    add(
        slug="gradient-descent-step",
        title="均方误差梯度下降一步",
        difficulty="中等",
        category="传统机器学习",
        function_name="gradient_descent_step",
        signature="def gradient_descent_step(w: float, b: float, x: list[float], y: list[float], lr: float) -> dict[str, float]",
        description="对一元线性模型 y_hat = w*x + b 的 MSE 损失执行一步批量梯度下降，返回更新后的 w 和 b。",
        reference=ref_gd,
        raw_cases=[
            case([0.0, 0.0, [1, 2], [2, 4], 0.1]),
            case([1.0, 0.0, [1, 2, 3], [1, 2, 3], 0.01]),
            case([2.0, 1.0, [0, 1], [1, 3], 0.1]),
            case([-1.0, 0.5, [1, -1], [0, 2], 0.05]),
            case([0.5, -0.5, [2, 4, 6], [1, 2, 3], 0.02]),
            case([1.5, 1.0, [1], [4], 0.1]),
            case([0.0, 1.0, [0, 0], [1, 2], 0.1]),
            case([3.0, -1.0, [-2, 2], [-7, 5], 0.01]),
        ],
        imports=py_imports(),
        solution_code="def gradient_descent_step(w, b, x, y, lr):\n    n = len(x)\n    preds = [w * xi + b for xi in x]\n    grad_w = sum(2 * (p - yi) * xi for p, yi, xi in zip(preds, y, x)) / n\n    grad_b = sum(2 * (p - yi) for p, yi in zip(preds, y)) / n\n    return {'w': w - lr * grad_w, 'b': b - lr * grad_b}\n",
        explanation="MSE 对 w 的梯度为 2/n * sum((pred-y)*x)，对 b 的梯度为 2/n * sum(pred-y)。",
        constraints=["len(x) == len(y) >= 1", "误差容忍 1e-6"],
    )

    # 深度学习基础
    add(
        slug="relu-activation",
        title="ReLU 激活函数",
        difficulty="简单",
        category="深度学习基础",
        function_name="relu",
        signature="def relu(x: Any) -> np.ndarray",
        description="实现 ReLU：逐元素返回 max(x, 0)。输入可以是一维或二维列表。",
        reference=lambda x: np.maximum(np.asarray(x, dtype=float), 0),
        raw_cases=[
            case([[-1, 0, 2]]),
            case([[[1, -2], [3, -4]]]),
            case([[0]]),
            case([[-5, -1]]),
            case([[1.5, -0.5, 2.5]]),
            case([[[0, 0], [0, 1]]]),
            case([[10]]),
            case([[[-1.0], [2.0]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef relu(x):\n    return np.maximum(np.asarray(x, dtype=float), 0)\n",
        explanation="ReLU 会截断负数并保留非负数。",
        constraints=["返回 np.ndarray", "支持任意可转成数组的输入"],
    )

    add(
        slug="sigmoid-activation",
        title="Sigmoid 激活函数",
        difficulty="简单",
        category="深度学习基础",
        function_name="sigmoid",
        signature="def sigmoid(x: Any) -> np.ndarray",
        description="实现逐元素 Sigmoid 激活函数，返回 NumPy 数组。",
        reference=lambda x: 1 / (1 + np.exp(-np.asarray(x, dtype=float))),
        raw_cases=[
            case([[-1, 0, 1]]),
            case([[[1, -2], [3, -4]]]),
            case([[0]]),
            case([[2.5, -3.5]]),
            case([[10, -10]]),
            case([[[0, 0], [1, -1]]]),
            case([[5]]),
            case([[[-1.0], [2.0]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef sigmoid(x):\n    arr = np.asarray(x, dtype=float)\n    return 1 / (1 + np.exp(-arr))\n",
        explanation="Sigmoid 将实数映射到 0 到 1 之间。",
        constraints=["返回 np.ndarray", "误差容忍 1e-6"],
    )

    def ref_softmax_2d(logits: Any) -> np.ndarray:
        arr = np.asarray(logits, dtype=float)
        shifted = arr - np.max(arr, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    add(
        slug="row-wise-softmax",
        title="二维 Softmax",
        difficulty="中等",
        category="深度学习基础",
        function_name="softmax_2d",
        signature="def softmax_2d(logits: Any) -> np.ndarray",
        description="对二维 logits 的每一行分别计算稳定版 Softmax。",
        reference=ref_softmax_2d,
        raw_cases=[
            case([[[1, 2, 3], [1, 1, 1]]]),
            case([[[1000, 1000], [-1000, -999]]]),
            case([[[0]]]),
            case([[[2, -1, 4]]]),
            case([[[-1, -2], [3, 0]]]),
            case([[[5.5, 5.5, 5.5]]]),
            case([[[10, 0, -10], [0, 0, 0]]]),
            case([[[1, 2]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef softmax_2d(logits):\n    arr = np.asarray(logits, dtype=float)\n    shifted = arr - np.max(arr, axis=1, keepdims=True)\n    exps = np.exp(shifted)\n    return exps / np.sum(exps, axis=1, keepdims=True)\n",
        explanation="行级最大值需要 keepdims=True，方便广播回原矩阵形状。",
        constraints=["输入为二维数组", "误差容忍 1e-6"],
    )

    def ref_ce(logits: Any, labels: list[int]) -> float:
        probs = ref_softmax_2d(logits)
        return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12)))

    add(
        slug="cross-entropy-loss",
        title="多分类交叉熵",
        difficulty="中等",
        category="深度学习基础",
        function_name="cross_entropy_loss",
        signature="def cross_entropy_loss(logits: Any, labels: list[int]) -> float",
        description="给定二维 logits 和每个样本的类别下标，返回平均交叉熵损失。",
        reference=ref_ce,
        raw_cases=[
            case([[[2, 1, 0], [0, 1, 2]], [0, 2]]),
            case([[[1, 1]], [0]]),
            case([[[10, 0], [0, 10]], [0, 1]]),
            case([[[-1, 2, 0]], [1]]),
            case([[[0, 0, 0], [3, 1, -1]], [2, 0]]),
            case([[[5, -5]], [1]]),
            case([[[1, 2], [3, 4], [5, 6]], [1, 1, 0]]),
            case([[[1000, 999]], [0]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef cross_entropy_loss(logits, labels):\n    arr = np.asarray(logits, dtype=float)\n    shifted = arr - np.max(arr, axis=1, keepdims=True)\n    exps = np.exp(shifted)\n    probs = exps / np.sum(exps, axis=1, keepdims=True)\n    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12)))\n",
        explanation="交叉熵只取真实类别的概率，先做稳定 softmax 再求负对数平均。",
        constraints=["logits 为二维数组", "labels 长度等于 batch size", "误差容忍 1e-6"],
    )

    add(
        slug="mean-squared-error",
        title="均方误差",
        difficulty="简单",
        category="深度学习基础",
        function_name="mse_loss",
        signature="def mse_loss(y_pred: Any, y_true: Any) -> float",
        description="计算预测值和真实值之间的平均平方误差。",
        reference=lambda y_pred, y_true: float(np.mean((np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2)),
        raw_cases=[
            case([[1, 2, 3], [1, 2, 4]]),
            case([[0], [1]]),
            case([[[1, 2], [3, 4]], [[1, 1], [3, 5]]]),
            case([[1.5, 2.5], [1.0, 3.0]]),
            case([[0, 0], [0, 0]]),
            case([[-1, 1], [1, -1]]),
            case([[10], [7]]),
            case([[[0.1, 0.2]], [[0.1, 0.4]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef mse_loss(y_pred, y_true):\n    return float(np.mean((np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))\n",
        explanation="MSE 是所有元素平方误差的平均值。",
        constraints=["y_pred 与 y_true 可广播到相同形状", "误差容忍 1e-6"],
    )

    add(
        slug="dropout-forward-train",
        title="训练模式 Dropout",
        difficulty="中等",
        category="深度学习基础",
        function_name="dropout_train",
        signature="def dropout_train(x: Any, p: float, mask: Any) -> np.ndarray",
        description="实现训练模式下的 inverted dropout：输出 x * mask / (1-p)。mask 由题目给定，避免随机性。",
        reference=lambda x, p, mask: np.asarray(x, dtype=float) * np.asarray(mask, dtype=float) / (1 - p),
        raw_cases=[
            case([[1, 2, 3], 0.5, [1, 0, 1]]),
            case([[[1, 2], [3, 4]], 0.25, [[1, 1], [0, 1]]]),
            case([[0, 0], 0.5, [1, 0]]),
            case([[10], 0.2, [1]]),
            case([[-1, 1], 0.5, [0, 1]]),
            case([[[1.5, 2.5]], 0.1, [[1, 0]]]),
            case([[5, 6, 7], 0.75, [1, 1, 0]]),
            case([[1], 0.5, [0]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef dropout_train(x, p, mask):\n    return np.asarray(x, dtype=float) * np.asarray(mask, dtype=float) / (1 - p)\n",
        explanation="inverted dropout 在训练时缩放保留的激活，使推理时无需额外缩放。",
        constraints=["0 <= p < 1", "mask 与 x 形状一致或可广播"],
    )

    def ref_batch_norm(x: Any, gamma: Any, beta: Any, eps: float = 1e-5) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        mean = arr.mean(axis=0, keepdims=True)
        var = arr.var(axis=0, keepdims=True)
        return (arr - mean) / np.sqrt(var + eps) * np.asarray(gamma) + np.asarray(beta)

    add(
        slug="batch-norm-forward",
        title="BatchNorm 前向计算",
        difficulty="中等",
        category="深度学习基础",
        function_name="batch_norm_forward",
        signature="def batch_norm_forward(x: Any, gamma: Any, beta: Any, eps: float = 1e-5) -> np.ndarray",
        description="对二维输入按特征维度执行 BatchNorm 前向计算，使用当前 batch 的均值和方差。",
        reference=ref_batch_norm,
        raw_cases=[
            case([[[1, 2], [3, 4]], [1, 1], [0, 0]]),
            case([[[1, 1], [1, 1]], [1, 1], [0, 0]]),
            case([[[1, 2, 3]], [1, 1, 1], [0, 0, 0]]),
            case([[[0, 2], [2, 4], [4, 6]], [0.5, 2.0], [1, -1]]),
            case([[[-1, 1], [1, -1]], [1, 1], [0, 0]]),
            case([[[1.5, 2.5], [3.5, 4.5]], [1, 2], [0.5, -0.5]]),
            case([[[10], [20], [30]], [1], [0]]),
            case([[[0, 0], [0, 1]], [1, 1], [0, 0], 1e-3]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef batch_norm_forward(x, gamma, beta, eps=1e-5):\n    arr = np.asarray(x, dtype=float)\n    mean = arr.mean(axis=0, keepdims=True)\n    var = arr.var(axis=0, keepdims=True)\n    return (arr - mean) / np.sqrt(var + eps) * np.asarray(gamma) + np.asarray(beta)\n",
        explanation="BatchNorm 在 batch 维度上统计每个特征的均值和方差，再应用缩放和平移。",
        constraints=["x 为二维数组", "gamma、beta 长度等于特征数", "误差容忍 1e-6"],
    )

    def ref_layer_norm(x: Any, gamma: Any, beta: Any, eps: float = 1e-5) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        mean = arr.mean(axis=1, keepdims=True)
        var = arr.var(axis=1, keepdims=True)
        return (arr - mean) / np.sqrt(var + eps) * np.asarray(gamma) + np.asarray(beta)

    add(
        slug="layer-norm-forward",
        title="LayerNorm 前向计算",
        difficulty="中等",
        category="深度学习基础",
        function_name="layer_norm_forward",
        signature="def layer_norm_forward(x: Any, gamma: Any, beta: Any, eps: float = 1e-5) -> np.ndarray",
        description="对二维输入按每个样本的特征维度执行 LayerNorm 前向计算。",
        reference=ref_layer_norm,
        raw_cases=[
            case([[[1, 2], [3, 4]], [1, 1], [0, 0]]),
            case([[[1, 1], [2, 2]], [1, 1], [0, 0]]),
            case([[[1, 2, 3]], [1, 1, 1], [0, 0, 0]]),
            case([[[0, 2], [2, 4], [4, 6]], [0.5, 2.0], [1, -1]]),
            case([[[-1, 1], [1, -1]], [1, 1], [0, 0]]),
            case([[[1.5, 2.5], [3.5, 4.5]], [1, 2], [0.5, -0.5]]),
            case([[[10, 20, 30]], [1, 1, 1], [0, 0, 0]]),
            case([[[0, 0], [0, 1]], [1, 1], [0, 0], 1e-3]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef layer_norm_forward(x, gamma, beta, eps=1e-5):\n    arr = np.asarray(x, dtype=float)\n    mean = arr.mean(axis=1, keepdims=True)\n    var = arr.var(axis=1, keepdims=True)\n    return (arr - mean) / np.sqrt(var + eps) * np.asarray(gamma) + np.asarray(beta)\n",
        explanation="LayerNorm 的统计维度在样本内部，与 batch size 无关。",
        constraints=["x 为二维数组", "gamma、beta 长度等于特征数", "误差容忍 1e-6"],
    )

    add(
        slug="conv1d-valid",
        title="一维有效卷积",
        difficulty="中等",
        category="深度学习基础",
        function_name="conv1d_valid",
        signature="def conv1d_valid(x: list[float], kernel: list[float]) -> list[float]",
        description="实现 stride=1、无 padding 的一维有效卷积。按深度学习中的互相关写法，不翻转 kernel。",
        reference=lambda x, kernel: [
            sum(x[i + j] * kernel[j] for j in range(len(kernel))) for i in range(len(x) - len(kernel) + 1)
        ],
        raw_cases=[
            case([[1, 2, 3], [1, 1]]),
            case([[1, 2, 3], [1, 0, -1]]),
            case([[5], [2]]),
            case([[0, 1, 0, 1], [1, 2]]),
            case([[-1, 2, -3, 4], [0.5, -0.5]]),
            case([[1, 1, 1], [1, 1, 1]]),
            case([[2, 4, 6, 8], [0, 1]]),
            case([[3, 2, 1], [-1]]),
        ],
        imports=py_imports(),
        solution_code="def conv1d_valid(x, kernel):\n    return [sum(x[i + j] * kernel[j] for j in range(len(kernel))) for i in range(len(x) - len(kernel) + 1)]\n",
        explanation="深度学习框架中的 Conv 通常实现互相关，本题不翻转卷积核。",
        constraints=["1 <= len(kernel) <= len(x)", "stride 固定为 1"],
    )

    add(
        slug="embedding-lookup",
        title="Embedding 查表",
        difficulty="简单",
        category="深度学习基础",
        function_name="embedding_lookup",
        signature="def embedding_lookup(embedding: Any, indices: Any) -> np.ndarray",
        description="给定 embedding 矩阵和下标，返回对应行。indices 可以是一维或二维列表。",
        reference=lambda embedding, indices: np.asarray(embedding, dtype=float)[np.asarray(indices, dtype=int)],
        raw_cases=[
            case([[[1, 2], [3, 4], [5, 6]], [0, 2]]),
            case([[[1], [2], [3]], [[0, 1], [2, 0]]]),
            case([[[0.1, 0.2]], [0]]),
            case([[[1, 0], [0, 1]], [1, 1, 0]]),
            case([[[1, 2, 3], [4, 5, 6]], [[1], [0]]]),
            case([[[9], [8], [7]], []]),
            case([[[1.5, 2.5], [3.5, 4.5]], [1]]),
            case([[[0, 0], [1, 1], [2, 2]], [[2, 1], [0, 2]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef embedding_lookup(embedding, indices):\n    return np.asarray(embedding, dtype=float)[np.asarray(indices, dtype=int)]\n",
        explanation="Embedding 本质是按 token id 从参数矩阵中取行。",
        constraints=["下标合法", "返回 np.ndarray"],
    )

    # PyTorch 基础
    add(
        slug="torch-add-relu",
        title="Tensor 加法后 ReLU",
        difficulty="简单",
        category="PyTorch 基础",
        function_name="tensor_add_relu",
        signature="def tensor_add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor",
        description="使用 PyTorch 返回 relu(a + b)，输入为形状相同或可广播的 Tensor。",
        reference=lambda a, b: torch.relu(a + b),
        raw_cases=[
            case([tensor([-1, 2]), tensor([2, -5])]),
            case([tensor([[1, -2], [3, -4]]), tensor([[0, 3], [-5, 5]])]),
            case([tensor([0]), tensor([0])]),
            case([tensor([1, 2, 3]), tensor(1)]),
            case([tensor([[-1.5, 2.5]]), tensor([[0.5, -3.0]])]),
            case([tensor([10]), tensor([-20])]),
            case([tensor([[1], [2]]), tensor([10, -10])]),
            case([tensor([-1, -2]), tensor([0, 1])]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef tensor_add_relu(a, b):\n    return torch.relu(a + b)\n",
        explanation="Tensor 支持广播，torch.relu 会逐元素截断负数。",
        constraints=["输入为 torch.Tensor", "返回 torch.Tensor"],
    )

    add(
        slug="autograd-square-grad",
        title="自动微分求平方和梯度",
        difficulty="中等",
        category="PyTorch 基础",
        function_name="autograd_square_grad",
        signature="def autograd_square_grad(values: list[float]) -> list[float]",
        description="用 PyTorch 自动微分计算 sum(x^2) 对 x 的梯度，返回 Python 列表。",
        reference=lambda values: (2 * torch.tensor(values, dtype=torch.float32)).tolist(),
        raw_cases=[
            case([[1.0, 2.0, -3.0]]),
            case([[0.0]]),
            case([[]]),
            case([[1.5, -0.5]]),
            case([[10.0, -10.0]]),
            case([[2, 4, 6]]),
            case([[-1]]),
            case([[0.25, 0.5, 0.75]]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef autograd_square_grad(values):\n    x = torch.tensor(values, dtype=torch.float32, requires_grad=True)\n    y = (x ** 2).sum()\n    y.backward()\n    return x.grad.tolist()\n",
        explanation="设置 requires_grad=True，反向传播后从 x.grad 读取梯度。",
        constraints=["返回 Python list", "误差容忍 1e-6"],
    )

    add(
        slug="torch-no-grad-update",
        title="torch.no_grad 参数更新",
        difficulty="中等",
        category="PyTorch 基础",
        function_name="no_grad_update",
        signature="def no_grad_update(param: torch.Tensor, grad: torch.Tensor, lr: float) -> torch.Tensor",
        description="模拟优化器更新：在 no_grad 语境下返回 param - lr * grad，不应把更新操作加入计算图。",
        reference=lambda param, grad, lr: param - lr * grad,
        raw_cases=[
            case([tensor([1, 2]), tensor([0.1, 0.2]), 0.1]),
            case([tensor([[1.0], [2.0]]), tensor([[1.0], [-1.0]]), 0.5]),
            case([tensor([0]), tensor([10]), 0.01]),
            case([tensor([-1, 1]), tensor([0.5, 0.5]), 0.2]),
            case([tensor([5.5]), tensor([1.5]), 1.0]),
            case([tensor([1, 2, 3]), tensor([3, 2, 1]), 0.0]),
            case([tensor([[1, 2], [3, 4]]), tensor([[0, 1], [1, 0]]), 0.1]),
            case([tensor([-10]), tensor([-2]), 0.25]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef no_grad_update(param, grad, lr):\n    with torch.no_grad():\n        return param - lr * grad\n",
        explanation="参数更新不需要梯度记录，no_grad 可以减少图构建和内存开销。",
        constraints=["输入为 Tensor", "返回更新后的 Tensor"],
    )

    def ref_accum(w: float, batches: list[list[list[float]]], lr: float) -> float:
        grad = 0.0
        total = 0
        for xs, ys in batches:
            for x, y in zip(xs, ys):
                grad += 2 * (w * x - y) * x
                total += 1
        return w - lr * grad / total

    add(
        slug="gradient-accumulation",
        title="梯度累积更新",
        difficulty="中等",
        category="PyTorch 基础",
        function_name="gradient_accumulation_step",
        signature="def gradient_accumulation_step(w: float, batches: list[list[list[float]]], lr: float) -> float",
        description="一元模型 y=w*x，给定多个小批次，按所有样本的平均 MSE 梯度累积后更新 w。",
        reference=ref_accum,
        raw_cases=[
            case([0.0, [[[1, 2], [2, 4]], [[3], [6]]], 0.1]),
            case([1.0, [[[1], [1]], [[2], [2]]], 0.01]),
            case([2.0, [[[1, 2], [0, 0]]], 0.1]),
            case([-1.0, [[[1, -1], [0, 2]]], 0.05]),
            case([0.5, [[[2, 4], [1, 2]], [[6], [3]]], 0.02]),
            case([1.5, [[[1], [4]]], 0.1]),
            case([0.0, [[[0, 0], [1, 2]]], 0.1]),
            case([3.0, [[[-2, 2], [-7, 5]]], 0.01]),
        ],
        imports=torch_imports(),
        solution_code="def gradient_accumulation_step(w, batches, lr):\n    grad = 0.0\n    total = 0\n    for xs, ys in batches:\n        for x, y in zip(xs, ys):\n            grad += 2 * (w * x - y) * x\n            total += 1\n    return w - lr * grad / total\n",
        explanation="梯度累积等价于把所有小批次样本的梯度求和后再按总样本数平均。",
        constraints=["batches 至少包含一个样本", "误差容忍 1e-6"],
    )

    add(
        slug="freeze-parameter-mask",
        title="冻结参数标记",
        difficulty="简单",
        category="PyTorch 基础",
        function_name="freeze_param_mask",
        signature="def freeze_param_mask(names: list[str], trainable_prefix: str) -> dict[str, bool]",
        description="根据参数名生成 requires_grad 标记：只有以 trainable_prefix 开头的参数可训练。",
        reference=lambda names, trainable_prefix: {name: name.startswith(trainable_prefix) for name in names},
        raw_cases=[
            case([["encoder.weight", "head.weight"], "head"]),
            case([["a", "b"], ""]),
            case([[], "layer"]),
            case([["backbone.conv", "backbone.bn", "head.fc"], "backbone"]),
            case([["layer1.w", "layer10.w"], "layer1"]),
            case([["x"], "y"]),
            case([["model.head.bias"], "model.head"]),
            case([["p", "prefix.p"], "prefix"]),
        ],
        imports=torch_imports(),
        solution_code="def freeze_param_mask(names, trainable_prefix):\n    return {name: name.startswith(trainable_prefix) for name in names}\n",
        explanation="真实项目中会把这个布尔值赋给 parameter.requires_grad。",
        constraints=["names 为参数名列表", "返回字典"],
    )

    add(
        slug="padding-mask",
        title="Padding Mask",
        difficulty="简单",
        category="PyTorch 基础",
        function_name="padding_mask",
        signature="def padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor",
        description="给定 token id Tensor，返回 bool mask，padding 位置为 True。",
        reference=lambda tokens, pad_id: tokens == pad_id,
        raw_cases=[
            case([tensor([[1, 0, 0], [2, 3, 0]], "int"), 0]),
            case([tensor([1, 2, 3], "int"), 0]),
            case([tensor([[5]], "int"), 5]),
            case([tensor([[0, 1], [0, 0]], "int"), 0]),
            case([tensor([], "int"), 0]),
            case([tensor([[7, 8]], "int"), 9]),
            case([tensor([[1, 1], [1, 2]], "int"), 1]),
            case([tensor([[-1, 0]], "int"), -1]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef padding_mask(tokens, pad_id):\n    return tokens == pad_id\n",
        explanation="Mask 通常用 True 表示需要被忽略的位置。",
        constraints=["输入为整数 Tensor", "返回 bool Tensor"],
    )

    add(
        slug="causal-mask",
        title="Causal Mask",
        difficulty="中等",
        category="PyTorch 基础",
        function_name="causal_mask",
        signature="def causal_mask(size: int) -> torch.Tensor",
        description="返回形状为 size x size 的 bool Tensor，上三角未来位置为 True，其余为 False。",
        reference=lambda size: torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1),
        raw_cases=[
            case([1]),
            case([3]),
            case([4]),
            case([0]),
            case([2]),
            case([5]),
            case([6]),
            case([7]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef causal_mask(size):\n    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)\n",
        explanation="自回归模型不能看未来 token，因此对主对角线以上位置做 mask。",
        constraints=["size >= 0", "返回 bool Tensor"],
    )

    add(
        slug="custom-mse-loss",
        title="自定义 MSE Loss",
        difficulty="简单",
        category="PyTorch 基础",
        function_name="custom_mse_loss",
        signature="def custom_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor",
        description="使用 PyTorch Tensor 实现平均平方误差，返回标量 Tensor。",
        reference=lambda pred, target: torch.mean((pred - target) ** 2),
        raw_cases=[
            case([tensor([1, 2, 3]), tensor([1, 2, 4])]),
            case([tensor([0]), tensor([1])]),
            case([tensor([[1, 2], [3, 4]]), tensor([[1, 1], [3, 5]])]),
            case([tensor([1.5, 2.5]), tensor([1.0, 3.0])]),
            case([tensor([0, 0]), tensor([0, 0])]),
            case([tensor([-1, 1]), tensor([1, -1])]),
            case([tensor([10]), tensor([7])]),
            case([tensor([[0.1, 0.2]]), tensor([[0.1, 0.4]])]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef custom_mse_loss(pred, target):\n    return torch.mean((pred - target) ** 2)\n",
        explanation="保持 Tensor 计算可以继续参与自动微分。",
        constraints=["pred 与 target 形状一致或可广播", "返回 torch.Tensor"],
    )

    add(
        slug="optimizer-step-list",
        title="简化优化器更新",
        difficulty="中等",
        category="PyTorch 基础",
        function_name="simple_optimizer_step",
        signature="def simple_optimizer_step(params: list[torch.Tensor], grads: list[torch.Tensor], lr: float) -> list[torch.Tensor]",
        description="给定参数 Tensor 列表和梯度列表，返回执行 SGD 更新后的新参数列表。",
        reference=lambda params, grads, lr: [p - lr * g for p, g in zip(params, grads)],
        raw_cases=[
            case([[tensor([1, 2])], [tensor([0.1, 0.2])], 0.1]),
            case([[tensor([1]), tensor([2])], [tensor([1]), tensor([-1])], 0.5]),
            case([[tensor([[1, 2]])], [tensor([[0, 1]])], 0.1]),
            case([[tensor([0])], [tensor([10])], 0.01]),
            case([[tensor([-1, 1])], [tensor([0.5, 0.5])], 0.2]),
            case([[tensor([5.5])], [tensor([1.5])], 1.0]),
            case([[tensor([1, 2, 3])], [tensor([3, 2, 1])], 0.0]),
            case([[tensor([-10])], [tensor([-2])], 0.25]),
        ],
        imports=torch_imports(),
        solution_code="def simple_optimizer_step(params, grads, lr):\n    return [p - lr * g for p, g in zip(params, grads)]\n",
        explanation="SGD 的核心是 param = param - lr * grad。",
        constraints=["params 与 grads 长度一致", "返回 Tensor 列表"],
    )

    dataset_starter = """from typing import Any

class TinyDataset:
    def __init__(self, values: list[Any]):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> Any:
        pass

def dataset_snapshot(values: list[Any]) -> dict[str, Any]:
    dataset = TinyDataset(values)
    if len(dataset) == 0:
        return {"length": 0, "first": None, "last": None}
    return {"length": len(dataset), "first": dataset[0], "last": dataset[len(dataset) - 1]}
"""

    add(
        slug="tiny-dataset-class",
        title="手写 Dataset 简化版本",
        difficulty="中等",
        category="PyTorch 基础",
        function_name="dataset_snapshot",
        signature="def dataset_snapshot(values: list[Any]) -> dict[str, Any]",
        description="补全 TinyDataset 类的 __init__、__len__ 和 __getitem__。辅助函数会返回数据集长度、首元素和末元素。",
        reference=lambda values: {"length": len(values), "first": None if not values else values[0], "last": None if not values else values[-1]},
        raw_cases=[
            case([[1, 2, 3]]),
            case([[]]),
            case([["a", "b"]]),
            case([[[1, 2], [3, 4]]]),
            case([[0]]),
            case([["x"]]),
            case([[{"a": 1}, {"b": 2}]]),
            case([[-1, 5, 9, 10]]),
        ],
        imports=torch_imports(),
        starter_code=dataset_starter,
        solution_code="from typing import Any\n\nclass TinyDataset:\n    def __init__(self, values: list[Any]):\n        self.values = values\n\n    def __len__(self) -> int:\n        return len(self.values)\n\n    def __getitem__(self, index: int) -> Any:\n        return self.values[index]\n\ndef dataset_snapshot(values):\n    dataset = TinyDataset(values)\n    if len(dataset) == 0:\n        return {'length': 0, 'first': None, 'last': None}\n    return {'length': len(dataset), 'first': dataset[0], 'last': dataset[len(dataset) - 1]}\n",
        explanation="Dataset 最关键的是保存样本、返回长度并支持按下标读取。",
        constraints=["values 可以为空", "__getitem__ 应按 Python 下标语义返回元素"],
    )

    # Attention 与 Transformer
    def ref_attention(q: Any, k: Any, v: Any) -> np.ndarray:
        q_arr, k_arr, v_arr = np.asarray(q, dtype=float), np.asarray(k, dtype=float), np.asarray(v, dtype=float)
        scores = q_arr @ k_arr.T / math.sqrt(q_arr.shape[-1])
        probs = ref_softmax_2d(scores)
        return probs @ v_arr

    add(
        slug="scaled-dot-product-attention",
        title="Scaled Dot-Product Attention",
        difficulty="困难",
        category="Attention 与 Transformer",
        function_name="scaled_dot_product_attention",
        signature="def scaled_dot_product_attention(q: Any, k: Any, v: Any) -> np.ndarray",
        description="实现单头 scaled dot-product attention：softmax(QK^T/sqrt(d))V。",
        reference=ref_attention,
        raw_cases=[
            case([[[1, 0]], [[1, 0], [0, 1]], [[10, 0], [0, 20]]]),
            case([[[1, 1], [0, 1]], [[1, 0], [0, 1]], [[1, 2], [3, 4]]]),
            case([[[0, 0]], [[1, 2]], [[5, 6]]]),
            case([[[2, -1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]),
            case([[[1, 2, 3]], [[1, 0, 0], [0, 1, 0]], [[1], [2]]]),
            case([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]),
            case([[[3, 4]], [[3, 4], [4, 3]], [[1, 1], [2, 2]]]),
            case([[[1]], [[1], [2]], [[10], [20]]]),
        ],
        imports=py_imports(),
        solution_code="import math\nimport numpy as np\n\ndef scaled_dot_product_attention(q, k, v):\n    q = np.asarray(q, dtype=float)\n    k = np.asarray(k, dtype=float)\n    v = np.asarray(v, dtype=float)\n    scores = q @ k.T / math.sqrt(q.shape[-1])\n    shifted = scores - np.max(scores, axis=1, keepdims=True)\n    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)\n    return probs @ v\n",
        explanation="注意力先计算缩放点积得分，再按 key 维度 softmax，最后对 value 加权求和。",
        constraints=["q、k、v 为二维数组", "k 与 v 的序列长度一致", "误差容忍 1e-6"],
    )

    def ref_masked(scores: Any, mask: Any) -> np.ndarray:
        arr = np.asarray(scores, dtype=float).copy()
        arr[np.asarray(mask, dtype=bool)] = -1e30
        return ref_softmax_2d(arr)

    add(
        slug="attention-mask-softmax",
        title="带 Mask 的 Attention Softmax",
        difficulty="中等",
        category="Attention 与 Transformer",
        function_name="attention_with_mask",
        signature="def attention_with_mask(scores: Any, mask: Any) -> np.ndarray",
        description="对 attention scores 做 softmax，mask 为 True 的位置不可见，概率应接近 0。",
        reference=ref_masked,
        raw_cases=[
            case([[[1, 2, 3]], [[False, False, True]]]),
            case([[[0, 0], [1, 1]], [[False, True], [True, False]]]),
            case([[[5]], [[False]]]),
            case([[[10, 0, -10]], [[False, True, False]]]),
            case([[[-1, 2]], [[True, False]]]),
            case([[[1, 1, 1]], [[False, False, False]]]),
            case([[[3, 4], [5, 6]], [[False, False], [False, True]]]),
            case([[[0, 1]], [[True, False]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef attention_with_mask(scores, mask):\n    arr = np.asarray(scores, dtype=float).copy()\n    arr[np.asarray(mask, dtype=bool)] = -1e30\n    shifted = arr - np.max(arr, axis=1, keepdims=True)\n    exps = np.exp(shifted)\n    return exps / np.sum(exps, axis=1, keepdims=True)\n",
        explanation="被 mask 的位置用极小值替代，再做 softmax。",
        constraints=["scores 与 mask 形状一致", "每行至少一个未 mask 位置"],
    )

    def ref_pos(length: int, dim: int) -> np.ndarray:
        pe = np.zeros((length, dim), dtype=float)
        for pos in range(length):
            for i in range(0, dim, 2):
                div = 10000 ** (i / dim)
                pe[pos, i] = math.sin(pos / div)
                if i + 1 < dim:
                    pe[pos, i + 1] = math.cos(pos / div)
        return pe

    add(
        slug="sinusoidal-positional-encoding",
        title="正弦位置编码",
        difficulty="困难",
        category="Attention 与 Transformer",
        function_name="positional_encoding",
        signature="def positional_encoding(length: int, dim: int) -> np.ndarray",
        description="实现 Transformer 经典正弦位置编码，偶数维使用 sin，奇数维使用 cos。",
        reference=ref_pos,
        raw_cases=[
            case([2, 4]),
            case([1, 3]),
            case([0, 4]),
            case([3, 2]),
            case([4, 6]),
            case([5, 1]),
            case([2, 5]),
            case([6, 4]),
        ],
        imports=py_imports(),
        solution_code="import math\nimport numpy as np\n\ndef positional_encoding(length, dim):\n    pe = np.zeros((length, dim), dtype=float)\n    for pos in range(length):\n        for i in range(0, dim, 2):\n            div = 10000 ** (i / dim)\n            pe[pos, i] = math.sin(pos / div)\n            if i + 1 < dim:\n                pe[pos, i + 1] = math.cos(pos / div)\n    return pe\n",
        explanation="位置编码用不同频率的 sin/cos 表示绝对位置，奇偶维公式不同。",
        constraints=["length >= 0", "dim >= 1", "误差容忍 1e-6"],
    )

    def ref_split_heads(x: Any, num_heads: int) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        batch, seq, dim = arr.shape
        head_dim = dim // num_heads
        return arr.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)

    add(
        slug="split-heads",
        title="拆分多头",
        difficulty="中等",
        category="Attention 与 Transformer",
        function_name="split_heads",
        signature="def split_heads(x: Any, num_heads: int) -> np.ndarray",
        description="把形状 (batch, seq, dim) 的张量拆成 (batch, heads, seq, head_dim)。",
        reference=ref_split_heads,
        raw_cases=[
            case([[[[1, 2, 3, 4]]], 2]),
            case([[[[1, 2], [3, 4]]], 1]),
            case([[[[1, 2, 3, 4], [5, 6, 7, 8]]], 4]),
            case([[[[1, 2, 3, 4]], [[5, 6, 7, 8]]], 2]),
            case([[[[0, 0]]], 2]),
            case([[[[1, 2, 3, 4, 5, 6]]], 3]),
            case([[[[1, 2], [3, 4], [5, 6]]], 2]),
            case([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], 2]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef split_heads(x, num_heads):\n    arr = np.asarray(x, dtype=float)\n    batch, seq, dim = arr.shape\n    head_dim = dim // num_heads\n    return arr.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)\n",
        explanation="先 reshape 出 heads 维度，再转置到注意力计算常用的维度顺序。",
        constraints=["dim 能被 num_heads 整除", "输入为三维数组"],
    )

    add(
        slug="combine-heads",
        title="合并多头",
        difficulty="中等",
        category="Attention 与 Transformer",
        function_name="combine_heads",
        signature="def combine_heads(x: Any) -> np.ndarray",
        description="把形状 (batch, heads, seq, head_dim) 的张量合并回 (batch, seq, heads*head_dim)。",
        reference=lambda x: np.asarray(x, dtype=float).transpose(0, 2, 1, 3).reshape(
            np.asarray(x).shape[0], np.asarray(x).shape[2], np.asarray(x).shape[1] * np.asarray(x).shape[3]
        ),
        raw_cases=[
            case([[[[[1, 2]], [[3, 4]]]]]),
            case([[[[[1], [2]], [[3], [4]]]]]),
            case([[[[[1, 2, 3, 4]]]]]),
            case([[[[[1]], [[2]], [[3]]]]]),
            case([[[[[0, 0]], [[1, 1]]]]]),
            case([[[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]]]),
            case([[[[[1], [2], [3]], [[4], [5], [6]]]]]),
            case([[[[[1, 2, 3]], [[4, 5, 6]]]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef combine_heads(x):\n    arr = np.asarray(x, dtype=float)\n    batch, heads, seq, head_dim = arr.shape\n    return arr.transpose(0, 2, 1, 3).reshape(batch, seq, heads * head_dim)\n",
        explanation="拆头的逆操作是先把 seq 放回第二维，再合并 heads 和 head_dim。",
        constraints=["输入为四维数组"],
    )

    add(
        slug="greedy-decode-step",
        title="Greedy Decoding 单步",
        difficulty="简单",
        category="Attention 与 Transformer",
        function_name="greedy_decode_step",
        signature="def greedy_decode_step(logits: Any) -> list[int]",
        description="给定 batch x vocab 的 logits，返回每个样本最大 logit 的下标。",
        reference=lambda logits: np.asarray(logits).argmax(axis=1).astype(int).tolist(),
        raw_cases=[
            case([[[1, 3, 2], [0, -1, 5]]]),
            case([[[0, 0]]]),
            case([[[5]]]),
            case([[[-1, -2], [3, 2]]]),
            case([[[0.1, 0.2, 0.3]]]),
            case([[[10, 9, 8], [1, 2, 3]]]),
            case([[[1, 1, 0]]]),
            case([[[2, 4], [4, 2], [3, 3]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef greedy_decode_step(logits):\n    return np.asarray(logits).argmax(axis=1).astype(int).tolist()\n",
        explanation="贪心解码每一步选择概率或 logit 最大的 token。",
        constraints=["logits 为二维数组", "并列时 NumPy argmax 返回最小下标"],
    )

    add(
        slug="top-k-sampling-candidates",
        title="Top-K Sampling 候选集",
        difficulty="简单",
        category="Attention 与 Transformer",
        function_name="top_k_sampling_candidates",
        signature="def top_k_sampling_candidates(probs: list[float], k: int) -> list[int]",
        description="返回概率最大的 k 个 token 下标，按概率从大到小排列；概率相同下标小的在前。",
        reference=lambda probs, k: [i for i, _ in sorted(enumerate(probs), key=lambda item: (-item[1], item[0]))[:k]],
        raw_cases=[
            case([[0.1, 0.7, 0.2], 2]),
            case([[0.5, 0.5], 1]),
            case([[1.0], 5]),
            case([[0.3, 0.2, 0.4, 0.1], 3]),
            case([[0, 0, 0], 2]),
            case([[0.9, 0.05, 0.05], 3]),
            case([[0.2, 0.8], 0]),
            case([[0.1, 0.2, 0.2], 2]),
        ],
        imports=py_imports(),
        solution_code="def top_k_sampling_candidates(probs, k):\n    return [i for i, _ in sorted(enumerate(probs), key=lambda item: (-item[1], item[0]))[:k]]\n",
        explanation="Top-K Sampling 先截断候选 token，再在候选集合内采样；本题只要求返回候选下标。",
        constraints=["0 <= k", "k 可超过词表大小"],
    )

    def ref_smooth(labels: list[int], num_classes: int, epsilon: float) -> np.ndarray:
        off = epsilon / num_classes
        arr = np.full((len(labels), num_classes), off, dtype=float)
        for i, label in enumerate(labels):
            arr[i, label] = 1 - epsilon + off
        return arr

    add(
        slug="label-smoothing",
        title="Label Smoothing",
        difficulty="中等",
        category="Attention 与 Transformer",
        function_name="label_smoothing",
        signature="def label_smoothing(labels: list[int], num_classes: int, epsilon: float) -> np.ndarray",
        description="把类别标签转换为 label smoothing 后的分布：真实类为 1-epsilon+epsilon/C，其余为 epsilon/C。",
        reference=ref_smooth,
        raw_cases=[
            case([[0, 2], 3, 0.1]),
            case([[1], 4, 0.2]),
            case([[], 3, 0.1]),
            case([[0], 1, 0.5]),
            case([[2, 2, 0], 3, 0.3]),
            case([[3], 5, 0.0]),
            case([[1, 0], 2, 0.1]),
            case([[4, 1, 4], 5, 0.2]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef label_smoothing(labels, num_classes, epsilon):\n    off = epsilon / num_classes\n    arr = np.full((len(labels), num_classes), off, dtype=float)\n    for i, label in enumerate(labels):\n        arr[i, label] = 1 - epsilon + off\n    return arr\n",
        explanation="Label smoothing 会降低真实类别的置信度，同时给其他类别分配少量概率。",
        constraints=["0 <= epsilon <= 1", "标签下标合法", "误差容忍 1e-6"],
    )

    # 计算机视觉
    def ref_conv2d(image: Any, kernel: Any) -> np.ndarray:
        img = np.asarray(image, dtype=float)
        ker = np.asarray(kernel, dtype=float)
        h, w = img.shape
        kh, kw = ker.shape
        out = np.zeros((h - kh + 1, w - kw + 1), dtype=float)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = np.sum(img[i : i + kh, j : j + kw] * ker)
        return out

    add(
        slug="conv2d-valid",
        title="二维有效卷积",
        difficulty="困难",
        category="计算机视觉",
        function_name="conv2d_valid",
        signature="def conv2d_valid(image: Any, kernel: Any) -> np.ndarray",
        description="实现 stride=1、无 padding 的二维有效卷积。按深度学习互相关写法，不翻转 kernel。",
        reference=ref_conv2d,
        raw_cases=[
            case([[[1, 2], [3, 4]], [[1, 0], [0, 1]]]),
            case([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 1], [1, 1]]]),
            case([[[5]], [[2]]]),
            case([[[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[1, -1], [-1, 1]]]),
            case([[[-1, 2], [3, -4]], [[0.5, 0.5], [0.5, 0.5]]]),
            case([[[1, 1, 1], [1, 1, 1]], [[1, 1]]]),
            case([[[2, 4, 6], [8, 10, 12], [14, 16, 18]], [[0, 1], [1, 0]]]),
            case([[[3, 2, 1]], [[-1]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef conv2d_valid(image, kernel):\n    img = np.asarray(image, dtype=float)\n    ker = np.asarray(kernel, dtype=float)\n    h, w = img.shape\n    kh, kw = ker.shape\n    out = np.zeros((h - kh + 1, w - kw + 1), dtype=float)\n    for i in range(out.shape[0]):\n        for j in range(out.shape[1]):\n            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * ker)\n    return out\n",
        explanation="二维卷积遍历每个窗口，与 kernel 逐元素相乘后求和。",
        constraints=["kernel 尺寸不大于 image", "返回 np.ndarray"],
    )

    def ref_pool(image: Any, kernel_size: int, stride: int) -> np.ndarray:
        img = np.asarray(image, dtype=float)
        h, w = img.shape
        out_h = (h - kernel_size) // stride + 1
        out_w = (w - kernel_size) // stride + 1
        out = np.zeros((out_h, out_w), dtype=float)
        for i in range(out_h):
            for j in range(out_w):
                out[i, j] = np.max(img[i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size])
        return out

    add(
        slug="max-pool2d",
        title="二维最大池化",
        difficulty="中等",
        category="计算机视觉",
        function_name="max_pool2d",
        signature="def max_pool2d(image: Any, kernel_size: int, stride: int) -> np.ndarray",
        description="实现单通道二维最大池化，输入为二维数组。",
        reference=ref_pool,
        raw_cases=[
            case([[[1, 2], [3, 4]], 2, 1]),
            case([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, 1]),
            case([[[5]], 1, 1]),
            case([[[0, 1, 0], [1, 0, 1], [0, 1, 0]], 2, 2]),
            case([[[-1, 2], [3, -4]], 2, 1]),
            case([[[1, 1, 1], [1, 1, 1]], 1, 1]),
            case([[[2, 4, 6], [8, 10, 12], [14, 16, 18]], 3, 1]),
            case([[[3, 2, 1, 0]], 1, 2]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef max_pool2d(image, kernel_size, stride):\n    img = np.asarray(image, dtype=float)\n    h, w = img.shape\n    out_h = (h - kernel_size) // stride + 1\n    out_w = (w - kernel_size) // stride + 1\n    out = np.zeros((out_h, out_w), dtype=float)\n    for i in range(out_h):\n        for j in range(out_w):\n            out[i, j] = np.max(img[i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size])\n    return out\n",
        explanation="最大池化在每个窗口中取最大值，stride 控制窗口移动距离。",
        constraints=["kernel_size >= 1", "stride >= 1", "返回 np.ndarray"],
    )

    def ref_iou(a: list[float], b: list[float]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return 0.0 if union == 0 else inter / union

    add(
        slug="box-iou",
        title="边界框 IoU",
        difficulty="简单",
        category="计算机视觉",
        function_name="box_iou",
        signature="def box_iou(box_a: list[float], box_b: list[float]) -> float",
        description="计算两个边界框的交并比。框格式为 [x1, y1, x2, y2]，坐标使用连续面积。",
        reference=ref_iou,
        raw_cases=[
            case([[0, 0, 2, 2], [1, 1, 3, 3]]),
            case([[0, 0, 1, 1], [2, 2, 3, 3]]),
            case([[0, 0, 1, 1], [0, 0, 1, 1]]),
            case([[0, 0, 0, 1], [0, 0, 1, 1]]),
            case([[1, 1, 4, 4], [2, 2, 3, 3]]),
            case([[-1, -1, 1, 1], [0, 0, 2, 2]]),
            case([[0, 0, 10, 5], [5, 0, 15, 5]]),
            case([[0, 0, 2, 3], [1, 0, 3, 3]]),
        ],
        imports=py_imports(),
        solution_code="def box_iou(box_a, box_b):\n    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])\n    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])\n    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)\n    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])\n    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])\n    union = area_a + area_b - inter\n    return 0.0 if union == 0 else inter / union\n",
        explanation="IoU 等于交集面积除以并集面积，无交集或零面积时要避免除零。",
        constraints=["坐标格式为 x1,y1,x2,y2", "误差容忍 1e-6"],
    )

    def ref_nms(boxes: list[list[float]], scores: list[float], iou_threshold: float) -> list[int]:
        order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
        keep = []
        while order:
            cur = order.pop(0)
            keep.append(cur)
            order = [idx for idx in order if ref_iou(boxes[cur], boxes[idx]) <= iou_threshold]
        return keep

    add(
        slug="nms",
        title="非极大值抑制 NMS",
        difficulty="困难",
        category="计算机视觉",
        function_name="nms",
        signature="def nms(boxes: list[list[float]], scores: list[float], iou_threshold: float) -> list[int]",
        description="按分数从高到低执行 NMS，返回保留框的原始下标。分数相同下标小的优先。",
        reference=ref_nms,
        raw_cases=[
            case([[[0, 0, 2, 2], [0.5, 0.5, 2.5, 2.5], [3, 3, 4, 4]], [0.9, 0.8, 0.7], 0.3]),
            case([[[0, 0, 1, 1], [2, 2, 3, 3]], [0.5, 0.6], 0.5]),
            case([[], [], 0.5]),
            case([[[0, 0, 1, 1]], [0.1], 0.5]),
            case([[[0, 0, 2, 2], [1, 1, 3, 3]], [0.5, 0.5], 0.1]),
            case([[[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 21, 21]], [0.7, 0.9, 0.1], 0.5]),
            case([[[0, 0, 2, 3], [1, 0, 3, 3], [4, 4, 5, 5]], [0.3, 0.4, 0.2], 0.4]),
            case([[[0, 0, 2, 2], [0, 0, 2, 2]], [0.1, 0.2], 0.5]),
        ],
        imports=py_imports(),
        solution_code="def _iou(a, b):\n    x1, y1 = max(a[0], b[0]), max(a[1], b[1])\n    x2, y2 = min(a[2], b[2]), min(a[3], b[3])\n    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)\n    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])\n    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])\n    union = area_a + area_b - inter\n    return 0.0 if union == 0 else inter / union\n\ndef nms(boxes, scores, iou_threshold):\n    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))\n    keep = []\n    while order:\n        cur = order.pop(0)\n        keep.append(cur)\n        order = [idx for idx in order if _iou(boxes[cur], boxes[idx]) <= iou_threshold]\n    return keep\n",
        explanation="每次选择当前最高分框，并删除与它 IoU 超过阈值的其他框。",
        constraints=["boxes 与 scores 长度一致", "返回原始下标列表"],
    )

    def ref_patch(image: Any, patch_size: int) -> np.ndarray:
        img = np.asarray(image, dtype=float)
        h, w = img.shape
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patches.append(img[i : i + patch_size, j : j + patch_size].reshape(-1))
        return np.vstack(patches) if patches else np.empty((0, patch_size * patch_size))

    add(
        slug="patch-embedding-flatten",
        title="图像 Patch 展平",
        difficulty="中等",
        category="计算机视觉",
        function_name="patch_embedding_flatten",
        signature="def patch_embedding_flatten(image: Any, patch_size: int) -> np.ndarray",
        description="把二维图像按不重叠 patch 切分，并把每个 patch 展平成一行。假设高宽都能被 patch_size 整除。",
        reference=ref_patch,
        raw_cases=[
            case([[[1, 2], [3, 4]], 1]),
            case([[[1, 2], [3, 4]], 2]),
            case([[[1, 2, 3, 4], [5, 6, 7, 8]], 2]),
            case([[[0]], 1]),
            case([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], 2]),
            case([[[1, 1], [1, 1]], 2]),
            case([[[2, 4, 6, 8]], 2]),
            case([[[3, 2], [1, 0], [5, 4], [7, 6]], 2]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef patch_embedding_flatten(image, patch_size):\n    img = np.asarray(image, dtype=float)\n    h, w = img.shape\n    patches = []\n    for i in range(0, h, patch_size):\n        for j in range(0, w, patch_size):\n            patches.append(img[i:i + patch_size, j:j + patch_size].reshape(-1))\n    return np.vstack(patches) if patches else np.empty((0, patch_size * patch_size))\n",
        explanation="ViT 的 patch embedding 通常先切分 patch，再展平并线性投影。",
        constraints=["高宽能被 patch_size 整除", "返回二维 np.ndarray"],
    )

    # 自然语言处理
    add(
        slug="word-count",
        title="词频统计",
        difficulty="简单",
        category="自然语言处理",
        function_name="word_count",
        signature="def word_count(tokens: list[str]) -> dict[str, int]",
        description="统计 token 列表中每个词出现的次数，返回字典。",
        reference=lambda tokens: dict(Counter(tokens)),
        raw_cases=[
            case([["我", "爱", "AI", "AI"]]),
            case([[]]),
            case([["a"]]),
            case([["深度", "学习", "深度"]]),
            case([["x", "y", "x", "z", "y"]]),
            case([["Hello", "hello"]]),
            case([["1", "1", "2"]]),
            case([["pad", "pad", "pad"]]),
        ],
        imports=py_imports(),
        solution_code="from collections import Counter\n\ndef word_count(tokens):\n    return dict(Counter(tokens))\n",
        explanation="词频统计是 NLP 文本特征构建的基础步骤。",
        constraints=["tokens 可以为空", "区分大小写"],
    )

    add(
        slug="ngram-list",
        title="N-Gram 生成",
        difficulty="简单",
        category="自然语言处理",
        function_name="ngrams",
        signature="def ngrams(tokens: list[str], n: int) -> list[list[str]]",
        description="返回连续 n 个 token 组成的片段列表。若 n 非法或超过长度，返回空列表。",
        reference=lambda tokens, n: [] if n <= 0 or n > len(tokens) else [tokens[i : i + n] for i in range(len(tokens) - n + 1)],
        raw_cases=[
            case([["我", "爱", "AI"], 2]),
            case([["a", "b", "c"], 3]),
            case([["a"], 2]),
            case([[], 1]),
            case([["x", "y", "z", "w"], 1]),
            case([["x", "y", "z", "w"], 4]),
            case([["x", "y"], 0]),
            case([["深度", "学习", "模型"], 2]),
        ],
        imports=py_imports(),
        solution_code="def ngrams(tokens, n):\n    if n <= 0 or n > len(tokens):\n        return []\n    return [tokens[i:i + n] for i in range(len(tokens) - n + 1)]\n",
        explanation="滑动一个长度为 n 的窗口即可得到 N-Gram。",
        constraints=["n 可以非法", "返回列表中的每个 N-Gram 使用 list"],
    )

    def ref_pad(sequences: list[list[int]], pad_value: int) -> list[list[int]]:
        max_len = max((len(seq) for seq in sequences), default=0)
        return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

    add(
        slug="pad-sequences",
        title="序列 Padding",
        difficulty="简单",
        category="自然语言处理",
        function_name="pad_sequences",
        signature="def pad_sequences(sequences: list[list[int]], pad_value: int = 0) -> list[list[int]]",
        description="把不同长度的整数序列补齐到当前 batch 的最大长度。",
        reference=ref_pad,
        raw_cases=[
            case([[[1, 2], [3]], 0]),
            case([[], 0]),
            case([[[1], [2, 3, 4]], -1]),
            case([[[1, 2]], 0]),
            case([[[], [1, 2]], 9]),
            case([[[5], [], [6, 7, 8]], 0]),
            case([[[0, 0], [1]], 0]),
            case([[[1], [2], [3]], 99]),
        ],
        imports=py_imports(),
        solution_code="def pad_sequences(sequences, pad_value=0):\n    max_len = max((len(seq) for seq in sequences), default=0)\n    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]\n",
        explanation="先找 batch 内最大长度，再对每条序列补 pad_value。",
        constraints=["sequences 可以为空", "不修改原输入"],
    )

    def ref_seq_ce(logits: Any, labels: Any, pad_id: int) -> float:
        arr = np.asarray(logits, dtype=float)
        lab = np.asarray(labels, dtype=int)
        total = 0.0
        count = 0
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                if lab[i, j] == pad_id:
                    continue
                probs = ref_softmax_2d(arr[i, j : j + 1])[0]
                total += -math.log(probs[lab[i, j]] + 1e-12)
                count += 1
        return 0.0 if count == 0 else total / count

    add(
        slug="sequence-cross-entropy-ignore-pad",
        title="忽略 Padding Token 的序列交叉熵",
        difficulty="困难",
        category="自然语言处理",
        function_name="sequence_cross_entropy",
        signature="def sequence_cross_entropy(logits: Any, labels: Any, pad_id: int) -> float",
        description="给定 batch x seq x vocab 的 logits 和标签，计算非 padding 位置的平均交叉熵。",
        reference=ref_seq_ce,
        raw_cases=[
            case([[[[2, 0], [0, 2]]], [[0, 1]], -100]),
            case([[[[1, 1], [3, 0]]], [[-1, 0]], -1]),
            case([[[[1, 2, 3]]], [[2]], 0]),
            case([[[[0, 0], [0, 0]]], [[0, 0]], 0]),
            case([[[[10, 0], [0, 10]], [[1, 1], [2, 2]]], [[0, 1], [1, 0]], -100]),
            case([[[[1, 0, 0], [0, 1, 0]]], [[0, -100]], -100]),
            case([[[[0, 5], [5, 0], [1, 1]]], [[1, 0, 1]], -1]),
            case([[[[2, 1], [1, 2]]], [[-9, -9]], -9]),
        ],
        imports=py_imports(),
        solution_code="import math\nimport numpy as np\n\ndef sequence_cross_entropy(logits, labels, pad_id):\n    arr = np.asarray(logits, dtype=float)\n    lab = np.asarray(labels, dtype=int)\n    total = 0.0\n    count = 0\n    for i in range(lab.shape[0]):\n        for j in range(lab.shape[1]):\n            if lab[i, j] == pad_id:\n                continue\n            row = arr[i, j]\n            shifted = row - np.max(row)\n            probs = np.exp(shifted) / np.sum(np.exp(shifted))\n            total += -math.log(probs[lab[i, j]] + 1e-12)\n            count += 1\n    return 0.0 if count == 0 else total / count\n",
        explanation="序列任务通常需要忽略 padding 位置，只对有效 token 求平均损失。",
        constraints=["logits 为三维数组", "labels 为二维数组", "误差容忍 1e-6"],
    )

    # 训练、调试与工程
    add(
        slug="detect-nan",
        title="检测 NaN",
        difficulty="简单",
        category="训练、调试与工程",
        function_name="has_nan",
        signature="def has_nan(values: Any) -> bool",
        description="判断输入数组或嵌套列表中是否存在 NaN。",
        reference=lambda values: bool(np.isnan(np.asarray(values, dtype=float)).any()),
        raw_cases=[
            case([[1.0, 2.0, 3.0]]),
            case([[1.0, {"__type__": "nan"}, 3.0]]),
            case([[[1.0], [2.0]]]),
            case([[[1.0], [{"__type__": "nan"}]]]),
            case([[0.0]]),
            case([[-1.0, float("inf")]]),
            case([[]]),
            case([[[[1.0, 2.0]]]]),
        ],
        imports=py_imports(),
        solution_code="import numpy as np\n\ndef has_nan(values):\n    return bool(np.isnan(np.asarray(values, dtype=float)).any())\n",
        explanation="NumPy 的 isnan 可以对数组逐元素检测 NaN。",
        constraints=["输入可转换为浮点数组", "正负无穷不算 NaN"],
    )

    add(
        slug="count-parameters",
        title="参数量统计",
        difficulty="简单",
        category="训练、调试与工程",
        function_name="count_parameters",
        signature="def count_parameters(shapes: list[list[int]]) -> int",
        description="给定多个参数张量的 shape，返回总参数量。",
        reference=lambda shapes: sum(math.prod(shape) for shape in shapes),
        raw_cases=[
            case([[[2, 3], [3]]]),
            case([[]]),
            case([[[10]]]),
            case([[[2, 3, 4], [4, 5]]]),
            case([[[1, 1], [1, 1, 1]]]),
            case([[[100, 200], [200]]]),
            case([[[0, 3]]]),
            case([[[2], [3], [4]]]),
        ],
        imports=py_imports(),
        solution_code="import math\n\ndef count_parameters(shapes):\n    return sum(math.prod(shape) for shape in shapes)\n",
        explanation="每个张量的参数量是 shape 各维度乘积，总参数量再求和。",
        constraints=["shape 中维度为非负整数", "空列表返回 0"],
    )

    def ref_split(n: int, val_ratio: float, seed: int) -> dict[str, list[int]]:
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)
        val_size = int(n * val_ratio)
        return {"train": indices[val_size:], "val": indices[:val_size]}

    add(
        slug="reproducible-train-val-split",
        title="可复现训练集与验证集划分",
        difficulty="中等",
        category="训练、调试与工程",
        function_name="train_val_split_indices",
        signature="def train_val_split_indices(n: int, val_ratio: float, seed: int) -> dict[str, list[int]]",
        description="返回可复现的训练/验证下标划分。先用 seed 打乱 range(n)，验证集大小为 int(n * val_ratio)。",
        reference=ref_split,
        raw_cases=[
            case([5, 0.4, 42]),
            case([0, 0.2, 1]),
            case([3, 0.0, 7]),
            case([4, 1.0, 0]),
            case([10, 0.3, 123]),
            case([6, 0.5, 42]),
            case([1, 0.5, 9]),
            case([8, 0.25, 2024]),
        ],
        imports=py_imports(),
        solution_code="import random\n\ndef train_val_split_indices(n, val_ratio, seed):\n    indices = list(range(n))\n    rng = random.Random(seed)\n    rng.shuffle(indices)\n    val_size = int(n * val_ratio)\n    return {'train': indices[val_size:], 'val': indices[:val_size]}\n",
        explanation="固定随机种子可以让实验划分可复现，验证集大小按题意向下取整。",
        constraints=["0 <= val_ratio <= 1", "n >= 0"],
    )

    assert len(problems) == 60
    return problems
