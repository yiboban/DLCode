from __future__ import annotations

import ast
from typing import Any

from .manual_solutions import build_solution_variants


FunctionNote = tuple[str, str, int]


EXACT_FUNCTION_NOTES: dict[str, FunctionNote] = {
    "torch.as_tensor": ("torch.as_tensor", "把 Python 标量或列表转换为 Tensor；输入本身已经是 Tensor 时会尽量避免额外复制。", 10),
    "torch.tensor": ("torch.tensor", "根据给定数据创建新的 Tensor；需要梯度时可同时设置 requires_grad=True。", 10),
    "torch.zeros": ("torch.zeros", "创建指定形状的全零 Tensor，常用于初始化输出、掩码或缓存。", 10),
    "torch.zeros_like": ("torch.zeros_like", "创建与输入形状、dtype 和 device 一致的全零 Tensor。", 10),
    "torch.ones": ("torch.ones", "创建指定形状的全一 Tensor，本题中常用于构造布尔掩码或基础张量。", 10),
    "torch.full": ("torch.full", "用同一个值填充指定形状的 Tensor，适合初始化平滑标签等结构。", 10),
    "torch.arange": ("torch.arange", "生成等间隔下标序列，常用于位置、类别标签和向量化索引。", 10),
    "torch.empty_like": ("torch.empty_like", "创建与输入形状、dtype 和 device 一致但未初始化的 Tensor。", 10),
    "torch.softmax": ("torch.softmax", "沿指定维度把 logits 归一化为概率；要特别确认 dim 对应的是类别或序列维。", 10),
    "torch.sigmoid": ("torch.sigmoid", "逐元素计算 Sigmoid，把实数 logits 映射到 0 到 1。", 10),
    "torch.relu": ("torch.relu", "逐元素保留正值并把负值截断为 0。", 10),
    "torch.sin": ("torch.sin", "逐元素计算正弦值；位置编码中与偶数维频率配合使用。", 10),
    "torch.cos": ("torch.cos", "逐元素计算余弦值；位置编码中与奇数维频率配合使用。", 10),
    "torch.pow": ("torch.pow", "执行幂运算，支持 Tensor 与标量广播。", 10),
    "torch.sqrt": ("torch.sqrt", "逐元素开平方；归一化时通常会和 eps 一起使用以避免除零。", 10),
    "torch.rsqrt": ("torch.rsqrt", "逐元素计算平方根的倒数，等价于 1 / sqrt(x)，常用于高效归一化。", 10),
    "torch.exp": ("torch.exp", "逐元素计算自然指数，常用于概率、困惑度和数值变换。", 10),
    "torch.cdist": ("torch.cdist", "批量计算两组向量之间的成对距离，输出最后两维组成的距离矩阵。", 10),
    "torch.argsort": ("torch.argsort", "返回排序后的下标而不是数值本身，适合候选筛选和重排。", 10),
    "torch.cumsum": ("torch.cumsum", "沿指定维度计算累积和，Top-p 采样用它寻找累计概率阈值。", 10),
    "torch.searchsorted": ("torch.searchsorted", "在有序序列中寻找插入位置，可快速定位累计概率首次越过阈值的位置。", 10),
    "torch.topk": ("torch.topk", "沿指定维度返回最大的 k 个值及其下标。", 10),
    "torch.clamp": ("torch.clamp", "把数值截断到给定区间，量化和数值稳定处理中很常见。", 10),
    "torch.round": ("torch.round", "把浮点数舍入到最近整数，常用于模拟整数量化。", 10),
    "torch.isnan": ("torch.isnan", "逐元素判断是否为 NaN，返回布尔 Tensor。", 10),
    "torch.triu": ("torch.triu", "提取矩阵上三角部分；设置 diagonal=1 可得到严格上三角因果掩码。", 10),
    "torch.where": ("torch.where", "根据布尔条件逐元素选择两个分支，适合向量化条件计算。", 10),
    "torch.no_grad": ("torch.no_grad", "临时关闭 autograd 记录，参数更新等不需要反向传播的操作应放在其中。", 10),
    "torch.linalg.vector_norm": ("torch.linalg.vector_norm", "沿指定维度计算向量范数，语义比手动平方求和更明确。", 10),
    "F.one_hot": ("torch.nn.functional.one_hot", "把整数类别下标转换为 One-hot Tensor，输出最后一维是类别数。", 10),
    "F.cross_entropy": ("torch.nn.functional.cross_entropy", "直接接收未做 Softmax 的 logits，内部使用稳定的 log-softmax 与负对数似然。", 10),
    "F.binary_cross_entropy_with_logits": (
        "torch.nn.functional.binary_cross_entropy_with_logits",
        "把 Sigmoid 与二元交叉熵合并计算，比先 Sigmoid 再取对数更稳定。",
        10,
    ),
    "F.conv1d": ("torch.nn.functional.conv1d", "执行一维卷积/互相关；输入、权重通常需要补齐 batch 和 channel 维。", 10),
    "F.conv2d": ("torch.nn.functional.conv2d", "执行二维卷积/互相关；需要明确 NCHW 输入和 OIHW 权重布局。", 10),
    "F.max_pool2d": ("torch.nn.functional.max_pool2d", "在二维滑动窗口内取最大值，实现空间下采样。", 10),
    "F.silu": ("torch.nn.functional.silu", "计算 SiLU(x)=x·sigmoid(x)，是 SwiGLU 门控分支的核心激活。", 10),
    "F.linear": ("torch.nn.functional.linear", "计算 y=xWᵀ+b；LoRA 等题目常在基础线性层外叠加低秩增量。", 10),
    "F.normalize": ("torch.nn.functional.normalize", "沿指定维度执行范数归一化，常用于余弦相似度和对比学习。", 10),
    "F.log_softmax": ("torch.nn.functional.log_softmax", "稳定地计算 log(softmax(x))，避免先 Softmax 再取对数。", 10),
    "F.softmax": ("torch.nn.functional.softmax", "沿指定维度把 logits 转换为概率分布。", 10),
    "F.kl_div": ("torch.nn.functional.kl_div", "计算 KL 散度；输入通常是对数概率，target 通常是概率。", 10),
    "math.sqrt": ("math.sqrt", "计算标量平方根，常用于范数、标准差和缩放因子。", 20),
    "math.exp": ("math.exp", "计算标量自然指数；大数输入时要先做平移以保证数值稳定。", 20),
    "math.log": ("math.log", "计算自然对数，输入必须为正数。", 20),
    "math.log2": ("math.log2", "计算以 2 为底的对数，信息熵常以 bit 为单位使用它。", 20),
    "math.cos": ("math.cos", "计算标量余弦值，余弦退火学习率中用于平滑下降。", 20),
    "math.prod": ("math.prod", "计算可迭代对象中所有元素的乘积，可用于由 shape 得到参数量。", 20),
    "Counter": ("collections.Counter", "统计可哈希元素出现次数，适合多数投票和类别频率计算。", 20),
    "random.Random": ("random.Random", "创建独立的伪随机数生成器，固定 seed 后不会污染全局随机状态。", 20),
    "zip": ("zip", "把多个可迭代对象按位置配对；矩阵转置时 zip(*matrix) 会按列聚合。", 30),
    "sorted": ("sorted", "返回新的有序列表，可通过 key 和 reverse 控制排序规则。", 30),
    "enumerate": ("enumerate", "遍历序列时同时得到下标和值。", 30),
    "sum": ("sum", "对可迭代对象累加；生成器表达式可避免创建中间列表。", 30),
    "max": ("max", "取得最大值，也可以通过 key 指定比较依据。", 30),
    "min": ("min", "取得最小值，也可以通过 key 指定比较依据。", 30),
    "range": ("range", "生成惰性的整数区间，常用于滑动窗口和下标遍历。", 35),
    "list": ("list", "把可迭代对象展开为列表，常用于把 zip、Tensor 结果转换为题目要求的结构。", 35),
    "dict": ("dict", "创建字典，用于键值映射或返回结构化结果。", 35),
    "TinyDataset": ("TinyDataset", "实例化题目中实现的数据集类，并通过 __len__ / __getitem__ 暴露样本。", 20),
}


METHOD_FUNCTION_NOTES: dict[str, FunctionNote] = {
    "mean": ("Tensor.mean", "沿指定维度求平均；keepdim=True 可保留维度，方便后续广播。", 12),
    "var": ("Tensor.var", "沿指定维度计算方差；归一化题要留意 unbiased 参数。", 12),
    "sum": ("Tensor.sum", "沿指定维度求和；不指定 dim 时会约简全部元素。", 12),
    "pow": ("Tensor.pow", "逐元素做幂运算，x.pow(2) 常用于平方。", 12),
    "square": ("Tensor.square", "逐元素平方，语义等价于 x ** 2。", 12),
    "view": ("Tensor.view", "在内存连续且元素数不变时重解释形状。", 12),
    "reshape": ("Tensor.reshape", "改变 Tensor 形状；必要时会创建连续副本。", 12),
    "flatten": ("Tensor.flatten", "把一段连续维度压平成一维。", 12),
    "permute": ("Tensor.permute", "按给定顺序重排维度，常用于多头注意力的 shape 变换。", 12),
    "transpose": ("Tensor.transpose", "交换两个维度，矩阵乘法前常用于转置最后两维。", 12),
    "contiguous": ("Tensor.contiguous", "把非连续视图整理为连续内存，通常在 view 前调用。", 12),
    "unfold": ("Tensor.unfold", "创建滑动窗口视图，可用于提取图像 patch 或局部窗口。", 12),
    "squeeze": ("Tensor.squeeze", "移除长度为 1 的维度。", 12),
    "unsqueeze": ("Tensor.unsqueeze", "在指定位置插入长度为 1 的维度，便于广播或补 batch/channel 维。", 12),
    "masked_fill": ("Tensor.masked_fill", "在 mask=True 的位置填入指定值，注意力中常填负无穷。", 12),
    "argmax": ("Tensor.argmax", "返回最大值所在下标，贪心解码通常沿词表维调用。", 12),
    "any": ("Tensor.any", "判断指定范围内是否至少有一个 True。", 12),
    "abs": ("Tensor.abs", "逐元素取绝对值。", 12),
    "max": ("Tensor.max", "返回最大值；指定 dim 时还可获得对应下标。", 12),
    "double": ("Tensor.double", "把 Tensor 转为 float64，本题中常用于把布尔正确性转换为可求均值的数值。", 12),
    "tolist": ("Tensor.tolist", "把 CPU Tensor 转为嵌套 Python 列表，便于返回或序列化。", 15),
    "backward": ("Tensor.backward", "从标量结果开始反向传播，把梯度累积到叶子 Tensor 的 .grad。", 10),
    "append": ("list.append", "把一个元素追加到列表末尾。", 32),
    "pop": ("list.pop", "删除并返回指定位置的元素；不传下标时处理末尾元素。", 32),
    "values": ("dict.values", "返回字典中所有值的动态视图。", 32),
    "items": ("dict.items", "返回键值对视图，适合同时遍历键和值。", 32),
    "startswith": ("str.startswith", "判断字符串是否以指定前缀开头。", 32),
    "shuffle": ("Random.shuffle", "原地打乱列表；使用独立 Random 实例可保证固定 seed 的可复现性。", 20),
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def extract_function_notes(solution_code: str, limit: int = 8) -> list[dict[str, str]]:
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])

    notes: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    for index, node in enumerate(ast.walk(tree)):
        if not isinstance(node, ast.Call):
            continue
        raw_name = _call_name(node.func)
        if not raw_name:
            continue
        note = EXACT_FUNCTION_NOTES.get(raw_name)
        if note is None and "." in raw_name:
            note = METHOD_FUNCTION_NOTES.get(raw_name.rsplit(".", 1)[-1])
        if note is None:
            if raw_name.startswith("_"):
                continue
            note = (raw_name, "参考实现中调用的辅助函数或构造器；复习时重点确认它的输入、输出和返回类型。", 80)
        display_name, description, priority = note
        if display_name in seen:
            continue
        seen.add(display_name)
        notes.append((priority, index, display_name, description))

    if not notes:
        return [
            {
                "name": "Python / Tensor 基础运算",
                "description": "本题主要通过索引、比较、广播或算术运算完成；复习时应重点检查 shape 与边界条件。",
            }
        ]

    notes.sort(key=lambda item: (item[0], item[1]))
    return [{"name": name, "description": description} for _, _, name, description in notes[:limit]]


def _append_unique(items: list[str], value: str) -> None:
    cleaned = value.strip()
    if cleaned and cleaned not in items:
        items.append(cleaned)


def build_problem_lecture(
    *,
    slug: str,
    explanation: str,
    presentation: dict[str, Any] | None,
    solution_code: str,
    categories: list[str],
) -> dict[str, Any]:
    presentation = presentation or {}
    solution: list[str] = []
    _append_unique(solution, explanation)
    for step in presentation.get("steps", []):
        _append_unique(solution, str(step))
    _append_unique(solution, "实现后先用默认样例核对返回类型、Tensor 形状和边界条件，再提交隐藏测试。")

    category = categories[0] if categories else ""
    memory_tips: list[str] = []
    if presentation.get("formulas"):
        memory_tips.append("先合上讲义默写核心公式，再逐项说出每个符号、维度和数值稳定性处理。")
    else:
        memory_tips.append("先用一句话复述输入、输出和核心变换，再开始写代码。")

    if any(keyword in category for keyword in ("Attention", "大模型", "多模态", "计算机视觉", "自然语言")):
        memory_tips.append("在纸上标注每一步的 Tensor shape，并说明哪一维执行归一化、转置或聚合。")
    elif any(keyword in category for keyword in ("优化器", "训练")):
        memory_tips.append("记住“状态初始化 → 梯度/统计量计算 → 参数更新”的顺序，并手算一步更新结果。")
    elif "传统机器学习" in category:
        memory_tips.append("用一个极小数据集手算距离、概率或指标，确认并列和空输入等规则。")
    else:
        memory_tips.append("先写最直接版本，再检查是否能用 PyTorch 广播或向量化减少循环。")

    memory_tips.extend(
        [
            "保留默认样例，再补一个最小边界样例和一个容易出错的反例，分别运行验证。",
            "按 1 天、3 天、7 天的间隔重新手写；卡住时只看函数名提示，不直接看参考代码。",
        ]
    )

    return {
        "solution": solution,
        "functions": extract_function_notes(solution_code),
        "memory_tips": memory_tips,
        "solutions": build_solution_variants(slug, solution_code),
    }
