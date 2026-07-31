from __future__ import annotations

from typing import Any


# 只有当高阶 API 基本包办了题目的核心算法时，才在讲义里补充手搓版本。
# torch.as_tensor、基础索引、广播和逐元素运算不视为“直接调 API 解题”。
# 手搓版优先直译题面公式；只有测试和题意明确要求时才保留数值稳定或边界分支。
MANUAL_SOLUTIONS: dict[str, dict[str, Any]] = {
    "batch-cosine-similarity": {
        "apis": ["torch.linalg.vector_norm"],
        "code": """import torch

def batch_cosine_similarity(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    dot = (a * b).sum(dim=1)
    norm_a = torch.sqrt((a * a).sum(dim=1))
    norm_b = torch.sqrt((b * b).sum(dim=1))
    denom = norm_a * norm_b
    out = torch.zeros(a.shape[0], dtype=torch.float64)
    valid = denom != 0
    out[valid] = dot[valid] / denom[valid]
    return out
""",
    },
    "one-hot-encoding": {
        "apis": ["torch.nn.functional.one_hot"],
        "code": """import torch

def one_hot(indices, num_classes):
    indices = torch.as_tensor(indices, dtype=torch.long)
    out = torch.zeros((indices.numel(), num_classes), dtype=torch.long)
    rows = torch.arange(indices.numel())
    out[rows, indices] = 1
    return out
""",
    },
    "relu-activation": {
        "apis": ["torch.relu"],
        "code": """import torch

def relu(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    out = x.clone()
    out[out < 0] = 0
    return out
""",
    },
    "sigmoid-activation": {
        "apis": ["torch.sigmoid"],
        "code": """import torch

def sigmoid(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return 1 / (1 + torch.exp(-x))
""",
    },
    "row-wise-softmax": {
        "apis": ["torch.softmax"],
        "code": """import torch

def softmax_2d(logits):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp_values = torch.exp(shifted)
    return exp_values / exp_values.sum(dim=1, keepdim=True)
""",
    },
    "cross-entropy-loss": {
        "apis": ["torch.nn.functional.cross_entropy"],
        "code": """import torch

def cross_entropy_loss(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_partition = torch.log(torch.exp(shifted).sum(dim=1))
    target_logits = shifted[torch.arange(labels.numel()), labels]
    return float((log_partition - target_logits).mean())
""",
    },
    "conv1d-valid": {
        "apis": ["torch.nn.functional.conv1d"],
        "code": """import torch

def conv1d_valid(x, kernel):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    kernel = torch.as_tensor(kernel, dtype=torch.float64).flatten()
    output_length = x.numel() - kernel.numel() + 1
    values = [
        (x[start:start + kernel.numel()] * kernel).sum()
        for start in range(output_length)
    ]
    return torch.stack(values)
""",
    },
    "torch-add-relu": {
        "apis": ["torch.relu"],
        "code": """import torch

def tensor_add_relu(a, b):
    out = (a + b).clone()
    out[out < 0] = 0
    return out
""",
    },
    "causal-mask": {
        "apis": ["torch.triu"],
        "code": """import torch

def causal_mask(size):
    rows = torch.arange(size).reshape(-1, 1)
    cols = torch.arange(size).reshape(1, -1)
    return cols > rows
""",
    },
    "scaled-dot-product-attention": {
        "apis": ["torch.softmax"],
        "code": """import math
import torch

def scaled_dot_product_attention(q, k, v):
    q = torch.as_tensor(q, dtype=torch.float64)
    k = torch.as_tensor(k, dtype=torch.float64)
    v = torch.as_tensor(v, dtype=torch.float64)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(shifted)
    weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
    return weights @ v
""",
    },
    "attention-mask-softmax": {
        "apis": ["torch.softmax"],
        "code": """import torch

def attention_with_mask(scores, mask):
    scores = torch.as_tensor(scores, dtype=torch.float64)
    mask = torch.as_tensor(mask, dtype=torch.bool)
    masked_scores = scores.clone()
    masked_scores[mask] = float('-inf')
    shifted = masked_scores - masked_scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(shifted)
    return exp_scores / exp_scores.sum(dim=-1, keepdim=True)
""",
    },
    "conv2d-valid": {
        "apis": ["torch.nn.functional.conv2d"],
        "code": """import torch

def conv2d_valid(image, kernel):
    image = torch.as_tensor(image, dtype=torch.float64)
    kernel = torch.as_tensor(kernel, dtype=torch.float64)
    out_h = image.shape[0] - kernel.shape[0] + 1
    out_w = image.shape[1] - kernel.shape[1] + 1
    rows = []
    for i in range(out_h):
        values = []
        for j in range(out_w):
            window = image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            values.append((window * kernel).sum())
        rows.append(torch.stack(values))
    return torch.stack(rows)
""",
    },
    "max-pool2d": {
        "apis": ["torch.nn.functional.max_pool2d"],
        "code": """import torch

def max_pool2d(image, kernel_size, stride):
    image = torch.as_tensor(image, dtype=torch.float64)
    out_h = (image.shape[0] - kernel_size) // stride + 1
    out_w = (image.shape[1] - kernel_size) // stride + 1
    rows = []
    for i in range(out_h):
        values = []
        for j in range(out_w):
            top = i * stride
            left = j * stride
            window = image[top:top + kernel_size, left:left + kernel_size]
            values.append(window.max())
        rows.append(torch.stack(values))
    return torch.stack(rows)
""",
    },
    "patch-embedding-flatten": {
        "apis": ["Tensor.unfold"],
        "code": """import torch

def patch_embedding_flatten(image, patch_size):
    image = torch.as_tensor(image, dtype=torch.float64)
    patches = [
        image[top:top + patch_size, left:left + patch_size].reshape(-1)
        for top in range(0, image.shape[0], patch_size)
        for left in range(0, image.shape[1], patch_size)
    ]
    return torch.stack(patches)
""",
    },
    "sequence-cross-entropy-ignore-pad": {
        "apis": ["torch.nn.functional.cross_entropy"],
        "code": """import torch

def sequence_cross_entropy(logits, labels, pad_id):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    valid = labels != pad_id
    if not bool(valid.any()):
        return 0.0
    valid_logits = logits[valid]
    valid_labels = labels[valid]
    shifted = valid_logits - valid_logits.max(dim=1, keepdim=True).values
    log_partition = torch.log(torch.exp(shifted).sum(dim=1))
    target_logits = shifted[torch.arange(valid_labels.numel()), valid_labels]
    return float((log_partition - target_logits).mean())
""",
    },
    "detect-nan": {
        "apis": ["torch.isnan"],
        "code": """import torch

def has_nan(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    return bool((values != values).any())
""",
    },
    "pairwise-euclidean-distance": {
        "apis": ["torch.cdist"],
        "code": """import torch

def pairwise_euclidean_distance(x, y):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    differences = x[:, None, :] - y[None, :, :]
    squared_distance = (differences * differences).sum(dim=-1)
    return torch.sqrt(squared_distance)
""",
    },
    "binary-cross-entropy-logits": {
        "apis": ["torch.nn.functional.binary_cross_entropy_with_logits"],
        "code": """import torch

def binary_cross_entropy_logits(logits, targets):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.float64)
    positive = torch.where(logits > 0, logits, torch.zeros_like(logits))
    loss = positive - logits * targets + torch.log1p(torch.exp(-torch.abs(logits)))
    return float(loss.mean())
""",
    },
    "binary-focal-loss": {
        "apis": ["torch.nn.functional.binary_cross_entropy_with_logits", "torch.sigmoid"],
        "code": """import torch

def binary_focal_loss(logits, targets, alpha, gamma):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.float64)
    prob = 1 / (1 + torch.exp(-logits))
    pt = targets * prob + (1 - targets) * (1 - prob)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    return float((-alpha_t * (1 - pt).pow(gamma) * torch.log(pt)).mean())
""",
    },
    "dice-loss": {
        "apis": ["torch.sigmoid"],
        "code": """import torch

def dice_loss(logits, targets, eps=1e-6):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.float64)
    prob = 1 / (1 + torch.exp(-logits))
    intersection = (prob * targets).sum()
    dice = (2 * intersection + eps) / (prob.sum() + targets.sum() + eps)
    return float(1 - dice)
""",
    },
    "swiglu-activation": {
        "apis": ["torch.nn.functional.silu"],
        "code": """import torch

def swiglu(gate, value):
    gate = torch.as_tensor(gate, dtype=torch.float64)
    value = torch.as_tensor(value, dtype=torch.float64)
    sigmoid = 1 / (1 + torch.exp(-gate))
    return gate * sigmoid * value
""",
    },
    "lora-linear-forward": {
        "apis": ["torch.nn.functional.linear"],
        "code": """import torch

def lora_linear(x, weight, a, b, alpha):
    x = torch.as_tensor(x, dtype=torch.float64)
    weight = torch.as_tensor(weight, dtype=x.dtype)
    a = torch.as_tensor(a, dtype=x.dtype)
    b = torch.as_tensor(b, dtype=x.dtype)
    base = x @ weight.transpose(-2, -1)
    low_rank = (x @ a.transpose(-2, -1)) @ b.transpose(-2, -1)
    return base + (alpha / a.shape[0]) * low_rank
""",
    },
    "top-p-sampling-candidates": {
        "apis": ["torch.argsort", "torch.cumsum", "torch.searchsorted"],
        "code": """def top_p_candidates(probs, p):
    order = sorted(range(len(probs)), key=lambda index: -probs[index])
    selected = []
    cumulative = 0.0
    for index in order:
        selected.append(index)
        cumulative += probs[index]
        if cumulative >= p:
            break
    return selected
""",
    },
    "info-nce-loss": {
        "apis": ["torch.nn.functional.normalize", "torch.nn.functional.cross_entropy"],
        "code": """import torch

def info_nce_loss(a, b, temperature):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    norm_a = torch.sqrt((a * a).sum(dim=1, keepdim=True))
    norm_b = torch.sqrt((b * b).sum(dim=1, keepdim=True))
    logits = (a / norm_a) @ (b / norm_b).transpose(0, 1) / temperature
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(shifted)
    positive = exp_logits.diagonal()
    return float(-torch.log(positive / exp_logits.sum(dim=1)).mean())
""",
    },
    "clip-symmetric-loss": {
        "apis": ["torch.nn.functional.normalize", "torch.nn.functional.cross_entropy"],
        "code": """import torch

def clip_symmetric_loss(image_embeddings, text_embeddings, temperature):
    image = torch.as_tensor(image_embeddings, dtype=torch.float64)
    text = torch.as_tensor(text_embeddings, dtype=torch.float64)
    image_norm = torch.sqrt((image * image).sum(dim=1, keepdim=True))
    text_norm = torch.sqrt((text * text).sum(dim=1, keepdim=True))
    logits = (image / image_norm) @ (text / text_norm).transpose(0, 1) / temperature

    def diagonal_cross_entropy(scores):
        shifted = scores - scores.max(dim=1, keepdim=True).values
        exp_scores = torch.exp(shifted)
        probabilities = exp_scores.diagonal() / exp_scores.sum(dim=1)
        return -torch.log(probabilities).mean()

    image_loss = diagonal_cross_entropy(logits)
    text_loss = diagonal_cross_entropy(logits.transpose(0, 1))
    return float((image_loss + text_loss) / 2)
""",
    },
    "sft-shifted-loss": {
        "apis": ["torch.nn.functional.cross_entropy"],
        "code": """import torch

def sft_shifted_loss(logits, input_ids, ignore_index=-100):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(input_ids, dtype=torch.long)
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    valid = shift_labels != ignore_index
    if not bool(valid.any()):
        return 0.0
    scores = shift_logits[valid]
    targets = shift_labels[valid]
    shifted = scores - scores.max(dim=1, keepdim=True).values
    log_denominator = torch.log(torch.exp(shifted).sum(dim=1))
    target_score = shifted.gather(1, targets.unsqueeze(1)).squeeze(1)
    return float((log_denominator - target_score).mean())
""",
    },
    "knowledge-distillation-kl": {
        "apis": ["torch.nn.functional.log_softmax", "torch.nn.functional.softmax", "torch.nn.functional.kl_div"],
        "code": """import torch

def distillation_kl_loss(student_logits, teacher_logits, temperature):
    student = torch.as_tensor(student_logits, dtype=torch.float64) / temperature
    teacher = torch.as_tensor(teacher_logits, dtype=torch.float64) / temperature

    def softmax(x):
        exp_x = torch.exp(x - x.max(dim=-1, keepdim=True).values)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    p = softmax(student)
    q = softmax(teacher)
    kl = (q * torch.log(q / p)).sum() / student.shape[0]
    return float(kl * temperature ** 2)
""",
    },
    "top-k-accuracy": {
        "apis": ["torch.topk"],
        "code": """import torch

def top_k_accuracy(logits, labels, k):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    correct = 0
    for row, label in zip(logits, labels):
        order = sorted(range(row.numel()), key=lambda index: (-float(row[index]), index))
        if int(label) in order[:k]:
            correct += 1
    return correct / logits.shape[0]
""",
    },
}


def build_solution_variants(slug: str, api_code: str) -> list[dict[str, str]]:
    manual = MANUAL_SOLUTIONS.get(slug)
    if manual is None:
        return [
            {
                "kind": "standard",
                "title": "参考实现",
                "summary": "这道题没有用高阶 API 直接包办核心逻辑。",
                "code": api_code.strip() + "\n",
            }
        ]

    api_names = "、".join(manual["apis"])
    return [
        {
            "kind": "manual",
            "title": "手搓实现",
            "summary": f"按公式和基础 Tensor 运算展开，不调用 {api_names}。",
            "code": str(manual["code"]).strip() + "\n",
        },
        {
            "kind": "api",
            "title": "API 实现",
            "summary": f"使用 {api_names} 简化实现，工程中通常更短也更稳健。",
            "code": api_code.strip() + "\n",
        },
    ]
