from __future__ import annotations

import math
import json
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


def formula(latex: str, label: str | None = None) -> dict[str, str]:
    item = {"latex": latex}
    if label:
        item["label"] = label
    return item


PRESENTATIONS: dict[str, dict[str, list[Any]]] = {
    "matrix-transpose": {
        "formulas": [formula(r"B_{j,i}=A_{i,j}")],
        "symbols": ["A 的形状为 m×n，转置结果 B 的形状为 n×m。"],
        "steps": ["交换行、列下标。", "空矩阵直接返回空结果。"],
    },
    "l2-normalize-vector": {
        "formulas": [formula(r"\lVert x\rVert_2=\sqrt{\sum_{j=1}^{d}x_j^2}"), formula(r"\hat{x}_i=\frac{x_i}{\lVert x\rVert_2}")],
        "symbols": ["x 是 d 维输入向量；零向量按题意返回全 0。"],
        "steps": ["计算 L2 范数。", "处理零范数，否则逐元素除以范数。"],
    },
    "numpy-broadcast-add": {
        "formulas": [formula(r"C_{i,j}=A_{i,j}+b_j", "按末尾维度广播的典型情形")],
        "symbols": ["实际输入遵循 PyTorch 广播规则：从末尾维开始比较，维度相等、为 1 或不存在时可广播。"],
        "steps": ["把输入转换为 Tensor。", "利用 PyTorch 广播完成逐元素相加。"],
    },
    "batch-cosine-similarity": {
        "formulas": [formula(r"s_i=\frac{a_i^\top b_i}{\lVert a_i\rVert_2\lVert b_i\rVert_2}")],
        "symbols": ["a_i、b_i 是第 i 个样本；任一向量范数为 0 时相似度记为 0。"],
        "steps": ["逐行计算点积和两侧范数。", "仅在分母非零处执行除法。"],
    },
    "stable-softmax": {
        "formulas": [formula(r"p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\qquad m=\max_j x_j")],
        "symbols": ["减去最大值不会改变 Softmax 结果，却能避免指数溢出。"],
        "steps": ["求最大值并平移 logits。", "计算指数，再除以指数和。"],
    },
    "log-sum-exp": {
        "formulas": [formula(r"\operatorname{LSE}(x)=m+\log\sum_i e^{x_i-m},\qquad m=\max_i x_i")],
        "symbols": ["m 是输入最大值，用于提升数值稳定性。"],
        "steps": ["提出最大值。", "对平移后的指数求和并取对数。"],
    },
    "one-hot-encoding": {
        "formulas": [formula(r"Y_{i,c}=\begin{cases}1,&c=y_i\\0,&c\ne y_i\end{cases}")],
        "symbols": ["y_i 是第 i 个类别下标，c 的范围是 0 到 num_classes−1。"],
        "steps": ["创建形状为 (N, C) 的全零 Tensor。", "在每行的目标类别位置写入 1。"],
    },
    "sliding-window-sum": {
        "formulas": [formula(r"s_i=\sum_{j=0}^{w-1}x_{i+j}")],
        "symbols": ["w 是窗口宽度；输出长度为 n−w+1。"],
        "steps": ["可用前缀和或一维卷积计算。", "窗口非法时返回空结果。"],
    },
    "linear-regression-predict": {
        "formulas": [formula(r"\hat{y}=Xw+b")],
        "symbols": ["X 为批量特征矩阵，w 为权重向量，b 为标量偏置。"],
        "steps": ["计算矩阵与权重向量的乘积。", "对每个样本加上偏置。"],
    },
    "sigmoid-probabilities": {
        "formulas": [formula(r"\sigma(x)=\frac{1}{1+e^{-x}}")],
        "symbols": ["对负数使用等价形式 e^x/(1+e^x) 可避免溢出。"],
        "steps": ["按数值正负选择稳定形式。", "逐元素返回 (0,1) 内的概率。"],
    },
    "knn-majority-vote": {
        "formulas": [formula(r"d(x,q)^2=\sum_j(x_j-q_j)^2"), formula(r"\hat y=\operatorname{mode}\{y_i:i\in\operatorname{TopKSmallest}(d_i)\}")],
        "symbols": ["距离相同时按样本原顺序；票数相同时返回较小标签。"],
        "steps": ["计算查询点到所有训练点的距离。", "取最近 k 个标签并多数投票。"],
    },
    "kmeans-assign": {
        "formulas": [formula(r"z_i=\arg\min_k\lVert x_i-\mu_k\rVert_2^2")],
        "symbols": ["μ_k 是第 k 个聚类中心；距离相同选择较小下标。"],
        "steps": ["计算每个样本到每个中心的平方距离。", "逐行取最小距离对应的中心。"],
    },
    "information-entropy": {
        "formulas": [formula(r"H(Y)=-\sum_c p_c\log_2 p_c")],
        "symbols": ["p_c 是类别 c 在标签中的频率；空输入的熵定义为 0。"],
        "steps": ["统计各类别频率。", "累加 −p·log₂p。"],
    },
    "gini-index": {
        "formulas": [formula(r"G(Y)=1-\sum_c p_c^2")],
        "symbols": ["p_c 是类别 c 的经验概率。"],
        "steps": ["统计类别概率。", "用 1 减去概率平方和。"],
    },
    "standardize-feature": {
        "formulas": [formula(r"\mu=\frac1N\sum_i x_i,\qquad \sigma=\sqrt{\frac1N\sum_i(x_i-\mu)^2}"), formula(r"z_i=\frac{x_i-\mu}{\sigma}")],
        "symbols": ["这里使用总体方差（unbiased=False）；标准差为 0 时返回全 0。"],
        "steps": ["计算均值和总体标准差。", "处理零标准差，否则执行标准化。"],
    },
    "precision-recall-f1": {
        "formulas": [formula(r"P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN}"), formula(r"F_1=\frac{2PR}{P+R}")],
        "symbols": ["分母为 0 时，对应指标按题意取 0。"],
        "steps": ["统计 TP、FP、FN。", "依次计算 Precision、Recall 和 F1。"],
    },
    "gradient-descent-step": {
        "formulas": [formula(r"L=\frac1N\sum_i(wx_i+b-y_i)^2"), formula(r"w\leftarrow w-\eta\frac{\partial L}{\partial w},\qquad b\leftarrow b-\eta\frac{\partial L}{\partial b}")],
        "symbols": ["η 是学习率 lr。"],
        "steps": ["计算预测和 MSE 对 w、b 的梯度。", "沿负梯度方向更新一次。"],
    },
    "relu-activation": {
        "formulas": [formula(r"\operatorname{ReLU}(x)=\max(0,x)")],
        "symbols": ["逐元素应用。"],
        "steps": ["把输入转换为 Tensor。", "使用 torch.relu 返回结果。"],
    },
    "sigmoid-activation": {
        "formulas": [formula(r"\sigma(x)=\frac{1}{1+e^{-x}}")],
        "symbols": ["逐元素应用。"],
        "steps": ["把输入转换为浮点 Tensor。", "使用 torch.sigmoid 计算。"],
    },
    "row-wise-softmax": {
        "formulas": [formula(r"p_{i,j}=\frac{e^{x_{i,j}-m_i}}{\sum_k e^{x_{i,k}-m_i}},\qquad m_i=\max_k x_{i,k}")],
        "symbols": ["每一行是一个独立样本，沿最后一维归一化。"],
        "steps": ["逐行减去最大值。", "沿最后一维计算 Softmax。"],
    },
    "cross-entropy-loss": {
        "formulas": [formula(r"L=-\frac1N\sum_{i=1}^N\log\frac{e^{z_{i,y_i}}}{\sum_c e^{z_{i,c}}}")],
        "symbols": ["z 是未经 Softmax 的 logits，y_i 是目标类别下标。"],
        "steps": ["直接对 logits 使用稳定的 log_softmax。", "取目标类别负对数似然并求平均。"],
    },
    "mean-squared-error": {
        "formulas": [formula(r"L=\frac1N\sum_{i=1}^N(\hat y_i-y_i)^2")],
        "symbols": ["N 表示所有元素的数量。"],
        "steps": ["计算逐元素误差平方。", "对全部元素取平均。"],
    },
    "dropout-forward-train": {
        "formulas": [formula(r"y_i=\frac{m_i x_i}{1-p},\qquad m_i\sim\operatorname{Bernoulli}(1-p)")],
        "symbols": ["p 是丢弃概率；题目直接给定 0/1 mask，因此结果可复现。"],
        "steps": ["应用给定 mask。", "除以保留概率 1−p 维持期望不变。"],
    },
    "batch-norm-forward": {
        "formulas": [formula(r"\mu_j=\frac1N\sum_i x_{i,j},\qquad \sigma_j^2=\frac1N\sum_i(x_{i,j}-\mu_j)^2"), formula(r"y_{i,j}=\gamma_j\frac{x_{i,j}-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}}+\beta_j")],
        "symbols": ["统计量沿 batch 维计算；方差使用 biased 估计（unbiased=False）。"],
        "steps": ["沿第 0 维求均值和方差并保留维度。", "归一化后应用可学习缩放 γ 和偏置 β。"],
    },
    "layer-norm-forward": {
        "formulas": [formula(r"\mu_i=\frac1D\sum_j x_{i,j},\qquad \sigma_i^2=\frac1D\sum_j(x_{i,j}-\mu_i)^2"), formula(r"y_{i,j}=\gamma_j\frac{x_{i,j}-\mu_i}{\sqrt{\sigma_i^2+\varepsilon}}+\beta_j")],
        "symbols": ["统计量对每个样本独立地沿特征维计算。"],
        "steps": ["沿最后一维计算均值和总体方差。", "归一化后应用 γ 和 β。"],
    },
    "conv1d-valid": {
        "formulas": [formula(r"y_i=\sum_{j=0}^{K-1}x_{i+j}w_j,\qquad i=0,\ldots,L-K")],
        "symbols": ["采用深度学习中的互相关定义，不翻转卷积核；无 padding、stride=1。"],
        "steps": ["滑动长度为 K 的窗口。", "窗口与核逐元素相乘后求和。"],
    },
    "autograd-square-grad": {
        "formulas": [formula(r"y=\sum_i x_i^2,\qquad \frac{\partial y}{\partial x_i}=2x_i")],
        "symbols": ["需要让输入 Tensor 开启 requires_grad。"],
        "steps": ["构建标量损失 y。", "调用 backward 并返回 x.grad。"],
    },
    "torch-no-grad-update": {
        "formulas": [formula(r"\theta\leftarrow\theta-\eta g")],
        "symbols": ["更新过程不应被 autograd 记录。"],
        "steps": ["进入 torch.no_grad 上下文。", "用学习率乘梯度并更新参数。"],
    },
    "gradient-accumulation": {
        "formulas": [formula(r"g=\frac1N\sum_{b}\sum_{i\in b}2(wx_i-y_i)x_i,\qquad w\leftarrow w-\eta g")],
        "symbols": ["N 是所有 micro-batch 的样本总数，不是 batch 数。"],
        "steps": ["累加所有样本的梯度和数量。", "统一取平均并只更新一次。"],
    },
    "causal-mask": {
        "formulas": [formula(r"M_{i,j}=\mathbb{1}[j>i]")],
        "symbols": ["True 表示需要屏蔽，即每个位置不能关注未来 token。"],
        "steps": ["创建方阵。", "取主对角线上方的严格上三角区域。"],
    },
    "custom-mse-loss": {
        "formulas": [formula(r"L=\operatorname{mean}\left((\hat y-y)^2\right)")],
        "symbols": ["必须保留 autograd 计算图。"],
        "steps": ["用 Tensor 运算计算差的平方。", "对全部元素求均值。"],
    },
    "optimizer-step-list": {
        "formulas": [formula(r"\theta_i\leftarrow\theta_i-\eta g_i")],
        "symbols": ["params 与 grads 一一对应。"],
        "steps": ["配对遍历参数和梯度。", "返回更新后的 Tensor 列表。"],
    },
    "scaled-dot-product-attention": {
        "formulas": [formula(r"\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V")],
        "symbols": ["d_k 是 Query/Key 的最后一维；Softmax 沿 Key 的序列维执行。"],
        "steps": ["计算 Q 与 K 转置的缩放点积。", "沿最后一维做 Softmax。", "注意力权重乘 V。"],
    },
    "attention-mask-softmax": {
        "formulas": [formula(r"P_{i,:}=\operatorname{softmax}(S_{i,:}+M_{i,:}),\qquad M_{i,j}=\begin{cases}-\infty,&\text{masked}\\0,&\text{otherwise}\end{cases}")],
        "symbols": ["输入 mask=True 的位置不可被关注。"],
        "steps": ["在被屏蔽位置填入负无穷。", "沿最后一维做稳定 Softmax。"],
    },
    "sinusoidal-positional-encoding": {
        "formulas": [formula(r"PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)", "偶数维"), formula(r"PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)", "奇数维")],
        "symbols": ["pos=0,…,length−1 是位置；i=0,… 是频率下标；d_model=dim 是编码维数。", "dim 为奇数时，最后一个偶数维仍计算 sin，没有与之配对的 cos 维。"],
        "steps": ["构造位置列向量 pos 和偶数维下标。", "一次性计算不同频率的角度矩阵。", "偶数列写入 sin，存在的奇数列写入 cos。"],
    },
    "split-heads": {
        "formulas": [formula(r"(B,L,D)\rightarrow(B,L,H,D/H)\rightarrow(B,H,L,D/H)")],
        "symbols": ["B 为 batch，L 为序列长度，H 为头数，D 必须能被 H 整除。"],
        "steps": ["把最后一维 reshape 为 (H, D/H)。", "交换序列维与头维。"],
    },
    "combine-heads": {
        "formulas": [formula(r"(B,H,L,D_h)\rightarrow(B,L,H,D_h)\rightarrow(B,L,HD_h)")],
        "symbols": ["这是 split_heads 的逆变换。"],
        "steps": ["把头维换回序列维之后。", "确保 contiguous，再合并头维和 head_dim。"],
    },
    "label-smoothing": {
        "formulas": [formula(r"q_c=(1-\varepsilon)\mathbb{1}[c=y]+\frac{\varepsilon}{C}")],
        "symbols": ["C 是类别数；目标类概率为 1−ε+ε/C，其他类为 ε/C。"],
        "steps": ["先用 ε/C 填满分布。", "给目标类别额外加上 1−ε。"],
    },
    "conv2d-valid": {
        "formulas": [formula(r"Y_{i,j}=\sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1}X_{i+u,j+v}W_{u,v}")],
        "symbols": ["采用互相关定义；无 padding、stride=1。"],
        "steps": ["提取所有有效滑动窗口。", "窗口与卷积核逐元素相乘并求和。"],
    },
    "max-pool2d": {
        "formulas": [formula(r"Y_{i,j}=\max_{0\le u,v<K}X_{is+u,js+v}")],
        "symbols": ["K 是 kernel_size，s 是 stride。"],
        "steps": ["按 stride 提取 K×K 窗口。", "对每个窗口取最大值。"],
    },
    "box-iou": {
        "formulas": [formula(r"\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{I}{|A|+|B|-I}")],
        "symbols": ["框格式为 [x₁,y₁,x₂,y₂]；无交集时 I=0。"],
        "steps": ["计算交集矩形宽高并截断到非负。", "计算交集、并集及其比值。"],
    },
    "nms": {
        "formulas": [formula(r"\text{保留 }i\iff s_i\text{ 当前最大，随后删除 }\{j:\operatorname{IoU}(b_i,b_j)>\tau\}")],
        "symbols": ["s_i 是置信度，τ 是 IoU 阈值；分数相同优先较小原下标。"],
        "steps": ["按分数降序排列候选框。", "反复保留最高分框。", "移除与其 IoU 超过阈值的剩余框。"],
    },
    "sequence-cross-entropy-ignore-pad": {
        "formulas": [formula(r"L=-\frac{1}{\sum_{b,t}m_{b,t}}\sum_{b,t}m_{b,t}\log p_{b,t,y_{b,t}},\qquad m_{b,t}=\mathbb{1}[y_{b,t}\ne pad\_id]")],
        "symbols": ["只对非 Padding token 求平均；若没有有效 token，返回 0。"],
        "steps": ["把 logits 展平并计算逐 token 交叉熵。", "用标签掩码筛掉 pad_id。", "对有效损失取平均。"],
    },
    "count-parameters": {
        "formulas": [formula(r"N_{\text{params}}=\sum_{t}\prod_{d\in\operatorname{shape}(t)}d")],
        "symbols": ["每个 Tensor 的参数量等于其各维长度的乘积。"],
        "steps": ["对每个 shape 求维度乘积。", "累加所有 Tensor 的参数量。"],
    },
    "pairwise-euclidean-distance": {
        "formulas": [formula(r"D_{i,j}=\lVert x_i-y_j\rVert_2=\sqrt{\sum_k(x_{i,k}-y_{j,k})^2}")],
        "symbols": ["输出形状为 (N,M)，结果需要截断微小负数后再开方。"],
        "steps": ["利用广播或平方范数恒等式构造两两平方距离。", "clamp 到非负后开方。"],
    },
    "binary-cross-entropy-logits": {
        "formulas": [formula(r"\ell(x,y)=\max(x,0)-xy+\log(1+e^{-|x|})")],
        "symbols": ["x 是 logits，y∈{0,1}；返回所有元素的平均损失。"],
        "steps": ["直接使用稳定的 binary_cross_entropy_with_logits。", "不要先手写 Sigmoid 再取对数。"],
    },
    "binary-focal-loss": {
        "formulas": [formula(r"FL=-\alpha_t(1-p_t)^\gamma\log p_t")],
        "symbols": ["p_t 是真实类别的预测概率；正类 α_t=α，负类 α_t=1−α。"],
        "steps": ["从 logits 稳定计算逐元素 BCE。", "构造 p_t 和 α_t，加权后取平均。"],
    },
    "dice-loss": {
        "formulas": [formula(r"L_{Dice}=1-\frac{2\sum_i p_i y_i+\varepsilon}{\sum_i p_i+\sum_i y_i+\varepsilon}")],
        "symbols": ["p 是 sigmoid(logits)；本题在整个 batch 上计算一个 Dice。"],
        "steps": ["对 logits 做 Sigmoid。", "计算交集和分母并返回 1−Dice。"],
    },
    "rms-norm": {
        "formulas": [formula(r"\operatorname{RMS}(x)=\sqrt{\frac1D\sum_{j=1}^D x_j^2+\varepsilon}"), formula(r"y_j=\gamma_j\frac{x_j}{\operatorname{RMS}(x)}")],
        "symbols": ["沿最后一维归一化；RMSNorm 不减均值。"],
        "steps": ["沿最后一维计算均方根。", "归一化后乘逐维权重 γ。"],
    },
    "swiglu-activation": {
        "formulas": [formula(r"\operatorname{SwiGLU}(a,b)=\operatorname{SiLU}(a)\odot b,\qquad \operatorname{SiLU}(a)=a\sigma(a)")],
        "symbols": ["a 与 b 形状相同，⊙ 表示逐元素乘法。"],
        "steps": ["对 gate 输入应用 SiLU。", "与 value 输入逐元素相乘。"],
    },
    "rotary-position-embedding": {
        "formulas": [formula(r"\theta_{pos,i}=\frac{pos}{base^{2i/d}}"), formula(r"\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}")],
        "symbols": ["输入形状为 (seq, dim)，dim 必须为偶数；base 默认 10000。"],
        "steps": ["按位置和维度对生成旋转角。", "把相邻偶/奇维成对旋转。"],
    },
    "lora-linear-forward": {
        "formulas": [formula(r"Y=XW^\top+\frac{\alpha}{r}(XA^\top)B^\top")],
        "symbols": ["W∈R^{o×d}，A∈R^{r×d}，B∈R^{o×r}，r 是 LoRA rank。"],
        "steps": ["计算冻结主权重的线性输出。", "计算低秩支路并按 α/r 缩放后相加。"],
    },
    "clip-grad-global-norm": {
        "formulas": [formula(r"G=\sqrt{\sum_k\lVert g_k\rVert_2^2},\qquad g'_k=g_k\min\left(1,\frac{c}{G+\varepsilon}\right)")],
        "symbols": ["c 是 max_norm；所有梯度共享同一个缩放系数。"],
        "steps": ["累加全部梯度的平方和得到全局范数。", "只在超限时按同一比例缩放。"],
    },
    "sgd-momentum-step": {
        "formulas": [formula(r"v_t=\mu v_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\eta v_t")],
        "symbols": ["μ 是 momentum，η 是学习率。"],
        "steps": ["先更新速度。", "再用新速度更新参数。"],
    },
    "adamw-step": {
        "formulas": [formula(r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2"), formula(r"\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\varepsilon}")],
        "symbols": ["λ 是解耦 weight_decay；t=step 从 1 开始。"],
        "steps": ["更新一、二阶矩。", "做偏差修正。", "分别应用权重衰减和自适应梯度更新。"],
    },
    "warmup-cosine-learning-rate": {
        "formulas": [formula(r"\eta_t=\eta_{max}\frac{t+1}{T_w},\quad t<T_w"), formula(r"\eta_t=\eta_{min}+\frac{\eta_{max}-\eta_{min}}2\left(1+\cos(\pi p)\right),\quad p=\frac{t-T_w}{T-T_w-1}")],
        "symbols": ["step=t 从 0 开始；最后一步恰好等于 min_lr。"],
        "steps": ["warmup 阶段线性升高。", "余下阶段按余弦曲线衰减。"],
    },
    "exponential-moving-average": {
        "formulas": [formula(r"\bar\theta_t=\beta\bar\theta_{t-1}+(1-\beta)\theta_t")],
        "symbols": ["β=decay；shadow 是上一步 EMA 参数。"],
        "steps": ["按 decay 保留历史 shadow。", "加入当前参数的 (1−decay) 权重。"],
    },
    "top-p-sampling-candidates": {
        "formulas": [formula(r"k=\min\left\{m:\sum_{i=1}^{m}p_{(i)}\ge p_{nucleus}\right\}")],
        "symbols": ["p_(i) 按概率降序排列；至少保留一个 token。"],
        "steps": ["稳定地按概率降序排序。", "找到累计概率首次达到阈值的位置。", "返回此前所有原下标。"],
    },
    "perplexity-from-token-losses": {
        "formulas": [formula(r"\operatorname{PPL}=\exp\left(\frac1N\sum_{i=1}^{N}\ell_i\right)")],
        "symbols": ["ℓ_i 是每个有效 token 的自然对数交叉熵。"],
        "steps": ["计算 token loss 的均值。", "对均值取指数。"],
    },
    "info-nce-loss": {
        "formulas": [formula(r"s_{i,j}=\frac{\hat a_i^\top\hat b_j}{\tau},\qquad L=-\frac1N\sum_i\log\frac{e^{s_{i,i}}}{\sum_j e^{s_{i,j}}}")],
        "symbols": ["先对两组 embedding 做 L2 归一化；正样本位于对角线。"],
        "steps": ["归一化两组 embedding。", "计算温度缩放的相似度矩阵。", "以对角线下标为标签计算交叉熵。"],
    },
    "knowledge-distillation-kl": {
        "formulas": [formula(r"L=T^2\frac1N\sum_i\operatorname{KL}\left(\operatorname{softmax}(z_i^t/T)\,\Vert\,\operatorname{softmax}(z_i^s/T)\right)")],
        "symbols": ["T 是温度；KL 按 batchmean 约简并乘 T²。"],
        "steps": ["教师 logits 用 softmax，学生 logits 用 log_softmax。", "计算 KL 散度并乘温度平方。"],
    },
    "global-average-pooling": {
        "formulas": [formula(r"y_c=\frac1{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{c,h,w}")],
        "symbols": ["输入形状为 (C,H,W)，输出形状为 (C,)。"],
        "steps": ["沿最后两个空间维求平均。"],
    },
    "top-k-accuracy": {
        "formulas": [formula(r"\operatorname{Acc@k}=\frac1N\sum_i\mathbb{1}\left[y_i\in\operatorname{TopK}(z_i)\right]")],
        "symbols": ["z_i 是第 i 个样本的 logits。"],
        "steps": ["沿类别维取 Top-K 下标。", "判断真实标签是否出现并求平均。"],
    },
    "symmetric-quantize-dequantize": {
        "formulas": [formula(r"s=\frac{\max_i|x_i|}{2^{b-1}-1},\qquad q_i=\operatorname{clip}(\operatorname{round}(x_i/s),-Q,Q)"), formula(r"\hat x_i=sq_i")],
        "symbols": ["b=num_bits，Q=2^{b−1}−1；全零输入直接返回全零。"],
        "steps": ["由最大绝对值计算对称量化 scale。", "舍入并截断到整数范围。", "乘 scale 得到反量化结果。"],
    },
}


def problem_presentation(slug: str, explanation: str) -> dict[str, list[Any]]:
    configured = PRESENTATIONS.get(slug, {})
    return {
        "formulas": configured.get("formulas", []),
        "symbols": configured.get("symbols", []),
        "steps": configured.get("steps", [explanation]),
    }


def ndarray(data: Any, dtype: str = "float", shape: list[int] | None = None) -> dict[str, Any]:
    value = {"__type__": "ndarray", "dtype": dtype, "data": data}
    if shape is not None:
        value["shape"] = shape
    return value


def tensor(data: Any, dtype: str = "float", shape: list[int] | None = None) -> dict[str, Any]:
    value = {"__type__": "tensor", "dtype": dtype, "data": data}
    if shape is not None:
        value["shape"] = shape
    return value


def materialize(value: Any) -> Any:
    if isinstance(value, dict) and "__type__" in value:
        if value["__type__"] == "ndarray":
            dtype = float if value.get("dtype") == "float" else int
            array = np.array(value["data"], dtype=dtype)
            return array.reshape(value["shape"]) if "shape" in value else array
        if value["__type__"] == "tensor":
            dtype = torch.float32 if value.get("dtype") == "float" else torch.long
            value_tensor = torch.tensor(value["data"], dtype=dtype)
            return value_tensor.reshape(value["shape"]) if "shape" in value else value_tensor
        if value["__type__"] == "nan":
            return float("nan")
    if isinstance(value, list):
        return [materialize(item) for item in value]
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ndarray(value.tolist(), "float" if value.dtype.kind == "f" else "int", list(value.shape))
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        dtype = "float" if detached.dtype.is_floating_point else "int"
        return tensor(detached.tolist(), dtype, list(detached.shape))
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
    return json.dumps(value, ensure_ascii=False)


def format_output_value(value: Any) -> str:
    value = to_jsonable(value)
    if isinstance(value, dict) and value.get("__type__") == "tensor":
        return f"torch.tensor({json.dumps(value['data'], ensure_ascii=False)})"
    if isinstance(value, dict) and value.get("__type__") == "ndarray":
        return f"np.array({json.dumps(value['data'], ensure_ascii=False)})"
    return json.dumps(value, ensure_ascii=False)


def display_input(args: list[Any], kwargs: dict[str, Any] | None = None) -> str:
    kwargs = kwargs or {}
    if len(args) == 1 and not kwargs:
        return format_value(args[0])
    if kwargs:
        return format_value({"args": args, "kwargs": kwargs})
    return format_value(args)


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
    company_tags: list[str] | None = None,
    source_note: str | None = None,
    time_limit: float = 2.0,
    memory_limit: int = 256,
) -> dict[str, Any]:
    tests = build_cases(reference, raw_cases)
    public_tests = tests[:3]
    hidden_tests = tests[3:]
    examples = [
        {
            "input": display_input(raw_cases[i]["args"], raw_cases[i].get("kwargs", {})),
            "output": format_output_value(tests[i]["expected"]),
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
        "company_tags": company_tags
        or [COMPANY_TAGS[pid % len(COMPANY_TAGS)], COMPANY_TAGS[(pid * 3) % len(COMPANY_TAGS)]],
        "source_note": source_note
        or "依据公开论文、PyTorch 官方接口与 Applied Scientist / MLE 常见能力范围原创改编；公司标签仅表示相似高频方向，不声明为真实原题。",
        "description": description,
        "function_name": function_name,
        "function_signature": signature,
        "starter_code": starter_code or starter(signature, imports),
        "solution_code": solution_code,
        "explanation": explanation,
        "presentation": problem_presentation(slug, explanation),
        "constraints": constraints,
        "examples": examples,
        "public_tests": public_tests,
        "hidden_tests": hidden_tests,
        "time_limit": time_limit,
        "memory_limit": memory_limit,
    }


def py_imports() -> str:
    return "from typing import Any\nimport math"


def torch_imports() -> str:
    return "from typing import Any\nimport torch"


def get_seed_problems() -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        problems.append(make_problem(pid=len(problems) + 1, **kwargs))

    add(
        slug="matrix-transpose",
        title="矩阵转置",
        difficulty="简单",
        category="Python 与 PyTorch 基础",
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
        category="Python 与 PyTorch 基础",
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
        category="Python 与 PyTorch 基础",
        function_name="broadcast_add",
        signature="def broadcast_add(a: Any, b: Any) -> torch.Tensor",
        description="使用 PyTorch 广播机制返回 a + b 的结果。输入可以是标量、列表或二维列表。",
        reference=lambda a, b: torch.as_tensor(a) + torch.as_tensor(b),
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef broadcast_add(a, b):\n    return torch.as_tensor(a) + torch.as_tensor(b)\n",
        explanation="把输入转换成 Tensor 后直接使用 +，PyTorch 会按照广播规则扩展维度。",
        constraints=["输入满足 PyTorch 可广播条件", "返回值必须是 torch.Tensor"],
    )

    def ref_cosine(a: Any, b: Any) -> torch.Tensor:
        a_arr = torch.as_tensor(a, dtype=torch.float64)
        b_arr = torch.as_tensor(b, dtype=torch.float64)
        denom = torch.linalg.vector_norm(a_arr, dim=1) * torch.linalg.vector_norm(b_arr, dim=1)
        out = torch.zeros(a_arr.shape[0], dtype=torch.float64)
        mask = denom != 0
        out[mask] = torch.sum(a_arr[mask] * b_arr[mask], dim=1) / denom[mask]
        return out

    add(
        slug="batch-cosine-similarity",
        title="批量余弦相似度",
        difficulty="中等",
        category="Python 与 PyTorch 基础",
        function_name="batch_cosine_similarity",
        signature="def batch_cosine_similarity(a: Any, b: Any) -> torch.Tensor",
        description="给定两个形状相同的二维输入，使用 PyTorch 返回每一行之间的余弦相似度。若某一行存在零向量，该行结果记为 0。",
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef batch_cosine_similarity(a, b):\n    a = torch.as_tensor(a, dtype=torch.float64)\n    b = torch.as_tensor(b, dtype=torch.float64)\n    denom = torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)\n    out = torch.zeros(a.shape[0], dtype=torch.float64)\n    mask = denom != 0\n    out[mask] = torch.sum(a[mask] * b[mask], dim=1) / denom[mask]\n    return out\n",
        explanation="分子是逐行点积，分母是两组向量范数乘积；零范数行不能直接相除。",
        constraints=["a 与 b 形状一致", "二维数组行数至少为 1", "误差容忍 1e-6"],
    )

    add(
        slug="top-k-values",
        title="Top-K 最大值",
        difficulty="简单",
        category="Python 与 PyTorch 基础",
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
        category="Python 与 PyTorch 基础",
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
        category="Python 与 PyTorch 基础",
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
        category="Python 与 PyTorch 基础",
        function_name="one_hot",
        signature="def one_hot(indices: list[int], num_classes: int) -> torch.Tensor",
        description="把类别下标列表转换为独热编码矩阵。下标均在合法范围内。",
        reference=lambda indices, num_classes: torch.nn.functional.one_hot(
            torch.as_tensor(indices, dtype=torch.long), num_classes=num_classes
        ),
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
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef one_hot(indices, num_classes):\n    indices = torch.as_tensor(indices, dtype=torch.long)\n    return F.one_hot(indices, num_classes=num_classes)\n",
        explanation="先把类别下标转换为 LongTensor，再使用 torch.nn.functional.one_hot 构造独热编码。",
        constraints=["0 <= index < num_classes", "num_classes >= 1", "返回 torch.Tensor"],
    )

    add(
        slug="batch-gather",
        title="批量索引",
        difficulty="中等",
        category="Python 与 PyTorch 基础",
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
        category="Python 与 PyTorch 基础",
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
        signature="def relu(x: Any) -> torch.Tensor",
        description="实现 ReLU：逐元素返回 max(x, 0)。输入可以是一维或二维列表。",
        reference=lambda x: torch.relu(torch.as_tensor(x, dtype=torch.float64)),
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef relu(x):\n    return torch.relu(torch.as_tensor(x, dtype=torch.float64))\n",
        explanation="ReLU 会截断负数并保留非负数。",
        constraints=["返回 torch.Tensor", "支持任意可转成 Tensor 的输入"],
    )

    add(
        slug="sigmoid-activation",
        title="Sigmoid 激活函数",
        difficulty="简单",
        category="深度学习基础",
        function_name="sigmoid",
        signature="def sigmoid(x: Any) -> torch.Tensor",
        description="使用 PyTorch 实现逐元素 Sigmoid 激活函数。",
        reference=lambda x: torch.sigmoid(torch.as_tensor(x, dtype=torch.float64)),
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef sigmoid(x):\n    return torch.sigmoid(torch.as_tensor(x, dtype=torch.float64))\n",
        explanation="Sigmoid 将实数映射到 0 到 1 之间。",
        constraints=["返回 torch.Tensor", "误差容忍 1e-6"],
    )

    def ref_softmax_2d(logits: Any) -> torch.Tensor:
        return torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)

    add(
        slug="row-wise-softmax",
        title="二维 Softmax",
        difficulty="中等",
        category="深度学习基础",
        function_name="softmax_2d",
        signature="def softmax_2d(logits: Any) -> torch.Tensor",
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef softmax_2d(logits):\n    return torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)\n",
        explanation="使用 torch.softmax 并指定 dim=1，让每一行独立归一化。",
        constraints=["输入为二维数组", "误差容忍 1e-6"],
    )

    def ref_ce(logits: Any, labels: list[int]) -> float:
        logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)
        return float(torch.nn.functional.cross_entropy(logits_tensor, labels_tensor))

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
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef cross_entropy_loss(logits, labels):\n    logits = torch.as_tensor(logits, dtype=torch.float64)\n    labels = torch.as_tensor(labels, dtype=torch.long)\n    return float(F.cross_entropy(logits, labels))\n",
        explanation="直接使用 PyTorch 的 cross_entropy 处理原始 logits，它内部使用数值稳定的 LogSoftmax 与 NLLLoss。",
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
        reference=lambda y_pred, y_true: float(
            torch.mean((torch.as_tensor(y_pred, dtype=torch.float64) - torch.as_tensor(y_true, dtype=torch.float64)) ** 2)
        ),
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef mse_loss(y_pred, y_true):\n    pred = torch.as_tensor(y_pred, dtype=torch.float64)\n    true = torch.as_tensor(y_true, dtype=torch.float64)\n    return float(torch.mean((pred - true) ** 2))\n",
        explanation="MSE 是所有元素平方误差的平均值。",
        constraints=["y_pred 与 y_true 可广播到相同形状", "误差容忍 1e-6"],
    )

    add(
        slug="dropout-forward-train",
        title="训练模式 Dropout",
        difficulty="中等",
        category="深度学习基础",
        function_name="dropout_train",
        signature="def dropout_train(x: Any, p: float, mask: Any) -> torch.Tensor",
        description="实现训练模式下的 inverted dropout：输出 x * mask / (1-p)。mask 由题目给定，避免随机性。",
        reference=lambda x, p, mask: torch.as_tensor(x, dtype=torch.float64)
        * torch.as_tensor(mask, dtype=torch.float64)
        / (1 - p),
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef dropout_train(x, p, mask):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    mask = torch.as_tensor(mask, dtype=torch.float64)\n    return x * mask / (1 - p)\n",
        explanation="inverted dropout 在训练时缩放保留的激活，使推理时无需额外缩放。",
        constraints=["0 <= p < 1", "mask 与 x 形状一致或可广播"],
    )

    def ref_batch_norm(x: Any, gamma: Any, beta: Any, eps: float = 1e-5) -> torch.Tensor:
        arr = torch.as_tensor(x, dtype=torch.float64)
        mean = arr.mean(dim=0, keepdim=True)
        var = arr.var(dim=0, correction=0, keepdim=True)
        return (arr - mean) / torch.sqrt(var + eps) * torch.as_tensor(gamma) + torch.as_tensor(beta)

    add(
        slug="batch-norm-forward",
        title="BatchNorm 前向计算",
        difficulty="中等",
        category="深度学习基础",
        function_name="batch_norm_forward",
        signature="def batch_norm_forward(x: Any, gamma: Any, beta: Any, eps: float = 1e-5) -> torch.Tensor",
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef batch_norm_forward(x, gamma, beta, eps=1e-5):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    gamma = torch.as_tensor(gamma, dtype=x.dtype)\n    beta = torch.as_tensor(beta, dtype=x.dtype)\n    mean = x.mean(dim=0, keepdim=True)\n    var = x.var(dim=0, correction=0, keepdim=True)\n    return (x - mean) / torch.sqrt(var + eps) * gamma + beta\n",
        explanation="BatchNorm 在 batch 维度上统计每个特征的均值和方差，再应用缩放和平移。",
        constraints=["x 为二维数组", "gamma、beta 长度等于特征数", "误差容忍 1e-6"],
    )

    def ref_layer_norm(x: Any, gamma: Any, beta: Any, eps: float = 1e-5) -> torch.Tensor:
        arr = torch.as_tensor(x, dtype=torch.float64)
        mean = arr.mean(dim=1, keepdim=True)
        var = arr.var(dim=1, correction=0, keepdim=True)
        return (arr - mean) / torch.sqrt(var + eps) * torch.as_tensor(gamma) + torch.as_tensor(beta)

    add(
        slug="layer-norm-forward",
        title="LayerNorm 前向计算",
        difficulty="中等",
        category="深度学习基础",
        function_name="layer_norm_forward",
        signature="def layer_norm_forward(x: Any, gamma: Any, beta: Any, eps: float = 1e-5) -> torch.Tensor",
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef layer_norm_forward(x, gamma, beta, eps=1e-5):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    gamma = torch.as_tensor(gamma, dtype=x.dtype)\n    beta = torch.as_tensor(beta, dtype=x.dtype)\n    mean = x.mean(dim=1, keepdim=True)\n    var = x.var(dim=1, correction=0, keepdim=True)\n    return (x - mean) / torch.sqrt(var + eps) * gamma + beta\n",
        explanation="LayerNorm 的统计维度在样本内部，与 batch size 无关。",
        constraints=["x 为二维数组", "gamma、beta 长度等于特征数", "误差容忍 1e-6"],
    )

    add(
        slug="conv1d-valid",
        title="一维有效卷积",
        difficulty="中等",
        category="深度学习基础",
        function_name="conv1d_valid",
        signature="def conv1d_valid(x: Any, kernel: Any) -> torch.Tensor",
        description="使用 PyTorch 实现 stride=1、无 padding 的一维有效卷积。按深度学习中的互相关写法，不翻转 kernel。",
        reference=lambda x, kernel: torch.nn.functional.conv1d(
            torch.as_tensor(x, dtype=torch.float64).view(1, 1, -1),
            torch.as_tensor(kernel, dtype=torch.float64).view(1, 1, -1),
        ).flatten(),
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
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef conv1d_valid(x, kernel):\n    x = torch.as_tensor(x, dtype=torch.float64).view(1, 1, -1)\n    kernel = torch.as_tensor(kernel, dtype=torch.float64).view(1, 1, -1)\n    return F.conv1d(x, kernel).flatten()\n",
        explanation="把序列和卷积核补成 (N,C,L) 与 (C_out,C_in,K) 形状，再调用 F.conv1d。",
        constraints=["1 <= len(kernel) <= len(x)", "stride 固定为 1"],
    )

    add(
        slug="embedding-lookup",
        title="Embedding 查表",
        difficulty="简单",
        category="深度学习基础",
        function_name="embedding_lookup",
        signature="def embedding_lookup(embedding: Any, indices: Any) -> torch.Tensor",
        description="给定 embedding 矩阵和下标，返回对应行。indices 可以是一维或二维列表。",
        reference=lambda embedding, indices: torch.as_tensor(embedding, dtype=torch.float64)[
            torch.as_tensor(indices, dtype=torch.long)
        ],
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef embedding_lookup(embedding, indices):\n    table = torch.as_tensor(embedding, dtype=torch.float64)\n    indices = torch.as_tensor(indices, dtype=torch.long)\n    return table[indices]\n",
        explanation="Embedding 本质是按 token id 从参数矩阵中取行。",
        constraints=["下标合法", "返回 torch.Tensor"],
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
    def ref_attention(q: Any, k: Any, v: Any) -> torch.Tensor:
        q_arr = torch.as_tensor(q, dtype=torch.float64)
        k_arr = torch.as_tensor(k, dtype=torch.float64)
        v_arr = torch.as_tensor(v, dtype=torch.float64)
        scores = q_arr @ k_arr.transpose(-2, -1) / math.sqrt(q_arr.shape[-1])
        return torch.softmax(scores, dim=-1) @ v_arr

    add(
        slug="scaled-dot-product-attention",
        title="Scaled Dot-Product Attention",
        difficulty="困难",
        category="Attention 与 Transformer",
        function_name="scaled_dot_product_attention",
        signature="def scaled_dot_product_attention(q: Any, k: Any, v: Any) -> torch.Tensor",
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
        imports=torch_imports(),
        solution_code="import math\nimport torch\n\ndef scaled_dot_product_attention(q, k, v):\n    q = torch.as_tensor(q, dtype=torch.float64)\n    k = torch.as_tensor(k, dtype=torch.float64)\n    v = torch.as_tensor(v, dtype=torch.float64)\n    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])\n    return torch.softmax(scores, dim=-1) @ v\n",
        explanation="注意力先计算缩放点积得分，再按 key 维度 softmax，最后对 value 加权求和。",
        constraints=["q、k、v 为二维数组", "k 与 v 的序列长度一致", "误差容忍 1e-6"],
    )

    def ref_masked(scores: Any, mask: Any) -> torch.Tensor:
        arr = torch.as_tensor(scores, dtype=torch.float64)
        bool_mask = torch.as_tensor(mask, dtype=torch.bool)
        return torch.softmax(arr.masked_fill(bool_mask, float("-inf")), dim=-1)

    add(
        slug="attention-mask-softmax",
        title="带 Mask 的 Attention Softmax",
        difficulty="中等",
        category="Attention 与 Transformer",
        function_name="attention_with_mask",
        signature="def attention_with_mask(scores: Any, mask: Any) -> torch.Tensor",
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef attention_with_mask(scores, mask):\n    scores = torch.as_tensor(scores, dtype=torch.float64)\n    mask = torch.as_tensor(mask, dtype=torch.bool)\n    return torch.softmax(scores.masked_fill(mask, float('-inf')), dim=-1)\n",
        explanation="用 masked_fill 把不可见位置设为负无穷，再沿 key 维执行 Softmax。",
        constraints=["scores 与 mask 形状一致", "每行至少一个未 mask 位置"],
    )

    def ref_pos(length: int, dim: int) -> torch.Tensor:
        pe = torch.zeros((length, dim), dtype=torch.float64)
        positions = torch.arange(length, dtype=torch.float64).unsqueeze(1)
        even_dims = torch.arange(0, dim, 2, dtype=torch.float64)
        angles = positions / torch.pow(10000.0, even_dims / dim)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
        return pe

    add(
        slug="sinusoidal-positional-encoding",
        title="正弦位置编码",
        difficulty="困难",
        category="Attention 与 Transformer",
        function_name="positional_encoding",
        signature="def positional_encoding(length: int, dim: int) -> torch.Tensor",
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef positional_encoding(length, dim):\n    pe = torch.zeros((length, dim), dtype=torch.float64)\n    positions = torch.arange(length, dtype=torch.float64).unsqueeze(1)\n    even_dims = torch.arange(0, dim, 2, dtype=torch.float64)\n    angles = positions / torch.pow(10000.0, even_dims / dim)\n    pe[:, 0::2] = torch.sin(angles)\n    pe[:, 1::2] = torch.cos(angles[:, :pe[:, 1::2].shape[1]])\n    return pe\n",
        explanation="用 PyTorch 广播一次构造所有位置与频率的角度，再分别把 sin/cos 写入偶数列和奇数列。",
        constraints=["length >= 0", "dim >= 1", "误差容忍 1e-6"],
    )

    def ref_split_heads(x: Any, num_heads: int) -> torch.Tensor:
        arr = torch.as_tensor(x, dtype=torch.float64)
        batch, seq, dim = arr.shape
        head_dim = dim // num_heads
        return arr.reshape(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)

    add(
        slug="split-heads",
        title="拆分多头",
        difficulty="中等",
        category="Attention 与 Transformer",
        function_name="split_heads",
        signature="def split_heads(x: Any, num_heads: int) -> torch.Tensor",
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef split_heads(x, num_heads):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    batch, seq, dim = x.shape\n    head_dim = dim // num_heads\n    return x.reshape(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)\n",
        explanation="先 reshape 出 heads 维度，再转置到注意力计算常用的维度顺序。",
        constraints=["dim 能被 num_heads 整除", "输入为三维数组"],
    )

    add(
        slug="combine-heads",
        title="合并多头",
        difficulty="中等",
        category="Attention 与 Transformer",
        function_name="combine_heads",
        signature="def combine_heads(x: Any) -> torch.Tensor",
        description="把形状 (batch, heads, seq, head_dim) 的张量合并回 (batch, seq, heads*head_dim)。",
        reference=lambda x: torch.as_tensor(x, dtype=torch.float64)
        .permute(0, 2, 1, 3)
        .reshape(len(x), len(x[0][0]), len(x[0]) * len(x[0][0][0])),
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef combine_heads(x):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    batch, heads, seq, head_dim = x.shape\n    return x.permute(0, 2, 1, 3).contiguous().reshape(batch, seq, heads * head_dim)\n",
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
        reference=lambda logits: torch.as_tensor(logits).argmax(dim=1).tolist(),
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef greedy_decode_step(logits):\n    return torch.as_tensor(logits).argmax(dim=1).tolist()\n",
        explanation="贪心解码每一步选择概率或 logit 最大的 token。",
        constraints=["logits 为二维输入", "并列时 torch.argmax 返回最小下标"],
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

    def ref_smooth(labels: list[int], num_classes: int, epsilon: float) -> torch.Tensor:
        off = epsilon / num_classes
        arr = torch.full((len(labels), num_classes), off, dtype=torch.float64)
        for i, label in enumerate(labels):
            arr[i, label] = 1 - epsilon + off
        return arr

    add(
        slug="label-smoothing",
        title="Label Smoothing",
        difficulty="中等",
        category="Attention 与 Transformer",
        function_name="label_smoothing",
        signature="def label_smoothing(labels: list[int], num_classes: int, epsilon: float) -> torch.Tensor",
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef label_smoothing(labels, num_classes, epsilon):\n    off = epsilon / num_classes\n    result = torch.full((len(labels), num_classes), off, dtype=torch.float64)\n    if labels:\n        rows = torch.arange(len(labels))\n        result[rows, torch.as_tensor(labels, dtype=torch.long)] += 1 - epsilon\n    return result\n",
        explanation="Label smoothing 会降低真实类别的置信度，同时给其他类别分配少量概率。",
        constraints=["0 <= epsilon <= 1", "标签下标合法", "误差容忍 1e-6"],
    )

    # 计算机视觉
    def ref_conv2d(image: Any, kernel: Any) -> torch.Tensor:
        img = torch.as_tensor(image, dtype=torch.float64).view(1, 1, len(image), len(image[0]))
        ker = torch.as_tensor(kernel, dtype=torch.float64).view(1, 1, len(kernel), len(kernel[0]))
        return torch.nn.functional.conv2d(img, ker).squeeze(0).squeeze(0)

    add(
        slug="conv2d-valid",
        title="二维有效卷积",
        difficulty="困难",
        category="计算机视觉",
        function_name="conv2d_valid",
        signature="def conv2d_valid(image: Any, kernel: Any) -> torch.Tensor",
        description="使用 PyTorch 实现 stride=1、无 padding 的二维有效卷积。按深度学习互相关写法，不翻转 kernel。",
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
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef conv2d_valid(image, kernel):\n    image = torch.as_tensor(image, dtype=torch.float64)[None, None]\n    kernel = torch.as_tensor(kernel, dtype=torch.float64)[None, None]\n    return F.conv2d(image, kernel).squeeze(0).squeeze(0)\n",
        explanation="把二维输入补成 PyTorch 所需的 (N,C,H,W) 形状，再使用 F.conv2d。",
        constraints=["kernel 尺寸不大于 image", "返回 torch.Tensor"],
    )

    def ref_pool(image: Any, kernel_size: int, stride: int) -> torch.Tensor:
        img = torch.as_tensor(image, dtype=torch.float64)[None, None]
        return torch.nn.functional.max_pool2d(img, kernel_size, stride).squeeze(0).squeeze(0)

    add(
        slug="max-pool2d",
        title="二维最大池化",
        difficulty="中等",
        category="计算机视觉",
        function_name="max_pool2d",
        signature="def max_pool2d(image: Any, kernel_size: int, stride: int) -> torch.Tensor",
        description="使用 PyTorch 实现单通道二维最大池化，输入为二维数组。",
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
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef max_pool2d(image, kernel_size, stride):\n    image = torch.as_tensor(image, dtype=torch.float64)[None, None]\n    return F.max_pool2d(image, kernel_size, stride).squeeze(0).squeeze(0)\n",
        explanation="把二维图像补齐 batch 和 channel 维，再调用 F.max_pool2d。",
        constraints=["kernel_size >= 1", "stride >= 1", "返回 torch.Tensor"],
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

    def ref_patch(image: Any, patch_size: int) -> torch.Tensor:
        img = torch.as_tensor(image, dtype=torch.float64)
        return img.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).contiguous().view(-1, patch_size**2)

    add(
        slug="patch-embedding-flatten",
        title="图像 Patch 展平",
        difficulty="中等",
        category="计算机视觉",
        function_name="patch_embedding_flatten",
        signature="def patch_embedding_flatten(image: Any, patch_size: int) -> torch.Tensor",
        description="把二维图像按不重叠 patch 切分，并把每个 patch 展平成一行。假设高宽都能被 patch_size 整除。",
        reference=ref_patch,
        raw_cases=[
            case([[[1, 2], [3, 4]], 1]),
            case([[[1, 2], [3, 4]], 2]),
            case([[[1, 2, 3, 4], [5, 6, 7, 8]], 2]),
            case([[[0]], 1]),
            case([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], 2]),
            case([[[1, 1], [1, 1]], 2]),
            case([[[2, 4, 6, 8], [1, 3, 5, 7]], 2]),
            case([[[3, 2], [1, 0], [5, 4], [7, 6]], 2]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef patch_embedding_flatten(image, patch_size):\n    image = torch.as_tensor(image, dtype=torch.float64)\n    patches = image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)\n    return patches.contiguous().view(-1, patch_size ** 2)\n",
        explanation="使用 Tensor.unfold 在高、宽维提取不重叠窗口，再把每个 patch 展平。",
        constraints=["高宽能被 patch_size 整除", "返回二维 torch.Tensor"],
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
        logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)
        valid = labels_tensor != pad_id
        if not bool(valid.any()):
            return 0.0
        return float(torch.nn.functional.cross_entropy(logits_tensor[valid], labels_tensor[valid]))

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
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef sequence_cross_entropy(logits, labels, pad_id):\n    logits = torch.as_tensor(logits, dtype=torch.float64)\n    labels = torch.as_tensor(labels, dtype=torch.long)\n    valid = labels != pad_id\n    if not bool(valid.any()):\n        return 0.0\n    return float(F.cross_entropy(logits[valid], labels[valid]))\n",
        explanation="用布尔 mask 选出非 Padding 的 logits 与标签，再调用 PyTorch 交叉熵。",
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
        reference=lambda values: bool(torch.isnan(torch.as_tensor(values, dtype=torch.float64)).any()),
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
        imports=torch_imports(),
        solution_code="import torch\n\ndef has_nan(values):\n    return bool(torch.isnan(torch.as_tensor(values, dtype=torch.float64)).any())\n",
        explanation="torch.isnan 可以对 Tensor 逐元素检测 NaN，再用 any 汇总。",
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

    # 现代大模型、优化与多模态高频手撕题
    add(
        slug="pairwise-euclidean-distance",
        title="批量两两欧氏距离",
        difficulty="中等",
        category="PyTorch 基础",
        function_name="pairwise_euclidean_distance",
        signature="def pairwise_euclidean_distance(x: Any, y: Any) -> torch.Tensor",
        description="给定形状 (N,D) 和 (M,D) 的两组向量，使用 PyTorch 返回形状 (N,M) 的两两欧氏距离矩阵。",
        reference=lambda x, y: torch.cdist(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)),
        raw_cases=[
            case([[[0, 0], [1, 0]], [[0, 1], [1, 1]]]),
            case([[[1, 2]], [[1, 2]]]),
            case([[[0], [2], [5]], [[1], [3]]]),
            case([[[-1, -1]], [[1, 1], [-1, -1]]]),
            case([[[1, 2, 3], [4, 5, 6]], [[0, 0, 0]]]),
            case([[[0.5, 1.5]], [[1.5, 0.5]]]),
            case([[[3, 4]], [[0, 0], [6, 8]]]),
            case([[[0, 0], [0, 0]], [[0, 0]]]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef pairwise_euclidean_distance(x, y):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    y = torch.as_tensor(y, dtype=torch.float64)\n    return torch.cdist(x, y)\n",
        explanation="torch.cdist 可以直接计算两批行向量之间的 p 范数距离，本题使用默认 p=2。",
        constraints=["x、y 均为二维输入", "特征维相同", "返回 torch.Tensor"],
        company_tags=["Meta", "Amazon", "字节跳动"],
    )

    add(
        slug="binary-cross-entropy-logits",
        title="稳定二元交叉熵",
        difficulty="中等",
        category="深度学习基础",
        function_name="binary_cross_entropy_logits",
        signature="def binary_cross_entropy_logits(logits: Any, targets: Any) -> float",
        description="直接从 logits 计算二元交叉熵均值，要求使用数值稳定写法，不要先显式计算概率再取对数。",
        reference=lambda logits, targets: float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.as_tensor(logits, dtype=torch.float64), torch.as_tensor(targets, dtype=torch.float64)
            )
        ),
        raw_cases=[
            case([[0.0], [0.0]]),
            case([[0.0], [1.0]]),
            case([[10.0, -10.0], [1.0, 0.0]]),
            case([[1000.0, -1000.0], [1.0, 0.0]]),
            case([[-2.0, 2.0], [1.0, 0.0]]),
            case([[[1.0, -1.0], [0.5, -0.5]], [[1.0, 0.0], [1.0, 0.0]]]),
            case([[3.5], [1.0]]),
            case([[-3.5], [0.0]]),
        ],
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef binary_cross_entropy_logits(logits, targets):\n    logits = torch.as_tensor(logits, dtype=torch.float64)\n    targets = torch.as_tensor(targets, dtype=torch.float64)\n    return float(F.binary_cross_entropy_with_logits(logits, targets))\n",
        explanation="BCEWithLogits 把 Sigmoid 和 BCE 合并为稳定的 log-sum-exp 计算。",
        constraints=["logits 与 targets 形状一致", "targets 元素为 0 或 1", "误差容忍 1e-6"],
        company_tags=["Amazon", "Google", "腾讯"],
    )

    def ref_focal(logits: Any, targets: Any, alpha: float, gamma: float) -> float:
        x = torch.as_tensor(logits, dtype=torch.float64)
        y = torch.as_tensor(targets, dtype=torch.float64)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(x, y, reduction="none")
        prob = torch.sigmoid(x)
        pt = torch.where(y == 1, prob, 1 - prob)
        alpha_t = torch.where(y == 1, alpha, 1 - alpha)
        return float((alpha_t * (1 - pt).pow(gamma) * bce).mean())

    add(
        slug="binary-focal-loss",
        title="二分类 Focal Loss",
        difficulty="困难",
        category="计算机视觉",
        function_name="binary_focal_loss",
        signature="def binary_focal_loss(logits: Any, targets: Any, alpha: float, gamma: float) -> float",
        description="从二分类 logits 计算带 alpha 平衡项的平均 Focal Loss，用于缓解正负样本不均衡。",
        reference=ref_focal,
        raw_cases=[
            case([[0.0], [1.0], 0.25, 2.0]),
            case([[0.0], [0.0], 0.25, 2.0]),
            case([[5.0, -5.0], [1.0, 0.0], 0.25, 2.0]),
            case([[-5.0, 5.0], [1.0, 0.0], 0.25, 2.0]),
            case([[1.0, -1.0, 0.0], [1.0, 0.0, 1.0], 0.5, 0.0]),
            case([[[1.0, -1.0]], [[0.0, 1.0]], 0.75, 1.0]),
            case([[10.0], [1.0], 0.25, 3.0]),
            case([[-10.0], [0.0], 0.25, 3.0]),
        ],
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef binary_focal_loss(logits, targets, alpha, gamma):\n    logits = torch.as_tensor(logits, dtype=torch.float64)\n    targets = torch.as_tensor(targets, dtype=torch.float64)\n    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')\n    prob = torch.sigmoid(logits)\n    pt = torch.where(targets == 1, prob, 1 - prob)\n    alpha_t = torch.where(targets == 1, alpha, 1 - alpha)\n    return float((alpha_t * (1 - pt).pow(gamma) * bce).mean())\n",
        explanation="先稳定计算逐元素 BCE，再按真实类别概率 p_t 添加难例调制因子。",
        constraints=["0 <= alpha <= 1", "gamma >= 0", "logits 与 targets 形状一致"],
        company_tags=["Meta", "Amazon", "字节跳动"],
    )

    def ref_dice_loss(logits: Any, targets: Any, eps: float = 1e-6) -> float:
        prob = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float64))
        target = torch.as_tensor(targets, dtype=torch.float64)
        return float(1 - (2 * (prob * target).sum() + eps) / (prob.sum() + target.sum() + eps))

    add(
        slug="dice-loss",
        title="分割 Dice Loss",
        difficulty="中等",
        category="计算机视觉",
        function_name="dice_loss",
        signature="def dice_loss(logits: Any, targets: Any, eps: float = 1e-6) -> float",
        description="给定二值分割 logits 和 0/1 mask，在整个输入上计算带平滑项的 Dice Loss。",
        reference=ref_dice_loss,
        raw_cases=[
            case([[0.0, 0.0], [1.0, 0.0]]),
            case([[10.0, -10.0], [1.0, 0.0]]),
            case([[-10.0, 10.0], [1.0, 0.0]]),
            case([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
            case([[[1.0, -1.0]], [[1.0, 1.0]]]),
            case([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], 1e-5]),
            case([[-2.0, -2.0], [0.0, 0.0]]),
            case([[0.5], [1.0]]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef dice_loss(logits, targets, eps=1e-6):\n    prob = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float64))\n    targets = torch.as_tensor(targets, dtype=torch.float64)\n    dice = (2 * (prob * targets).sum() + eps) / (prob.sum() + targets.sum() + eps)\n    return float(1 - dice)\n",
        explanation="Dice 直接度量预测 mask 与目标 mask 的重叠，损失取 1−Dice。",
        constraints=["logits 与 targets 形状一致", "targets 只含 0/1", "eps > 0"],
        company_tags=["NVIDIA", "百度", "字节跳动"],
    )

    def ref_rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> torch.Tensor:
        value = torch.as_tensor(x, dtype=torch.float64)
        scale = torch.as_tensor(weight, dtype=torch.float64)
        return value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps) * scale

    add(
        slug="rms-norm",
        title="RMSNorm 前向计算",
        difficulty="中等",
        category="大模型核心组件",
        function_name="rms_norm",
        signature="def rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> torch.Tensor",
        description="使用 PyTorch 手写 RMSNorm：沿最后一维按均方根归一化，再乘逐维权重；不要减均值。",
        reference=ref_rms_norm,
        raw_cases=[
            case([[[1.0, 2.0]], [1.0, 1.0]]),
            case([[[1.0, -1.0], [2.0, -2.0]], [1.0, 1.0]]),
            case([[[0.0, 0.0]], [1.0, 1.0]]),
            case([[[1.0, 2.0, 3.0]], [1.0, 0.5, 2.0]]),
            case([[[[1.0, 0.0], [0.0, 1.0]]], [1.0, 1.0]]),
            case([[[-3.0, 4.0]], [2.0, 0.5], 1e-5]),
            case([[[5.0]], [1.0]]),
            case([[[0.1, 0.2]], [0.0, 1.0]]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef rms_norm(x, weight, eps=1e-6):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    weight = torch.as_tensor(weight, dtype=x.dtype)\n    rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)\n    return x * rms_inv * weight\n",
        explanation="RMSNorm 只按平方均值缩放，不做中心化，现代大语言模型中非常常见。",
        constraints=["weight 长度等于 x 的最后一维", "eps > 0", "返回 torch.Tensor"],
        company_tags=["Meta", "Google", "OpenAI", "字节跳动"],
    )

    add(
        slug="swiglu-activation",
        title="SwiGLU 门控激活",
        difficulty="中等",
        category="大模型核心组件",
        function_name="swiglu",
        signature="def swiglu(gate: Any, value: Any) -> torch.Tensor",
        description="给定形状相同的 gate 和 value，使用 PyTorch 实现 SwiGLU 核心门控：SiLU(gate) * value。",
        reference=lambda gate, value: torch.nn.functional.silu(torch.as_tensor(gate, dtype=torch.float64))
        * torch.as_tensor(value, dtype=torch.float64),
        raw_cases=[
            case([[0.0], [1.0]]),
            case([[1.0, -1.0], [2.0, 3.0]]),
            case([[[1.0, 2.0], [-1.0, 0.0]], [[1.0, 0.5], [2.0, 3.0]]]),
            case([[10.0], [0.0]]),
            case([[-10.0], [2.0]]),
            case([[0.5, 1.5], [-1.0, 1.0]]),
            case([[[0.0, 0.0]], [[2.0, -2.0]]]),
            case([[2.0], [-3.0]]),
        ],
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef swiglu(gate, value):\n    gate = torch.as_tensor(gate, dtype=torch.float64)\n    value = torch.as_tensor(value, dtype=torch.float64)\n    return F.silu(gate) * value\n",
        explanation="SwiGLU 用 SiLU 激活其中一条支路，并与另一条支路逐元素相乘。",
        constraints=["gate 与 value 形状一致", "返回 torch.Tensor"],
        company_tags=["Google", "Meta", "OpenAI"],
    )

    def ref_rope(x: Any, base: float = 10000.0) -> torch.Tensor:
        value = torch.as_tensor(x, dtype=torch.float64)
        seq, dim = value.shape
        pos = torch.arange(seq, dtype=value.dtype).unsqueeze(1)
        pair = torch.arange(0, dim, 2, dtype=value.dtype)
        angle = pos / torch.pow(base, pair / dim)
        even, odd = value[:, 0::2], value[:, 1::2]
        out = torch.empty_like(value)
        out[:, 0::2] = even * torch.cos(angle) - odd * torch.sin(angle)
        out[:, 1::2] = even * torch.sin(angle) + odd * torch.cos(angle)
        return out

    add(
        slug="rotary-position-embedding",
        title="RoPE 旋转位置编码",
        difficulty="困难",
        category="大模型核心组件",
        function_name="apply_rope",
        signature="def apply_rope(x: Any, base: float = 10000.0) -> torch.Tensor",
        description="对形状 (seq, dim) 且 dim 为偶数的 Tensor 应用 Rotary Position Embedding，相邻偶/奇维组成一个二维旋转对。",
        reference=ref_rope,
        raw_cases=[
            case([[[1.0, 0.0], [1.0, 0.0]]]),
            case([[[1.0, 2.0, 3.0, 4.0]]]),
            case([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]),
            case([[[0.0, 0.0], [2.0, -1.0]], 100.0]),
            case([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]]),
            case([[[5.0, -2.0]]]),
            case([[[1.0, 1.0], [1.0, 1.0]], 10.0]),
            case([[[0.5, -0.5, 1.5, -1.5], [2.0, 3.0, 4.0, 5.0]]]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef apply_rope(x, base=10000.0):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    seq, dim = x.shape\n    pos = torch.arange(seq, dtype=x.dtype).unsqueeze(1)\n    pair = torch.arange(0, dim, 2, dtype=x.dtype)\n    angle = pos / torch.pow(base, pair / dim)\n    even, odd = x[:, 0::2], x[:, 1::2]\n    out = torch.empty_like(x)\n    out[:, 0::2] = even * torch.cos(angle) - odd * torch.sin(angle)\n    out[:, 1::2] = even * torch.sin(angle) + odd * torch.cos(angle)\n    return out\n",
        explanation="RoPE 不把位置向量直接相加，而是根据位置对 Query/Key 的相邻通道做二维旋转。",
        constraints=["x 为二维输入", "dim 为正偶数", "base > 1"],
        company_tags=["Meta", "Google", "字节跳动", "阿里巴巴"],
    )

    def ref_lora(x: Any, weight: Any, a: Any, b: Any, alpha: float) -> torch.Tensor:
        value = torch.as_tensor(x, dtype=torch.float64)
        w = torch.as_tensor(weight, dtype=torch.float64)
        lora_a = torch.as_tensor(a, dtype=torch.float64)
        lora_b = torch.as_tensor(b, dtype=torch.float64)
        return torch.nn.functional.linear(value, w) + (alpha / lora_a.shape[0]) * torch.nn.functional.linear(
            torch.nn.functional.linear(value, lora_a), lora_b
        )

    add(
        slug="lora-linear-forward",
        title="LoRA 线性层前向",
        difficulty="困难",
        category="大模型核心组件",
        function_name="lora_linear",
        signature="def lora_linear(x: Any, weight: Any, a: Any, b: Any, alpha: float) -> torch.Tensor",
        description="实现无 bias 的 LoRA 线性层前向：冻结主权重 W，并叠加由 A、B 构成的低秩更新，缩放为 alpha/r。",
        reference=ref_lora,
        raw_cases=[
            case([[[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0]], [[1.0], [2.0]], 1.0]),
            case([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 2.0]], [[1.0, -1.0]], [[0.5]], 2.0]),
            case([[[0.0, 0.0]], [[1.0, 1.0]], [[1.0, 0.0]], [[1.0]], 1.0]),
            case([[[1.0]], [[2.0]], [[3.0]], [[4.0]], 0.0]),
            case([[[1.0, -1.0]], [[0.5, 0.5]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0]], 2.0]),
            case([[[2.0, 3.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5]], [[2.0], [-1.0]], 4.0]),
            case([[[[1.0, 2.0], [3.0, 4.0]]], [[1.0, 1.0]], [[1.0, 0.0]], [[1.0]], 1.0]),
            case([[[-1.0, 1.0]], [[1.0, -1.0]], [[2.0, 2.0]], [[0.25]], 8.0]),
        ],
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef lora_linear(x, weight, a, b, alpha):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    weight = torch.as_tensor(weight, dtype=x.dtype)\n    a = torch.as_tensor(a, dtype=x.dtype)\n    b = torch.as_tensor(b, dtype=x.dtype)\n    base = F.linear(x, weight)\n    update = F.linear(F.linear(x, a), b)\n    return base + (alpha / a.shape[0]) * update\n",
        explanation="LoRA 用两个小矩阵表达低秩权重增量，无需显式构造完整的 B@A。",
        constraints=["矩阵形状满足题面公式", "r=a.shape[0] >= 1", "无 bias"],
        company_tags=["Microsoft", "Meta", "OpenAI", "阿里巴巴"],
    )

    def ref_clip_grad(grads: list[Any], max_norm: float, eps: float = 1e-6) -> list[torch.Tensor]:
        tensors = [torch.as_tensor(grad, dtype=torch.float64) for grad in grads]
        total = torch.sqrt(sum((grad * grad).sum() for grad in tensors))
        coef = min(1.0, max_norm / (float(total) + eps))
        return [grad * coef for grad in tensors]

    add(
        slug="clip-grad-global-norm",
        title="按全局范数裁剪梯度",
        difficulty="中等",
        category="训练、调试与工程",
        function_name="clip_grad_global_norm",
        signature="def clip_grad_global_norm(grads: list[Any], max_norm: float, eps: float = 1e-6) -> list[torch.Tensor]",
        description="给定多个梯度 Tensor，按所有梯度共同的全局 L2 范数进行裁剪，并返回裁剪后的新 Tensor 列表。",
        reference=ref_clip_grad,
        raw_cases=[
            case([[[3.0, 4.0]], 5.0]),
            case([[[3.0, 4.0]], 1.0]),
            case([[[1.0, 2.0], [2.0]], 10.0]),
            case([[[0.0, 0.0]], 1.0]),
            case([[[[1.0, -1.0]], [[2.0]]], 2.0]),
            case([[[6.0], [8.0]], 5.0, 1e-8]),
            case([[[-3.0, -4.0], [0.0]], 2.5]),
            case([[[0.1, 0.2, 0.3]], 0.1]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef clip_grad_global_norm(grads, max_norm, eps=1e-6):\n    grads = [torch.as_tensor(g, dtype=torch.float64) for g in grads]\n    total_norm = torch.sqrt(sum((g * g).sum() for g in grads))\n    coef = min(1.0, max_norm / (float(total_norm) + eps))\n    return [g * coef for g in grads]\n",
        explanation="全局裁剪必须先联合计算所有梯度的范数，再对每个梯度使用同一个缩放系数。",
        constraints=["grads 非空", "max_norm > 0", "不修改原输入"],
        company_tags=["NVIDIA", "OpenAI", "字节跳动"],
    )

    def ref_sgd_momentum(param: Any, grad: Any, velocity: Any, lr: float, momentum: float) -> dict[str, torch.Tensor]:
        p = torch.as_tensor(param, dtype=torch.float64)
        g = torch.as_tensor(grad, dtype=torch.float64)
        v = momentum * torch.as_tensor(velocity, dtype=torch.float64) + g
        return {"param": p - lr * v, "velocity": v}

    add(
        slug="sgd-momentum-step",
        title="SGD Momentum 单步更新",
        difficulty="中等",
        category="优化器与训练",
        function_name="sgd_momentum_step",
        signature="def sgd_momentum_step(param: Any, grad: Any, velocity: Any, lr: float, momentum: float) -> dict[str, torch.Tensor]",
        description="实现一次经典 SGD Momentum 更新，返回更新后的 param 和 velocity，均为 Tensor。",
        reference=ref_sgd_momentum,
        raw_cases=[
            case([[1.0], [0.5], [0.0], 0.1, 0.9]),
            case([[1.0, 2.0], [0.1, -0.2], [0.5, 0.5], 0.01, 0.9]),
            case([[0.0], [1.0], [2.0], 0.5, 0.0]),
            case([[[-1.0, 1.0]], [[0.5, -0.5]], [[0.0, 0.0]], 0.1, 0.5]),
            case([[10.0], [-2.0], [1.0], 0.01, 0.99]),
            case([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], 0.1, 0.9]),
            case([[3.0], [4.0], [-2.0], 0.25, 0.5]),
            case([[[1.0], [2.0]], [[3.0], [4.0]], [[0.0], [1.0]], 0.05, 0.8]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef sgd_momentum_step(param, grad, velocity, lr, momentum):\n    param = torch.as_tensor(param, dtype=torch.float64)\n    grad = torch.as_tensor(grad, dtype=torch.float64)\n    velocity = momentum * torch.as_tensor(velocity, dtype=torch.float64) + grad\n    return {'param': param - lr * velocity, 'velocity': velocity}\n",
        explanation="Momentum 把历史速度与当前梯度累加，再用新速度更新参数。",
        constraints=["param、grad、velocity 形状一致", "lr > 0", "0 <= momentum < 1"],
        company_tags=["Google", "Amazon", "华为"],
    )

    def ref_adamw(
        param: Any,
        grad: Any,
        m: Any,
        v: Any,
        step: int,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> dict[str, torch.Tensor]:
        p = torch.as_tensor(param, dtype=torch.float64)
        g = torch.as_tensor(grad, dtype=torch.float64)
        first = beta1 * torch.as_tensor(m, dtype=torch.float64) + (1 - beta1) * g
        second = beta2 * torch.as_tensor(v, dtype=torch.float64) + (1 - beta2) * g.square()
        first_hat = first / (1 - beta1**step)
        second_hat = second / (1 - beta2**step)
        updated = p * (1 - lr * weight_decay) - lr * first_hat / (torch.sqrt(second_hat) + eps)
        return {"param": updated, "m": first, "v": second}

    add(
        slug="adamw-step",
        title="AdamW 单步更新",
        difficulty="困难",
        category="优化器与训练",
        function_name="adamw_step",
        signature="def adamw_step(param: Any, grad: Any, m: Any, v: Any, step: int, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float) -> dict[str, torch.Tensor]",
        description="实现一次带偏差修正和解耦权重衰减的 AdamW 更新，返回新的 param、m、v。step 从 1 开始。",
        reference=ref_adamw,
        raw_cases=[
            case([[1.0], [0.1], [0.0], [0.0], 1, 0.001, 0.9, 0.999, 1e-8, 0.01]),
            case([[1.0, 2.0], [0.1, -0.2], [0.0, 0.0], [0.0, 0.0], 1, 0.01, 0.9, 0.99, 1e-8, 0.0]),
            case([[0.0], [1.0], [0.2], [0.3], 2, 0.1, 0.5, 0.9, 1e-6, 0.1]),
            case([[[-1.0, 1.0]], [[0.5, -0.5]], [[0.1, -0.1]], [[0.2, 0.2]], 5, 0.001, 0.9, 0.999, 1e-8, 0.01]),
            case([[10.0], [0.0], [1.0], [2.0], 3, 0.01, 0.8, 0.9, 1e-5, 0.1]),
            case([[1.0], [-2.0], [0.0], [0.0], 1, 0.1, 0.0, 0.0, 1e-8, 0.0]),
            case([[3.0, 4.0], [0.3, 0.4], [0.1, 0.2], [0.01, 0.02], 10, 0.005, 0.9, 0.999, 1e-8, 0.05]),
            case([[[1.0], [2.0]], [[0.1], [0.2]], [[0.0], [0.0]], [[0.0], [0.0]], 1, 0.01, 0.9, 0.999, 1e-8, 0.01]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef adamw_step(param, grad, m, v, step, lr, beta1, beta2, eps, weight_decay):\n    param = torch.as_tensor(param, dtype=torch.float64)\n    grad = torch.as_tensor(grad, dtype=torch.float64)\n    m = beta1 * torch.as_tensor(m, dtype=torch.float64) + (1 - beta1) * grad\n    v = beta2 * torch.as_tensor(v, dtype=torch.float64) + (1 - beta2) * grad.square()\n    m_hat = m / (1 - beta1 ** step)\n    v_hat = v / (1 - beta2 ** step)\n    param = param * (1 - lr * weight_decay) - lr * m_hat / (torch.sqrt(v_hat) + eps)\n    return {'param': param, 'm': m, 'v': v}\n",
        explanation="AdamW 与 L2 正则化版 Adam 的关键区别是权重衰减从自适应梯度项中解耦。",
        constraints=["step >= 1", "张量形状一致", "0 <= beta1,beta2 < 1", "eps > 0"],
        company_tags=["Google", "Meta", "NVIDIA", "字节跳动"],
    )

    def ref_warmup_cosine(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps - 1)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    add(
        slug="warmup-cosine-learning-rate",
        title="Warmup + Cosine 学习率",
        difficulty="中等",
        category="优化器与训练",
        function_name="warmup_cosine_lr",
        signature="def warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float",
        description="实现从 step=0 开始的线性 warmup 与余弦衰减。warmup 最后一步到达 max_lr，训练最后一步到达 min_lr。",
        reference=ref_warmup_cosine,
        raw_cases=[
            case([0, 2, 6, 1.0, 0.0]),
            case([1, 2, 6, 1.0, 0.0]),
            case([2, 2, 6, 1.0, 0.0]),
            case([5, 2, 6, 1.0, 0.0]),
            case([3, 1, 5, 0.1, 0.01]),
            case([0, 1, 3, 0.01, 0.001]),
            case([2, 1, 3, 0.01, 0.001]),
            case([4, 3, 7, 2.0, 0.5]),
        ],
        imports=py_imports(),
        solution_code="import math\n\ndef warmup_cosine_lr(step, warmup_steps, total_steps, max_lr, min_lr):\n    if step < warmup_steps:\n        return max_lr * (step + 1) / warmup_steps\n    progress = (step - warmup_steps) / (total_steps - warmup_steps - 1)\n    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))\n",
        explanation="先线性升温，再把余下 step 归一化到 [0,1] 并套用半个余弦周期。",
        constraints=["1 <= warmup_steps < total_steps-1", "0 <= step < total_steps", "0 <= min_lr <= max_lr"],
        company_tags=["Google", "OpenAI", "字节跳动"],
    )

    add(
        slug="exponential-moving-average",
        title="模型参数 EMA 更新",
        difficulty="简单",
        category="优化器与训练",
        function_name="ema_update",
        signature="def ema_update(shadow: Any, current: Any, decay: float) -> torch.Tensor",
        description="给定上一时刻的 shadow 参数和当前模型参数，返回一次指数移动平均更新后的 Tensor。",
        reference=lambda shadow, current, decay: decay * torch.as_tensor(shadow, dtype=torch.float64)
        + (1 - decay) * torch.as_tensor(current, dtype=torch.float64),
        raw_cases=[
            case([[0.0], [1.0], 0.9]),
            case([[1.0, 2.0], [3.0, 4.0], 0.5]),
            case([[[1.0, 0.0]], [[0.0, 1.0]], 0.99]),
            case([[5.0], [-5.0], 0.0]),
            case([[5.0], [-5.0], 1.0]),
            case([[-1.0, 1.0], [1.0, -1.0], 0.8]),
            case([[[[1.0], [2.0]]], [[[3.0], [4.0]]], 0.25]),
            case([[0.1], [0.2], 0.999]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef ema_update(shadow, current, decay):\n    shadow = torch.as_tensor(shadow, dtype=torch.float64)\n    current = torch.as_tensor(current, dtype=torch.float64)\n    return decay * shadow + (1 - decay) * current\n",
        explanation="EMA 用较大权重保留历史平滑参数，并吸收少量当前参数。",
        constraints=["shadow 与 current 形状一致", "0 <= decay <= 1"],
        company_tags=["NVIDIA", "Meta", "百度"],
    )

    def ref_top_p(probs: list[float], p: float) -> list[int]:
        values = torch.as_tensor(probs, dtype=torch.float64)
        order = torch.argsort(values, descending=True, stable=True)
        cumulative = torch.cumsum(values[order], dim=0)
        count = int(torch.searchsorted(cumulative, torch.tensor(p, dtype=values.dtype), right=False)) + 1
        return order[: min(count, len(probs))].tolist()

    add(
        slug="top-p-sampling-candidates",
        title="Top-P Nucleus 候选集",
        difficulty="中等",
        category="大模型推理与解码",
        function_name="top_p_candidates",
        signature="def top_p_candidates(probs: list[float], p: float) -> list[int]",
        description="按概率降序返回最小的 token 下标集合，使累计概率达到 p；概率相同保持原下标顺序，且至少保留一个 token。",
        reference=ref_top_p,
        raw_cases=[
            case([[0.6, 0.3, 0.1], 0.8]),
            case([[0.4, 0.3, 0.2, 0.1], 0.5]),
            case([[1.0], 0.9]),
            case([[0.5, 0.5], 0.5]),
            case([[0.25, 0.25, 0.25, 0.25], 0.7]),
            case([[0.05, 0.15, 0.8], 1.0]),
            case([[0.9, 0.05, 0.05], 0.1]),
            case([[0.1, 0.2, 0.3, 0.4], 0.6]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef top_p_candidates(probs, p):\n    probs = torch.as_tensor(probs, dtype=torch.float64)\n    order = torch.argsort(probs, descending=True, stable=True)\n    cumulative = torch.cumsum(probs[order], dim=0)\n    count = int(torch.searchsorted(cumulative, torch.tensor(p, dtype=probs.dtype))) + 1\n    return order[:min(count, len(probs))].tolist()\n",
        explanation="Nucleus Sampling 动态选择刚好覆盖目标概率质量的最小候选集合。",
        constraints=["probs 非空且和为 1", "0 < p <= 1"],
        company_tags=["OpenAI", "Meta", "字节跳动"],
    )

    add(
        slug="perplexity-from-token-losses",
        title="由 Token Loss 计算困惑度",
        difficulty="简单",
        category="大模型评估",
        function_name="perplexity",
        signature="def perplexity(token_losses: list[float]) -> float",
        description="给定一组使用自然对数计算的有效 token 交叉熵，返回 exp(mean(loss)) 形式的困惑度。",
        reference=lambda token_losses: float(torch.exp(torch.as_tensor(token_losses, dtype=torch.float64).mean())),
        raw_cases=[
            case([[0.0]]),
            case([[math.log(2.0)]]),
            case([[0.0, math.log(4.0)]]),
            case([[1.0, 1.0, 1.0]]),
            case([[0.1, 0.2, 0.3]]),
            case([[2.0, 3.0]]),
            case([[math.log(10.0), math.log(10.0)]]),
            case([[0.5]]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef perplexity(token_losses):\n    losses = torch.as_tensor(token_losses, dtype=torch.float64)\n    return float(torch.exp(losses.mean()))\n",
        explanation="困惑度是平均负对数似然的指数，越低通常表示语言模型对数据越不困惑。",
        constraints=["token_losses 非空", "元素为有限非负数", "使用自然对数"],
        company_tags=["OpenAI", "Google", "Meta", "百度"],
    )

    def ref_info_nce(a: Any, b: Any, temperature: float) -> float:
        left = torch.nn.functional.normalize(torch.as_tensor(a, dtype=torch.float64), dim=1)
        right = torch.nn.functional.normalize(torch.as_tensor(b, dtype=torch.float64), dim=1)
        logits = left @ right.transpose(0, 1) / temperature
        labels = torch.arange(left.shape[0])
        return float(torch.nn.functional.cross_entropy(logits, labels))

    add(
        slug="info-nce-loss",
        title="InfoNCE 对比学习损失",
        difficulty="困难",
        category="多模态与表征学习",
        function_name="info_nce_loss",
        signature="def info_nce_loss(a: Any, b: Any, temperature: float) -> float",
        description="两组形状 (N,D) 的 embedding 按行一一配对为正样本。先做 L2 归一化，再计算单向 a→b 的温度缩放 InfoNCE 损失。",
        reference=ref_info_nce,
        raw_cases=[
            case([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], 1.0]),
            case([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], 1.0]),
            case([[[1.0, 2.0]], [[2.0, 4.0]], 0.5]),
            case([[[1.0, 1.0], [1.0, -1.0]], [[2.0, 2.0], [2.0, -2.0]], 0.1]),
            case([[[1.0, 0.0], [-1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]], 2.0]),
            case([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], 0.7]),
            case([[[0.5, 0.5], [0.2, 0.8]], [[0.6, 0.4], [0.1, 0.9]], 1.0]),
            case([[[3.0, 4.0]], [[-3.0, -4.0]], 1.0]),
        ],
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef info_nce_loss(a, b, temperature):\n    a = F.normalize(torch.as_tensor(a, dtype=torch.float64), dim=1)\n    b = F.normalize(torch.as_tensor(b, dtype=torch.float64), dim=1)\n    logits = a @ b.transpose(0, 1) / temperature\n    labels = torch.arange(a.shape[0])\n    return float(F.cross_entropy(logits, labels))\n",
        explanation="归一化点积等于余弦相似度，对角线配对就是每行交叉熵的目标类别。",
        constraints=["a、b 形状相同且无零向量", "temperature > 0", "正样本位于对角线"],
        company_tags=["OpenAI", "Google", "Meta"],
    )

    def ref_distill(student: Any, teacher: Any, temperature: float) -> float:
        s = torch.as_tensor(student, dtype=torch.float64) / temperature
        t = torch.as_tensor(teacher, dtype=torch.float64) / temperature
        return float(
            torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(s, dim=-1),
                torch.nn.functional.softmax(t, dim=-1),
                reduction="batchmean",
            )
            * temperature**2
        )

    add(
        slug="knowledge-distillation-kl",
        title="知识蒸馏 KL Loss",
        difficulty="困难",
        category="模型压缩与部署",
        function_name="distillation_kl_loss",
        signature="def distillation_kl_loss(student_logits: Any, teacher_logits: Any, temperature: float) -> float",
        description="计算教师分布到学生分布的知识蒸馏 KL 损失，使用 batchmean 约简并乘 temperature²。",
        reference=ref_distill,
        raw_cases=[
            case([[[1.0, 2.0]], [[1.0, 2.0]], 1.0]),
            case([[[2.0, 0.0]], [[0.0, 2.0]], 1.0]),
            case([[[1.0, 2.0], [3.0, 1.0]], [[1.5, 1.5], [2.0, 2.0]], 2.0]),
            case([[[0.0, 0.0]], [[10.0, -10.0]], 4.0]),
            case([[[1000.0, 999.0]], [[999.0, 1000.0]], 1.0]),
            case([[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]], 0.5]),
            case([[[0.0], [1.0]], [[2.0], [-3.0]], 3.0]),
            case([[[-1.0, 1.0]], [[-2.0, 2.0]], 2.0]),
        ],
        imports=torch_imports(),
        solution_code="import torch\nimport torch.nn.functional as F\n\ndef distillation_kl_loss(student_logits, teacher_logits, temperature):\n    student = torch.as_tensor(student_logits, dtype=torch.float64) / temperature\n    teacher = torch.as_tensor(teacher_logits, dtype=torch.float64) / temperature\n    log_p = F.log_softmax(student, dim=-1)\n    q = F.softmax(teacher, dim=-1)\n    return float(F.kl_div(log_p, q, reduction='batchmean') * temperature ** 2)\n",
        explanation="PyTorch 的 kl_div 第一个参数应是学生 log-prob，第二个参数是教师 probability。",
        constraints=["两组 logits 形状一致且为二维", "temperature > 0", "使用 batchmean"],
        company_tags=["Google", "Microsoft", "华为", "百度"],
    )

    add(
        slug="global-average-pooling",
        title="全局平均池化",
        difficulty="简单",
        category="计算机视觉",
        function_name="global_average_pooling",
        signature="def global_average_pooling(x: Any) -> torch.Tensor",
        description="给定形状 (C,H,W) 的特征图，使用 PyTorch 对空间维做全局平均池化，返回形状 (C,) 的 Tensor。",
        reference=lambda x: torch.as_tensor(x, dtype=torch.float64).mean(dim=(-2, -1)),
        raw_cases=[
            case([[[[1.0, 2.0], [3.0, 4.0]]]]),
            case([[[[1.0]], [[2.0]]]]),
            case([[[[1.0, 1.0]], [[2.0, 4.0]]]]),
            case([[[[-1.0, 1.0], [2.0, -2.0]]]]),
            case([[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]]]),
            case([[[[5.0, 7.0, 9.0]]]]),
            case([[[[1.5], [2.5], [3.5]]]]),
            case([[[[1.0]], [[-1.0]], [[0.0]]]]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef global_average_pooling(x):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    return x.mean(dim=(-2, -1))\n",
        explanation="全局平均池化保留通道维，并对最后两个空间维同时求均值。",
        constraints=["x 为三维 (C,H,W) 输入", "H、W >= 1", "返回 torch.Tensor"],
        company_tags=["NVIDIA", "Google", "百度"],
    )

    def ref_topk_accuracy(logits: Any, labels: list[int], k: int) -> float:
        scores = torch.as_tensor(logits, dtype=torch.float64)
        target = torch.as_tensor(labels, dtype=torch.long)
        topk = torch.topk(scores, k, dim=1).indices
        return float((topk == target.unsqueeze(1)).any(dim=1).double().mean())

    add(
        slug="top-k-accuracy",
        title="分类 Top-K Accuracy",
        difficulty="中等",
        category="大模型评估",
        function_name="top_k_accuracy",
        signature="def top_k_accuracy(logits: Any, labels: list[int], k: int) -> float",
        description="给定 batch×classes 的 logits 和真实类别，返回真实类别落入每行 Top-K 预测的样本比例。",
        reference=ref_topk_accuracy,
        raw_cases=[
            case([[[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]], [1, 0], 1]),
            case([[[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]], [2, 1], 2]),
            case([[[1.0]], [0], 1]),
            case([[[0.1, 0.2, 0.3]], [0], 3]),
            case([[[-1.0, -2.0], [2.0, 3.0]], [0, 0], 1]),
            case([[[10.0, 0.0, -1.0], [0.0, 10.0, -1.0], [0.0, -1.0, 10.0]], [0, 1, 2], 1]),
            case([[[1.0, 2.0, 3.0, 4.0]], [1], 2]),
            case([[[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]], [3, 0], 3]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef top_k_accuracy(logits, labels, k):\n    logits = torch.as_tensor(logits, dtype=torch.float64)\n    labels = torch.as_tensor(labels, dtype=torch.long)\n    topk = torch.topk(logits, k, dim=1).indices\n    correct = (topk == labels.unsqueeze(1)).any(dim=1)\n    return float(correct.double().mean())\n",
        explanation="每行只需检查真实类别是否出现在 torch.topk 返回的下标集合中。",
        constraints=["1 <= k <= 类别数", "labels 长度等于 batch size", "测试数据不含并列边界"],
        company_tags=["Amazon", "Google", "Meta", "腾讯"],
    )

    def ref_quantize(x: Any, num_bits: int) -> torch.Tensor:
        value = torch.as_tensor(x, dtype=torch.float64)
        qmax = 2 ** (num_bits - 1) - 1
        max_abs = value.abs().max()
        if float(max_abs) == 0:
            return torch.zeros_like(value)
        scale = max_abs / qmax
        quantized = torch.clamp(torch.round(value / scale), -qmax, qmax)
        return quantized * scale

    add(
        slug="symmetric-quantize-dequantize",
        title="对称量化与反量化",
        difficulty="中等",
        category="模型压缩与部署",
        function_name="symmetric_quantize_dequantize",
        signature="def symmetric_quantize_dequantize(x: Any, num_bits: int) -> torch.Tensor",
        description="实现有符号、逐 Tensor、对称量化后立即反量化。整数范围使用 [−Q,Q]，Q=2^(num_bits−1)−1。",
        reference=ref_quantize,
        raw_cases=[
            case([[-1.0, 0.0, 1.0], 8]),
            case([[0.0, 0.0], 8]),
            case([[-2.0, 1.0, 2.0], 4]),
            case([[[1.0, -1.0], [0.5, -0.5]], 8]),
            case([[10.0], 2]),
            case([[-3.0, 0.7, 2.2], 3]),
            case([[0.1, 0.2, 0.3], 8]),
            case([[-100.0, 50.0, 0.0], 6]),
        ],
        imports=torch_imports(),
        solution_code="import torch\n\ndef symmetric_quantize_dequantize(x, num_bits):\n    x = torch.as_tensor(x, dtype=torch.float64)\n    qmax = 2 ** (num_bits - 1) - 1\n    max_abs = x.abs().max()\n    if float(max_abs) == 0:\n        return torch.zeros_like(x)\n    scale = max_abs / qmax\n    q = torch.clamp(torch.round(x / scale), -qmax, qmax)\n    return q * scale\n",
        explanation="对称量化用一个 scale 把浮点范围映射到有符号整数，再乘 scale 得到可比较的反量化值。",
        constraints=["2 <= num_bits <= 16", "x 非空", "返回 torch.Tensor"],
        company_tags=["NVIDIA", "Microsoft", "华为", "阿里巴巴"],
    )

    assert len(problems) == 80
    return problems
