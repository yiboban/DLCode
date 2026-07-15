# DLCode：深度学习手撕题库

DLCode 是一个本地运行的机器学习编程题练习网站，用于练习 Python、NumPy、传统机器学习、深度学习、PyTorch、Attention、Transformer、计算机视觉、自然语言处理和训练调试相关手撕题。

项目交互方式参考主流在线判题产品，但不使用 LeetCode 的商标、Logo、页面代码或受版权保护题目内容。首批题目来自常见公开知识点整理与原创改编，公司标签仅表示相似高频方向，不声明为真实公司原题。

## 功能截图位置

截图可放在 `docs/screenshots/` 目录。当前目录已预留，建议保存：

- 首页：`docs/screenshots/home.png`
- 题库列表：`docs/screenshots/problems.png`
- 题目详情与判题结果：`docs/screenshots/problem-detail.png`
- 提交记录：`docs/screenshots/submissions.png`

## 技术栈

- 前端：React、TypeScript、Vite、Tailwind CSS
- 代码编辑器：Monaco Editor
- 后端：FastAPI、Pydantic
- 数据库：SQLite、SQLAlchemy
- 判题语言：Python 3
- 数值库：NumPy、PyTorch
- 测试：pytest、前端 TypeScript 构建校验

## 项目目录结构

```text
DLCode/
  backend/
    app/
      main.py            # FastAPI 接口与启动初始化
      judge.py           # 本地 Python 判题核心
      models.py          # SQLAlchemy 数据模型
      schemas.py         # Pydantic 请求与响应模型
      problem_bank.py    # 首批 60 道结构化题目
      database.py        # SQLite 连接
    tests/
      test_api.py        # 后端接口与判题测试
  frontend/
    src/
      pages/             # 首页、题库、详情、提交记录
      api.ts             # 前端 API 客户端
      types.ts           # TypeScript 类型
      App.tsx
    package.json
  docs/screenshots/
  requirements.txt
  package.json
  start.bat
  start.sh
```

## 环境要求

- Python 3.10 或更高版本
- Node.js 18 或更高版本
- npm 9 或更高版本
- 本地可安装 Python 和 npm 依赖

## 安装步骤

Windows、macOS、Linux 通用：

```bash
npm run install:all
```

也可以分步安装：

```bash
python -m pip install -r requirements.txt
npm install
npm --prefix frontend install
```

## 启动步骤

同时启动前后端：

```bash
npm run dev
```

Windows 也可以运行：

```bat
start.bat
```

macOS / Linux：

```bash
chmod +x start.sh
./start.sh
```

访问地址：

- 前端：http://localhost:5173
- 后端：http://localhost:8000
- 接口文档：http://localhost:8000/docs

首次启动时会自动创建 `dlcode.db` 并写入内置题库，不需要手动建表或导入数据。

## 测试命令

```bash
npm run test
```

等价于：

```bash
python -m pytest backend/tests -q
npm --prefix frontend run build
```

后端测试覆盖题目列表、题目详情、隐藏测试隔离、正确提交、错误答案、语法错误、运行错误、无限循环超时、浮点误差、NumPy 数组、PyTorch Tensor、提交记录和草稿保存恢复。

## 题库规模

首批内置 60 道可运行、可提交的题目：

- Python 与 NumPy 基础：10 道
- 传统机器学习：10 道
- 深度学习基础：10 道
- PyTorch 基础：10 道
- Attention 与 Transformer：8 道
- 计算机视觉：5 道
- 自然语言处理：4 道
- 训练、调试与工程：3 道

每题包含 3 个公开测试和 5 个隐藏测试。普通题目接口不会返回 `solution_code` 和 `hidden_tests`。

## 添加新题目的方法

当前版本的题目数据集中在 `backend/app/problem_bank.py`。新增题目时：

1. 在 `get_seed_problems()` 中新增一个 `add(...)` 结构化题目。
2. 提供 `slug`、`title`、`difficulty`、`category`、`function_name`、`signature`、中文 `description`、`solution_code`、`explanation`、`constraints`。
3. 提供至少 8 个 `raw_cases`，前 3 个会作为公开测试，后 5 个作为隐藏测试。
4. 提供 `reference` 函数，启动时会用它生成测试期望值。
5. 重启后端，数据库会按 `slug` 同步题目信息。

后续更适合把每题拆成独立 JSON/YAML 文件或 Python 模块，使扩展更细粒度。

## 题目数据格式

每道题在数据库中包含：

```text
id
slug
title
difficulty
categories
company_tags
source_note
description
function_name
function_signature
starter_code
solution_code
explanation
constraints
examples
public_tests
hidden_tests
time_limit
memory_limit
```

前端题目详情接口只返回公开字段，不返回参考答案和隐藏测试。

## 判题流程

“运行代码”执行公开测试或用户填写的自定义测试。

“提交答案”执行公开测试和隐藏测试，并保存提交记录、题目完成状态、最近一次结果、通过次数和提交次数。

判题状态包括：

- 通过
- 答案错误
- 运行错误
- 语法错误
- 超出时间限制
- 内存超出限制
- 输出格式错误
- 系统错误

判题器会递归比较 Python 数字、列表、字典、NumPy 数组和 PyTorch Tensor。浮点比较使用 `atol=1e-6`、`rtol=1e-5`，并处理 `NaN`、正无穷和负无穷。

## 安全边界说明

本项目是可信本地学习环境使用的练习系统，不是公网生产级代码沙箱。

当前判题器会通过独立 Python 子进程运行用户代码，并实现：

- 总运行超时
- 临时目录执行
- 运行结束清理临时文件
- 捕获标准输出和标准错误
- 限制返回给前端的输出长度
- 捕获语法错误、运行错误和超时
- 子进程不直接连接题库数据库
- Linux/macOS 下尽量使用 `resource` 设置内存上限，Windows 下以内存错误捕获为主

这些措施不能阻止所有恶意代码行为。不要把该服务直接暴露到公网，也不要在不可信用户场景中运行。

## 常见问题

1. **前端打不开**
   确认 `npm run dev` 中前端服务已经启动，并访问 `http://localhost:5173`。

2. **后端接口失败**
   确认 `http://localhost:8000/api/health` 返回 `{"status":"ok"}`。

3. **PyTorch 安装很慢**
   可以先确认本机是否已有 PyTorch：`python -c "import torch; print(torch.__version__)"`。

4. **草稿没有恢复**
   草稿保存在 SQLite 的 `drafts` 表中。确认没有删除 `dlcode.db`。

5. **隐藏测试能否看到**
   普通 API 不返回隐藏测试。后端源码中的题库数据用于本地开发测试，不会发送给前端。

## 后续开发计划

- 将题目拆分为独立结构化文件，增加题目导入校验命令
- 增加更完整的前端自动化测试
- 增加题解折叠显示和复杂度标注
- 增加按知识点生成练习计划
- 增加本地错题本和收藏功能
- 增加更严格的沙箱方案，例如容器隔离或专用判题服务
