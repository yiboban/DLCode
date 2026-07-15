const frontend = "http://localhost:5173";
const api = "http://localhost:8000";

async function getJson(path) {
  const response = await fetch(`${api}${path}`);
  if (!response.ok) throw new Error(`${path} 返回 ${response.status}`);
  return response.json();
}

async function postJson(path, body, method = "POST") {
  const response = await fetch(`${api}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${path} 返回 ${response.status}`);
  return response.json();
}

const html = await fetch(frontend).then((response) => response.text());
if (!html.includes("DLCode")) throw new Error("前端首页未返回 DLCode 页面");

const problems = await getJson("/api/problems");
if (problems.total < 60) throw new Error("题库数量不足 60");

const filtered = await getJson(`/api/problems?category=${encodeURIComponent("Python 与 NumPy 基础")}`);
if (!filtered.items.length) throw new Error("分类筛选没有返回题目");

const detail = await getJson("/api/problems/matrix-transpose");
if (!detail.starter_code || detail.hidden_tests) throw new Error("题目详情字段不符合预期");

const draftCode = "def matrix_transpose(matrix):\n    return matrix\n";
await postJson(`/api/drafts/${detail.id}`, { code: draftCode }, "PUT");
const draft = await getJson(`/api/drafts/${detail.id}`);
if (draft.code !== draftCode) throw new Error("草稿保存恢复失败");

const wrong = await postJson("/api/run", { problem_id: detail.id, code: draftCode });
if (wrong.status !== "答案错误") throw new Error("错误样例验证失败");

const correctCode = "def matrix_transpose(matrix):\n    return [list(col) for col in zip(*matrix)] if matrix else []\n";
const accepted = await postJson("/api/submit", { problem_id: detail.id, code: correctCode });
if (accepted.status !== "通过") throw new Error("正确提交没有通过");

console.log("前端与接口冒烟验证通过");
