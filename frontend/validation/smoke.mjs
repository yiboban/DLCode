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
if (problems.total < 92) throw new Error("题库数量不足 92");

const filtered = await getJson(`/api/problems?category=${encodeURIComponent("Python 与 PyTorch 基础")}`);
if (!filtered.items.length) throw new Error("分类筛选没有返回题目");

const detail = await getJson("/api/problems/matrix-transpose");
if (!detail.starter_code || detail.hidden_tests) throw new Error("题目详情字段不符合预期");
if (typeof detail.favorited !== "boolean") throw new Error("题目收藏状态缺失");
if (!detail.default_test_case) throw new Error("默认测试样例缺失");
if (!detail.lecture?.solution?.length || !detail.lecture?.functions?.length || !detail.lecture?.memory_tips?.length) {
  throw new Error("讲义模块字段缺失");
}
if (!detail.lecture.solutions?.[0]?.code.includes("def matrix_transpose")) throw new Error("讲义参考实现缺失");

await postJson(`/api/favorites/${detail.id}`, { favorited: true }, "PUT");
const favorites = await getJson("/api/favorites");
if (!favorites.some((item) => item.id === detail.id && item.favorited)) throw new Error("题目收藏失败");
await postJson(`/api/favorites/${detail.id}`, { favorited: false }, "PUT");

const relu = await getJson("/api/problems/relu-activation");
if (relu.lecture?.solutions?.length !== 2) throw new Error("直接调用 API 的题目没有提供双解法");
if (relu.lecture.solutions[0].kind !== "manual" || relu.lecture.solutions[1].kind !== "api") {
  throw new Error("手搓与 API 解法顺序不正确");
}

const clip = await getJson("/api/problems/clip-symmetric-loss");
if (clip.presentation?.formulas?.length !== 2) throw new Error("CLIP 对称损失公式缺失");
if (clip.lecture?.solutions?.length !== 2) throw new Error("CLIP 对称损失没有提供手搓与 API 双解法");

const positional = await getJson("/api/problems/sinusoidal-positional-encoding");
if (positional.presentation?.formulas?.length !== 2) throw new Error("正弦位置编码公式缺失");
if (!positional.starter_code.includes("torch")) throw new Error("正弦位置编码未使用 PyTorch 模板");

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
