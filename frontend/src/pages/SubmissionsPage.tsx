import { ClipboardList, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { SubmissionDetail, SubmissionItem } from "../types";
import { formatValue, statusClass } from "../utils";

export default function SubmissionsPage() {
  const [items, setItems] = useState<SubmissionItem[]>([]);
  const [selected, setSelected] = useState<SubmissionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .submissions()
      .then(setItems)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  async function openDetail(id: number) {
    try {
      setSelected(await api.submission(id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "读取提交记录失败");
    }
  }

  return (
    <main className="mx-auto max-w-[1500px] px-4 py-5">
      <div className="mb-4 flex items-center gap-3">
        <ClipboardList className="text-moss" />
        <div>
          <h1 className="text-2xl font-bold">提交记录</h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">查看提交时间、状态、运行时间和错误样例。</p>
        </div>
      </div>

      {error && <div className="mb-4 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{error}</div>}

      <section className="overflow-hidden rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="hidden grid-cols-[180px_1fr_110px_120px_120px_100px] gap-3 border-b border-zinc-200 px-4 py-3 text-xs font-semibold text-zinc-500 md:grid dark:border-zinc-800">
          <div>提交时间</div>
          <div>题目</div>
          <div>提交状态</div>
          <div>运行时间</div>
          <div>通过测试</div>
          <div>语言</div>
        </div>
        {loading ? (
          <div className="p-8 text-center text-sm text-zinc-500">提交记录加载中...</div>
        ) : items.length === 0 ? (
          <div className="p-8 text-center text-sm text-zinc-500">暂无提交记录。</div>
        ) : (
          <div className="divide-y divide-zinc-100 dark:divide-zinc-800">
            {items.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => void openDetail(item.id)}
                className="grid w-full gap-2 px-4 py-4 text-left text-sm hover:bg-zinc-50 md:grid-cols-[180px_1fr_110px_120px_120px_100px] md:items-center dark:hover:bg-zinc-800/70"
              >
                <div className="text-zinc-500">{new Date(item.created_at).toLocaleString()}</div>
                <div className="font-medium">
                  <Link to={`/problems/${item.problem_slug}`} className="hover:text-moss">
                    {item.problem_title}
                  </Link>
                </div>
                <div>
                  <span className={`rounded px-2 py-1 text-xs ${statusClass(item.status)}`}>{item.status}</span>
                </div>
                <div>{item.runtime_ms.toFixed(2)} ms</div>
                <div>
                  {item.passed_tests}/{item.total_tests}
                </div>
                <div>{item.language}</div>
              </button>
            ))}
          </div>
        )}
      </section>

      {selected && (
        <div className="fixed inset-0 z-50 bg-black/50 p-4" onClick={() => setSelected(null)}>
          <div
            className="mx-auto flex max-h-[92vh] max-w-5xl flex-col rounded-lg border border-zinc-200 bg-white shadow-xl dark:border-zinc-800 dark:bg-zinc-900"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-800">
              <div>
                <h2 className="text-lg font-semibold">{selected.problem_title}</h2>
                <div className="mt-1 text-sm text-zinc-500">
                  {new Date(selected.created_at).toLocaleString()} · {selected.language}
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="rounded-md p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                title="关闭"
              >
                <X size={18} />
              </button>
            </div>
            <div className="scrollbar-thin grid min-h-0 gap-4 overflow-y-auto p-5 lg:grid-cols-2">
              <div>
                <div className="mb-2 flex items-center gap-2">
                  <span className={`rounded px-2 py-1 text-xs ${statusClass(selected.status)}`}>{selected.status}</span>
                  <span className="text-sm text-zinc-500">
                    {selected.passed_tests}/{selected.total_tests} · {selected.runtime_ms.toFixed(2)} ms
                  </span>
                </div>
                <pre className="max-h-[62vh] overflow-auto rounded-md bg-zinc-50 p-3 font-mono text-xs dark:bg-zinc-950">{selected.code}</pre>
              </div>
              <div>
                <h3 className="mb-2 text-sm font-semibold">错误样例</h3>
                {selected.error_sample ? (
                  <div className="space-y-3 text-xs">
                    <Field label="输入" value={selected.error_sample.input} />
                    <Field label="预期输出" value={selected.error_sample.expected_output} />
                    <Field label="实际输出" value={selected.error_sample.actual_output} />
                    {selected.error_sample.error_message && (
                      <Field label="错误信息" value={`${selected.error_sample.error_type ?? ""}: ${selected.error_sample.error_message}`} />
                    )}
                    {selected.error_sample.traceback && <Field label="错误堆栈" value={selected.error_sample.traceback} />}
                  </div>
                ) : (
                  <div className="rounded-md bg-zinc-50 p-4 text-sm text-zinc-500 dark:bg-zinc-950">本次提交没有错误样例。</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function Field({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <div className="mb-1 text-zinc-500">{label}</div>
      <pre className="max-h-36 overflow-auto rounded-md bg-zinc-50 p-2 font-mono text-[12px] dark:bg-zinc-950">
        {typeof value === "string" ? value : formatValue(value)}
      </pre>
    </div>
  );
}
