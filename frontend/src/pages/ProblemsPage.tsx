import { ArrowDownUp, CheckCircle2, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import type { ProblemListItem, ProblemsResponse } from "../types";
import { cx, difficultyClass } from "../utils";

export default function ProblemsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [data, setData] = useState<ProblemsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const filters = useMemo(
    () => ({
      search: searchParams.get("search") ?? "",
      category: searchParams.get("category") ?? "",
      difficulty: searchParams.get("difficulty") ?? "",
      company: searchParams.get("company") ?? "",
      status: searchParams.get("status") ?? "",
      sort_by: searchParams.get("sort_by") ?? "id",
      order: searchParams.get("order") ?? "asc",
    }),
    [searchParams],
  );

  useEffect(() => {
    setLoading(true);
    api
      .problems(filters)
      .then((res) => {
        setData(res);
        setError("");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [filters]);

  function update(key: keyof typeof filters, value: string) {
    const next = new URLSearchParams(searchParams);
    if (value) next.set(key, value);
    else next.delete(key);
    setSearchParams(next);
  }

  return (
    <main className="mx-auto max-w-[1500px] px-4 py-5">
      <div className="mb-4 flex flex-col justify-between gap-3 md:flex-row md:items-end">
        <div>
          <h1 className="text-2xl font-bold">题库</h1>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
            按分类、难度、公司标签和完成状态筛选机器学习手撕题。
          </p>
        </div>
        <div className="text-sm text-zinc-500 dark:text-zinc-400">当前结果：{data?.total ?? 0} 道</div>
      </div>

      <section className="mb-4 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="grid gap-3 md:grid-cols-[1.4fr_repeat(5,1fr)]">
          <label className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400" size={17} />
            <input
              value={filters.search}
              onChange={(event) => update("search", event.target.value)}
              placeholder="搜索题名或题号"
              className="h-10 w-full rounded-md border border-zinc-300 bg-white pl-9 pr-3 text-sm outline-none focus:border-moss dark:border-zinc-700 dark:bg-zinc-950"
            />
          </label>
          <Select value={filters.category} onChange={(value) => update("category", value)} options={data?.categories ?? []} placeholder="全部分类" />
          <Select value={filters.difficulty} onChange={(value) => update("difficulty", value)} options={["简单", "中等", "困难"]} placeholder="全部难度" />
          <Select value={filters.company} onChange={(value) => update("company", value)} options={data?.companies ?? []} placeholder="全部公司" />
          <Select value={filters.status} onChange={(value) => update("status", value)} options={["已完成", "未完成"]} placeholder="全部状态" />
          <Select
            value={`${filters.sort_by}:${filters.order}`}
            onChange={(value) => {
              const [sortBy, order] = value.split(":");
              const next = new URLSearchParams(searchParams);
              next.set("sort_by", sortBy);
              next.set("order", order);
              setSearchParams(next);
            }}
            options={["id:asc", "id:desc", "difficulty:asc", "difficulty:desc", "acceptance_rate:desc", "acceptance_rate:asc"]}
            labels={{
              "id:asc": "题号升序",
              "id:desc": "题号降序",
              "difficulty:asc": "难度升序",
              "difficulty:desc": "难度降序",
              "acceptance_rate:desc": "通过率降序",
              "acceptance_rate:asc": "通过率升序",
            }}
            placeholder="排序"
          />
        </div>
      </section>

      {error && <div className="mb-4 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{error}</div>}

      <section className="overflow-hidden rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="hidden grid-cols-[80px_1.2fr_120px_1fr_1fr_110px_90px] gap-3 border-b border-zinc-200 px-4 py-3 text-xs font-semibold text-zinc-500 md:grid dark:border-zinc-800">
          <div>题号</div>
          <div>题目名称</div>
          <div>难度</div>
          <div>分类</div>
          <div>公司标签</div>
          <div className="flex items-center gap-1">
            通过率 <ArrowDownUp size={13} />
          </div>
          <div>状态</div>
        </div>
        {loading ? (
          <div className="p-8 text-center text-sm text-zinc-500">题库加载中...</div>
        ) : (
          <div className="divide-y divide-zinc-100 dark:divide-zinc-800">
            {(data?.items ?? []).map((problem) => (
              <ProblemRow key={problem.id} problem={problem} />
            ))}
            {data?.items.length === 0 && <div className="p-8 text-center text-sm text-zinc-500">没有符合条件的题目。</div>}
          </div>
        )}
      </section>
    </main>
  );
}

function ProblemRow({ problem }: { problem: ProblemListItem }) {
  return (
    <Link
      to={`/problems/${problem.slug}`}
      className="grid gap-2 px-4 py-4 text-sm hover:bg-zinc-50 md:grid-cols-[80px_1.2fr_120px_1fr_1fr_110px_90px] md:items-center dark:hover:bg-zinc-800/70"
    >
      <div className="text-zinc-500">#{problem.id}</div>
      <div className="font-medium">
        <span className="mr-2 md:hidden">题目：</span>
        {problem.title}
      </div>
      <div>
        <span className={`rounded px-2 py-1 text-xs ${difficultyClass(problem.difficulty)}`}>{problem.difficulty}</span>
      </div>
      <div className="truncate text-zinc-600 dark:text-zinc-300">{problem.categories.join("、")}</div>
      <div className="flex flex-wrap gap-1">
        {problem.company_tags.slice(0, 2).map((tag) => (
          <span key={tag} className="rounded border border-zinc-200 px-2 py-0.5 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
            {tag}
          </span>
        ))}
      </div>
      <div>{problem.acceptance_rate.toFixed(1)}%</div>
      <div className={cx("inline-flex items-center gap-1", problem.completed ? "text-emerald-600" : "text-zinc-400")}>
        {problem.completed && <CheckCircle2 size={16} />}
        {problem.completed ? "已完成" : "未完成"}
      </div>
    </Link>
  );
}

function Select({
  value,
  onChange,
  options,
  placeholder,
  labels = {},
}: {
  value: string;
  onChange: (value: string) => void;
  options: string[];
  placeholder: string;
  labels?: Record<string, string>;
}) {
  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-moss dark:border-zinc-700 dark:bg-zinc-950"
    >
      <option value="">{placeholder}</option>
      {options.map((option) => (
        <option key={option} value={option}>
          {labels[option] ?? option}
        </option>
      ))}
    </select>
  );
}
