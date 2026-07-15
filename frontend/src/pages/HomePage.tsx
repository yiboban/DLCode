import { ArrowRight, CheckCircle2, Layers, ListChecks } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { StatisticsResponse } from "../types";
import { statusClass } from "../utils";

const paths = [
  "Python 与 PyTorch 基础",
  "传统机器学习",
  "深度学习基础",
  "PyTorch 基础",
  "Attention 与 Transformer",
  "大模型核心组件",
  "大模型推理与解码",
  "优化器与训练",
  "多模态与表征学习",
  "模型压缩与部署",
  "计算机视觉",
  "自然语言处理",
  "训练、调试与工程",
];

export default function HomePage() {
  const [stats, setStats] = useState<StatisticsResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.statistics().then(setStats).catch((err: Error) => setError(err.message));
  }, []);

  return (
    <main className="mx-auto max-w-[1500px] px-4 py-6">
      <section className="grid gap-5 lg:grid-cols-[1.25fr_0.75fr]">
        <div className="rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mb-4 inline-flex rounded-md bg-moss/10 px-3 py-1 text-sm font-medium text-moss dark:text-emerald-300">
            本地机器学习编程练习
          </div>
          <h1 className="text-3xl font-bold tracking-normal md:text-4xl">DLCode：深度学习手撕题库</h1>
          <p className="mt-3 max-w-3xl text-zinc-600 dark:text-zinc-300">
            面向互联网面试中常见的机器学习、深度学习、PyTorch、Attention 和工程调试题，支持本地运行、提交判题、错误样例回放和草稿保存。
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              to="/problems"
              className="inline-flex items-center gap-2 rounded-md bg-moss px-4 py-2 text-sm font-semibold text-white hover:bg-moss/90"
            >
              进入题库
              <ArrowRight size={17} />
            </Link>
            <Link
              to="/submissions"
              className="inline-flex items-center gap-2 rounded-md border border-zinc-300 px-4 py-2 text-sm font-semibold text-zinc-700 hover:border-moss hover:text-moss dark:border-zinc-700 dark:text-zinc-200"
            >
              查看提交记录
            </Link>
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
          <Metric icon={Layers} label="题目总数" value={stats?.total_problems ?? "--"} />
          <Metric icon={CheckCircle2} label="已完成题目" value={stats?.completed_problems ?? "--"} />
        </div>
      </section>

      {error && <div className="mt-4 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{error}</div>}

      <section className="mt-5 grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
        <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold">
            <ListChecks size={19} />
            难度完成情况
          </h2>
          <div className="space-y-4">
            {(["简单", "中等", "困难"] as const).map((difficulty) => {
              const item = stats?.by_difficulty[difficulty];
              const total = item?.total ?? 0;
              const completed = item?.completed ?? 0;
              const percent = total ? Math.round((completed / total) * 100) : 0;
              return (
                <div key={difficulty}>
                  <div className="mb-1 flex justify-between text-sm">
                    <span>{difficulty}</span>
                    <span className="text-zinc-500 dark:text-zinc-400">
                      {completed}/{total}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-zinc-100 dark:bg-zinc-800">
                    <div className="h-2 rounded-full bg-moss" style={{ width: `${percent}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="mb-4 text-lg font-semibold">推荐练习路径</h2>
          <div className="grid gap-2 sm:grid-cols-2">
            {paths.map((path, index) => (
              <Link
                key={path}
                to="/problems"
                className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2 text-sm hover:border-moss hover:text-moss dark:border-zinc-800"
              >
                <span>
                  {index + 1}. {path}
                </span>
                <ArrowRight size={15} />
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-[1fr_1fr]">
        <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="mb-4 text-lg font-semibold">最近提交记录</h2>
          <div className="space-y-2">
            {stats?.recent_submissions.length ? (
              stats.recent_submissions.map((item) => (
                <Link
                  key={item.id}
                  to="/submissions"
                  className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2 text-sm dark:border-zinc-800"
                >
                  <span className="truncate">{item.problem_title}</span>
                  <span className={`rounded px-2 py-1 text-xs ${statusClass(item.status)}`}>{item.status}</span>
                </Link>
              ))
            ) : (
              <div className="rounded-md bg-zinc-50 p-4 text-sm text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
                暂无提交记录，完成第一题后这里会显示最近结果。
              </div>
            )}
          </div>
        </div>

        <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="mb-4 text-lg font-semibold">分类入口</h2>
          <div className="flex flex-wrap gap-2">
            {(stats?.categories ?? paths).map((category) => (
              <Link
                key={category}
                to={`/problems?category=${encodeURIComponent(category)}`}
                className="rounded-md border border-zinc-200 px-3 py-2 text-sm hover:border-moss hover:text-moss dark:border-zinc-800"
              >
                {category}
              </Link>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

function Metric({ icon: Icon, label, value }: { icon: typeof Layers; label: string; value: number | string }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
      <Icon className="mb-4 text-moss" size={24} />
      <div className="text-sm text-zinc-500 dark:text-zinc-400">{label}</div>
      <div className="mt-1 text-3xl font-bold">{value}</div>
    </div>
  );
}
