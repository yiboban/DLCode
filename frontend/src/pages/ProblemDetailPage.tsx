import Editor from "@monaco-editor/react";
import type { editor as MonacoEditor } from "monaco-editor";
import { Copy, Loader2, Play, RotateCcw, Send, Settings } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import type { JudgeResponse, ProblemDetail } from "../types";
import { difficultyClass, formatValue, parseUserCodeLine, statusClass } from "../utils";

type MonacoApi = typeof import("monaco-editor");

export default function ProblemDetailPage() {
  const { slug = "" } = useParams();
  const [problem, setProblem] = useState<ProblemDetail | null>(null);
  const [code, setCode] = useState("");
  const [customTests, setCustomTests] = useState("");
  const [fontSize, setFontSize] = useState(() => Number(localStorage.getItem("dlcode-font-size") ?? 15));
  const [leftWidth, setLeftWidth] = useState(() => Number(localStorage.getItem("dlcode-left-width") ?? 46));
  const [runResult, setRunResult] = useState<JudgeResponse | null>(null);
  const [submitResult, setSubmitResult] = useState<JudgeResponse | null>(null);
  const [loadingRun, setLoadingRun] = useState(false);
  const [loadingSubmit, setLoadingSubmit] = useState(false);
  const [error, setError] = useState("");
  const [draftLoaded, setDraftLoaded] = useState(false);
  const editorRef = useRef<MonacoEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<MonacoApi | null>(null);
  const decorationsRef = useRef<MonacoEditor.IEditorDecorationsCollection | null>(null);
  const runCodeRef = useRef<() => Promise<void>>(async () => undefined);
  const submitCodeRef = useRef<() => Promise<void>>(async () => undefined);

  useEffect(() => {
    setDraftLoaded(false);
    api
      .problem(slug)
      .then(async (item) => {
        setProblem(item);
        const draft = await api.draft(item.id);
        setCode(draft.code || item.starter_code);
        setDraftLoaded(true);
      })
      .catch((err: Error) => setError(err.message));
  }, [slug]);

  useEffect(() => {
    localStorage.setItem("dlcode-font-size", String(fontSize));
  }, [fontSize]);

  useEffect(() => {
    localStorage.setItem("dlcode-left-width", String(leftWidth));
  }, [leftWidth]);

  useEffect(() => {
    if (!problem || !draftLoaded) return;
    const timer = window.setTimeout(() => {
      api.saveDraft(problem.id, code).catch(() => undefined);
    }, 700);
    return () => window.clearTimeout(timer);
  }, [code, draftLoaded, problem]);

  const highlightError = useCallback((traceback?: string | null) => {
    const line = parseUserCodeLine(traceback);
    if (!editorRef.current || !line) {
      decorationsRef.current?.clear();
      return;
    }
    const monaco = monacoRef.current;
    if (!monaco) return;
    decorationsRef.current?.clear();
    decorationsRef.current = editorRef.current.createDecorationsCollection([
      {
        range: new monaco.Range(line, 1, line, 1),
        options: {
          isWholeLine: true,
          className: "error-line-highlight",
          glyphMarginClassName: "error-line-highlight",
        },
      },
    ]);
    editorRef.current.revealLineInCenter(line);
  }, []);

  const parseCustomTests = useCallback(() => {
    if (!customTests.trim()) return undefined;
    const parsed = JSON.parse(customTests) as unknown;
    if (!Array.isArray(parsed)) throw new Error("自定义测试用例必须是 JSON 数组。");
    return parsed as Array<Record<string, unknown>>;
  }, [customTests]);

  const runCode = useCallback(async () => {
    if (!problem || loadingRun || loadingSubmit) return;
    try {
      setLoadingRun(true);
      setError("");
      const result = await api.run(problem.id, code, parseCustomTests());
      setRunResult(result);
      highlightError(result.first_error?.traceback);
    } catch (err) {
      setError(err instanceof Error ? err.message : "运行失败");
    } finally {
      setLoadingRun(false);
    }
  }, [code, highlightError, loadingRun, loadingSubmit, parseCustomTests, problem]);

  const submitCode = useCallback(async () => {
    if (!problem || loadingRun || loadingSubmit) return;
    try {
      setLoadingSubmit(true);
      setError("");
      const result = await api.submit(problem.id, code);
      setSubmitResult(result);
      highlightError(result.first_error?.traceback);
    } catch (err) {
      setError(err instanceof Error ? err.message : "提交失败");
    } finally {
      setLoadingSubmit(false);
    }
  }, [code, highlightError, loadingRun, loadingSubmit, problem]);

  useEffect(() => {
    runCodeRef.current = runCode;
    submitCodeRef.current = submitCode;
  }, [runCode, submitCode]);

  function startDrag(event: React.MouseEvent<HTMLDivElement>) {
    event.preventDefault();
    const move = (moveEvent: MouseEvent) => {
      const percent = Math.min(65, Math.max(32, (moveEvent.clientX / window.innerWidth) * 100));
      setLeftWidth(percent);
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function handleEditorMount(editor: MonacoEditor.IStandaloneCodeEditor, monaco: MonacoApi) {
    editorRef.current = editor;
    monacoRef.current = monaco;
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => void runCodeRef.current());
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => void submitCodeRef.current());
  }

  if (error && !problem) {
    return <main className="p-6 text-rose-700">{error}</main>;
  }

  if (!problem) {
    return <main className="p-6 text-sm text-zinc-500">题目加载中...</main>;
  }

  return (
    <main className="h-[calc(100vh-56px)] overflow-hidden">
      <div
        className="grid h-full"
        style={{ gridTemplateColumns: `${leftWidth}% 8px minmax(0, 1fr)` }}
      >
        <section className="scrollbar-thin overflow-y-auto border-r border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
          <div className="mb-3 text-sm text-zinc-500">
            <Link to="/problems" className="hover:text-moss">
              题库
            </Link>
            <span className="mx-2">/</span>
            #{problem.id}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="mr-2 text-2xl font-bold">{problem.title}</h1>
            <span className={`rounded px-2 py-1 text-xs ${difficultyClass(problem.difficulty)}`}>{problem.difficulty}</span>
            {problem.completed && <span className="rounded bg-emerald-50 px-2 py-1 text-xs text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">已通过</span>}
          </div>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            {problem.categories.map((item) => (
              <span key={item} className="rounded border border-zinc-200 px-2 py-1 dark:border-zinc-700">
                {item}
              </span>
            ))}
            {problem.company_tags.map((tag) => (
              <span key={tag} className="rounded border border-zinc-200 px-2 py-1 text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
                {tag}
              </span>
            ))}
          </div>
          <p className="mt-4 text-sm leading-7 text-zinc-700 dark:text-zinc-300">{problem.description}</p>
          <InfoBlock title="函数签名">
            <code className="block rounded-md bg-zinc-100 p-3 text-sm dark:bg-zinc-950">{problem.function_signature}</code>
          </InfoBlock>
          <InfoBlock title="参数说明">
            <p>请按照函数签名接收参数。数组类输入会由判题器转换为对应的 Python、NumPy 或 PyTorch 对象。</p>
          </InfoBlock>
          <InfoBlock title="返回值说明">
            <p>返回值会按结构递归比较，浮点数、NumPy 数组和 PyTorch Tensor 支持误差比较。</p>
          </InfoBlock>
          <InfoBlock title="示例">
            <div className="space-y-3">
              {problem.examples.map((example, index) => (
                <pre key={index} className="overflow-auto rounded-md bg-zinc-100 p-3 text-xs dark:bg-zinc-950">
{`输入：${String(example.input)}
输出：${String(example.output)}
说明：${String(example.explanation ?? "")}`}
                </pre>
              ))}
            </div>
          </InfoBlock>
          <InfoBlock title="限制条件">
            <ul className="list-disc space-y-1 pl-5">
              {problem.constraints.map((item) => (
                <li key={item}>{item}</li>
              ))}
              <li>时间限制：{problem.time_limit} 秒；内存限制：{problem.memory_limit} MB。</li>
            </ul>
          </InfoBlock>
          <InfoBlock title="解题提示">
            <p>{problem.explanation}</p>
          </InfoBlock>
          <InfoBlock title="相关知识点">
            <p>{problem.categories.join("、")}；{problem.source_note}</p>
          </InfoBlock>
        </section>

        <div
          role="separator"
          onMouseDown={startDrag}
          className="cursor-col-resize bg-zinc-200 hover:bg-moss dark:bg-zinc-800"
          title="拖动调整左右宽度"
        />

        <section className="flex min-w-0 flex-col bg-zinc-50 dark:bg-zinc-950">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center gap-2 text-sm">
              <Settings size={17} />
              <span>Python 3</span>
              <label className="ml-3 flex items-center gap-2">
                字号
                <input
                  type="range"
                  min={12}
                  max={22}
                  value={fontSize}
                  onChange={(event) => setFontSize(Number(event.target.value))}
                />
                <span className="w-6 text-right">{fontSize}</span>
              </label>
            </div>
            <div className="flex gap-2">
              <ActionButton icon={Play} label="运行代码" loading={loadingRun} disabled={loadingRun || loadingSubmit} onClick={runCode} />
              <ActionButton icon={Send} label="提交答案" loading={loadingSubmit} disabled={loadingRun || loadingSubmit} onClick={submitCode} primary />
              <button
                type="button"
                onClick={() => setCode(problem.starter_code)}
                className="inline-flex items-center gap-2 rounded-md border border-zinc-300 px-3 py-2 text-sm hover:border-moss hover:text-moss dark:border-zinc-700"
              >
                <RotateCcw size={16} />
                重置代码
              </button>
            </div>
          </div>

          {error && <div className="border-b border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-700">{error}</div>}

          <div className="min-h-0 flex-1">
            <Editor
              height="100%"
              language="python"
              theme={document.documentElement.classList.contains("dark") ? "vs-dark" : "light"}
              value={code}
              onChange={(value) => setCode(value ?? "")}
              onMount={handleEditorMount}
              options={{
                fontSize,
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 4,
                glyphMargin: true,
              }}
            />
          </div>

          <div className="scrollbar-thin max-h-[42%] overflow-y-auto border-t border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
              <div>
                <label className="mb-2 block text-sm font-semibold">自定义测试用例</label>
                <textarea
                  value={customTests}
                  onChange={(event) => setCustomTests(event.target.value)}
                  placeholder={'可选，JSON 数组，例如：\n[\n  {"args": [[[1, 2], [3, 4]]], "expected": [[1, 3], [2, 4]]}\n]'}
                  className="h-40 w-full resize-none rounded-md border border-zinc-300 bg-white p-3 font-mono text-xs outline-none focus:border-moss dark:border-zinc-700 dark:bg-zinc-950"
                />
                <div className="mt-2 text-xs text-zinc-500">Ctrl + Enter 运行代码；Ctrl + Shift + Enter 提交答案。macOS 可使用 Command。</div>
              </div>
              <div className="grid gap-3 lg:grid-cols-2">
                <ResultPanel title="运行结果" result={runResult} />
                <ResultPanel title="提交结果" result={submitResult} />
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

function InfoBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-6">
      <h2 className="mb-2 text-base font-semibold">{title}</h2>
      <div className="text-sm leading-7 text-zinc-700 dark:text-zinc-300">{children}</div>
    </section>
  );
}

function ActionButton({
  icon: Icon,
  label,
  loading,
  disabled,
  onClick,
  primary = false,
}: {
  icon: typeof Play;
  label: string;
  loading: boolean;
  disabled: boolean;
  onClick: () => void;
  primary?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={
        primary
          ? "inline-flex items-center gap-2 rounded-md bg-moss px-3 py-2 text-sm font-semibold text-white hover:bg-moss/90 disabled:cursor-not-allowed disabled:opacity-60"
          : "inline-flex items-center gap-2 rounded-md border border-zinc-300 px-3 py-2 text-sm font-semibold hover:border-moss hover:text-moss disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-700"
      }
    >
      {loading ? <Loader2 className="animate-spin" size={16} /> : <Icon size={16} />}
      {loading ? "运行中..." : label}
    </button>
  );
}

function ResultPanel({ title, result }: { title: string; result: JudgeResponse | null }) {
  const first = result?.first_error ?? result?.results[0] ?? null;
  return (
    <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold">{title}</h3>
        {result && <span className={`rounded px-2 py-1 text-xs ${statusClass(result.status)}`}>{result.status}</span>}
      </div>
      {!result ? (
        <div className="rounded-md bg-zinc-50 p-4 text-sm text-zinc-500 dark:bg-zinc-950">暂无结果。</div>
      ) : (
        <div className="space-y-2 text-xs">
          <div className="grid grid-cols-3 gap-2 text-center">
            <SmallStat label="通过" value={`${result.passed_tests}/${result.total_tests}`} />
            <SmallStat label="耗时" value={`${result.runtime_ms.toFixed(2)} ms`} />
            <SmallStat label="状态" value={result.status} />
          </div>
          {first && (
            <div className="space-y-2">
              <ResultField label="输入" value={first.input} />
              <ResultField label="预期输出" value={first.expected_output} />
              <ResultField label="实际输出" value={first.actual_output} />
              {first.stdout && <ResultField label="标准输出" value={first.stdout} />}
              {first.stderr && <ResultField label="标准错误" value={first.stderr} />}
              {first.error_message && <ResultField label="错误信息" value={`${first.error_type ?? ""}: ${first.error_message}`} />}
              {first.traceback && <ResultField label="错误堆栈" value={first.traceback} />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SmallStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-zinc-50 p-2 dark:bg-zinc-950">
      <div className="text-zinc-500">{label}</div>
      <div className="mt-1 font-semibold">{value}</div>
    </div>
  );
}

function ResultField({ label, value }: { label: string; value: unknown }) {
  const text = typeof value === "string" ? value : formatValue(value);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-zinc-500">
        <span>{label}</span>
        <button
          type="button"
          onClick={() => navigator.clipboard.writeText(text)}
          className="inline-flex items-center gap-1 rounded px-2 py-1 hover:bg-zinc-100 dark:hover:bg-zinc-800"
        >
          <Copy size={13} />
          复制
        </button>
      </div>
      <pre className="max-h-32 overflow-auto rounded-md bg-zinc-50 p-2 font-mono text-[12px] leading-5 dark:bg-zinc-950">{text}</pre>
    </div>
  );
}
