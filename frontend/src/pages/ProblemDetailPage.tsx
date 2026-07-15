import Editor from "@monaco-editor/react";
import type { editor as MonacoEditor } from "monaco-editor";
import { Copy, Loader2, Play, RotateCcw, Send, Settings } from "lucide-react";
import { BlockMath } from "react-katex";
import { useCallback, useEffect, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import type { JudgeResponse, ProblemDetail } from "../types";
import { difficultyClass, formatValue, parseUserCodeLine, statusClass } from "../utils";

type MonacoApi = typeof import("monaco-editor");
type EditorTheme = "light" | "vs-dark";

function currentEditorTheme(): EditorTheme {
  return document.documentElement.classList.contains("dark") ? "vs-dark" : "light";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStructuredTest(value: unknown): value is Record<string, unknown> {
  return isRecord(value) && ("args" in value || "input" in value || "expected" in value || "kwargs" in value);
}

function normalizeStructuredTest(test: Record<string, unknown>): Record<string, unknown> {
  if ("args" in test) {
    const args = Array.isArray(test.args) ? test.args : [test.args];
    return { ...test, args, kwargs: isRecord(test.kwargs) ? test.kwargs : {} };
  }
  if ("input" in test) {
    return {
      args: [test.input],
      kwargs: isRecord(test.kwargs) ? test.kwargs : {},
      ...("expected" in test ? { expected: test.expected } : {}),
    };
  }
  return { args: [test], kwargs: {} };
}

function splitTopLevelParams(params: string): string[] {
  const result: string[] = [];
  let current = "";
  let depth = 0;
  for (const char of params) {
    if (char === "[" || char === "(" || char === "{") depth += 1;
    if (char === "]" || char === ")" || char === "}") depth -= 1;
    if (char === "," && depth === 0) {
      if (current.trim()) result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  if (current.trim()) result.push(current.trim());
  return result;
}

function getSignatureParams(signature: string): string[] {
  const match = signature.match(/\((.*)\)\s*->/);
  if (!match) return [];
  return splitTopLevelParams(match[1]).filter((item) => item && item !== "self");
}

function inferredListDepth(param: string): number {
  const annotation = param.split(":").slice(1).join(":");
  return (annotation.match(/list\s*\[/g) ?? []).length;
}

function arrayDepth(value: unknown): number {
  if (!Array.isArray(value)) return 0;
  if (value.length === 0) return 1;
  return 1 + Math.max(...value.map(arrayDepth));
}

function normalizeDirectInput(parsed: unknown, problem: ProblemDetail): Record<string, unknown> {
  const params = getSignatureParams(problem.function_signature);
  if (params.length > 1 && Array.isArray(parsed)) {
    return { args: parsed, kwargs: {} };
  }

  if (params.length === 1 && Array.isArray(parsed) && parsed.length === 1 && Array.isArray(parsed[0])) {
    const expectedDepth = inferredListDepth(params[0]);
    if (expectedDepth > 0 && arrayDepth(parsed) === expectedDepth + 1 && arrayDepth(parsed[0]) === expectedDepth) {
      return { args: [parsed[0]], kwargs: {} };
    }
  }

  return { args: [parsed], kwargs: {} };
}

function normalizeCustomTestInput(parsed: unknown, problem: ProblemDetail): Array<Record<string, unknown>> {
  if (Array.isArray(parsed)) {
    if (parsed.length === 0) return [];
    if (parsed.every(isStructuredTest)) {
      return parsed.map(normalizeStructuredTest);
    }
    return [normalizeDirectInput(parsed, problem)];
  }
  if (isStructuredTest(parsed)) {
    return [normalizeStructuredTest(parsed)];
  }
  return [normalizeDirectInput(parsed, problem)];
}

export default function ProblemDetailPage() {
  const { slug = "" } = useParams();
  const [problem, setProblem] = useState<ProblemDetail | null>(null);
  const [code, setCode] = useState("");
  const [customTests, setCustomTests] = useState("");
  const [fontSize, setFontSize] = useState(() => Number(localStorage.getItem("dlcode-font-size") ?? 15));
  const [leftWidth, setLeftWidth] = useState(() => Number(localStorage.getItem("dlcode-left-width") ?? 46));
  const [editorTheme, setEditorTheme] = useState<EditorTheme>(() => currentEditorTheme());
  const [runResult, setRunResult] = useState<JudgeResponse | null>(null);
  const [submitResult, setSubmitResult] = useState<JudgeResponse | null>(null);
  const [loadingRun, setLoadingRun] = useState(false);
  const [loadingSubmit, setLoadingSubmit] = useState(false);
  const [error, setError] = useState("");
  const [draftLoaded, setDraftLoaded] = useState(false);
  const editorRef = useRef<MonacoEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<MonacoApi | null>(null);
  const decorationsRef = useRef<MonacoEditor.IEditorDecorationsCollection | null>(null);
  const resultAreaRef = useRef<HTMLDivElement | null>(null);
  const runCodeRef = useRef<() => Promise<void>>(async () => undefined);
  const submitCodeRef = useRef<() => Promise<void>>(async () => undefined);

  useEffect(() => {
    setDraftLoaded(false);
    setRunResult(null);
    setSubmitResult(null);
    setError("");
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
    const syncTheme = () => setEditorTheme(currentEditorTheme());
    syncTheme();
    const observer = new MutationObserver(syncTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    monacoRef.current?.editor.setTheme(editorTheme);
  }, [editorTheme]);

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
    const text = customTests.trim();
    if (!text) return undefined;
    try {
      if (!problem) return undefined;
      return normalizeCustomTestInput(JSON.parse(text), problem);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      throw new Error(`自定义测试用例不是合法 JSON：${detail}`);
    }
  }, [customTests, problem]);

  const runCode = useCallback(async () => {
    if (!problem || loadingRun || loadingSubmit) return;
    try {
      setLoadingRun(true);
      setError("");
      const result = await api.run(problem.id, code, parseCustomTests());
      setRunResult(result);
      highlightError(result.first_error?.traceback);
      window.setTimeout(() => resultAreaRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }), 0);
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
      window.setTimeout(() => resultAreaRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }), 0);
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

  function startDrag(event: MouseEvent<HTMLDivElement>) {
    event.preventDefault();
    const move = (moveEvent: globalThis.MouseEvent) => {
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
    monaco.editor.setTheme(editorTheme);
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
    <main className="overflow-hidden" style={{ height: "calc(100dvh - 57px)" }}>
      <div className="grid h-full min-h-0 overflow-hidden" style={{ gridTemplateColumns: `${leftWidth}% 8px minmax(0, 1fr)` }}>
        <section className="scrollbar-thin min-h-0 overflow-y-auto overscroll-contain border-r border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
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
            {problem.completed && (
              <span className="rounded bg-emerald-50 px-2 py-1 text-xs text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
                已通过
              </span>
            )}
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
            <p>请按照函数签名接收参数。数值计算题默认提供 PyTorch 模板；标注为 Tensor 的输入会由判题器转换为 PyTorch 对象。</p>
          </InfoBlock>
          <InfoBlock title="返回值说明">
            <p>返回值会按结构递归比较，浮点数和 PyTorch Tensor 支持误差比较；兼容返回 NumPy 数组。</p>
          </InfoBlock>
          <InfoBlock title="示例">
            <div className="space-y-3">
              {problem.examples.map((example, index) => (
                <ExampleCard key={index} example={example} index={index} />
              ))}
            </div>
          </InfoBlock>
          <AlgorithmBlock problem={problem} />
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
            <p>
              {problem.categories.join("、")}；{problem.source_note}
            </p>
          </InfoBlock>
        </section>

        <div
          role="separator"
          onMouseDown={startDrag}
          className="cursor-col-resize bg-zinc-200 hover:bg-moss dark:bg-zinc-800"
          title="拖动调整左右宽度"
        />

        <section className="grid min-h-0 min-w-0 grid-rows-[auto_minmax(220px,1fr)_minmax(230px,38%)] bg-zinc-50 dark:bg-zinc-950">
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
              <ActionButton icon={Play} label="运行代码" loadingText="运行中..." loading={loadingRun} disabled={loadingRun || loadingSubmit} onClick={runCode} />
              <ActionButton
                icon={Send}
                label="提交答案"
                loadingText="提交中..."
                loading={loadingSubmit}
                disabled={loadingRun || loadingSubmit}
                onClick={submitCode}
                primary
              />
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

          <div className="min-h-0 overflow-hidden">
            <Editor
              height="100%"
              language="python"
              theme={editorTheme}
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

          <div
            ref={resultAreaRef}
            className="scrollbar-thin min-h-0 overflow-y-auto overscroll-contain border-t border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900"
          >
            <div className="grid min-h-0 gap-4 xl:grid-cols-[0.85fr_1.15fr]">
              <div>
                <label className="mb-2 block text-sm font-semibold">自定义测试用例</label>
                <textarea
                  value={customTests}
                  onChange={(event) => setCustomTests(event.target.value)}
                  placeholder={
                    '可选。单参数题直接填输入，例如矩阵转置：\n[[1, 2], [3, 4], [5, 6]]\n\n多参数题直接填参数数组，例如：\n[[3, 1, 5], 2]\n\n也支持高级格式：\n[{"input": [[1, 2], [3, 4]], "expected": [[1, 3], [2, 4]]}]'
                  }
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

function ExampleCard({ example, index }: { example: Record<string, unknown>; index: number }) {
  const input = String(example.input ?? "");
  const output = String(example.output ?? "");
  const explanation = String(example.explanation ?? "");
  return (
    <div className="rounded-md bg-zinc-100 p-3 text-xs dark:bg-zinc-950">
      <div className="mb-2 flex items-center justify-between gap-2 text-zinc-500">
        <span>示例 {index + 1}</span>
        <button
          type="button"
          onClick={() => navigator.clipboard.writeText(input)}
          className="inline-flex items-center gap-1 rounded px-2 py-1 hover:bg-white dark:hover:bg-zinc-900"
        >
          <Copy size={13} />
          复制输入
        </button>
      </div>
      <ExampleLine label="输入" value={input} />
      <ExampleLine label="输出" value={output} />
      {explanation && <div className="mt-2 text-zinc-500">说明：{explanation}</div>}
    </div>
  );
}

function ExampleLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid gap-1 py-1 sm:grid-cols-[44px_minmax(0,1fr)]">
      <span className="font-semibold text-zinc-600 dark:text-zinc-300">{label}：</span>
      <pre className="scrollbar-thin overflow-x-auto whitespace-pre rounded bg-white px-2 py-1 font-mono leading-5 dark:bg-zinc-900">{value}</pre>
    </div>
  );
}

function AlgorithmBlock({ problem }: { problem: ProblemDetail }) {
  const { formulas, symbols, steps } = problem.presentation;
  return (
    <InfoBlock title={formulas.length > 0 ? "公式与算法" : "算法思路"}>
      <div className="overflow-hidden rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
        {formulas.length > 0 && (
          <div className="divide-y divide-zinc-100 bg-gradient-to-br from-emerald-50/60 to-white dark:divide-zinc-800 dark:from-emerald-950/20 dark:to-zinc-950">
            {formulas.map((item, index) => (
              <div key={`${item.latex}-${index}`} className="px-4 py-3">
                {item.label && <div className="mb-1 text-center text-xs font-medium text-moss dark:text-emerald-300">{item.label}</div>}
                <div className="scrollbar-thin overflow-x-auto overflow-y-hidden py-2 text-center text-lg sm:text-xl">
                  <BlockMath math={item.latex} />
                </div>
              </div>
            ))}
          </div>
        )}
        {symbols.length > 0 && (
          <div className="border-t border-zinc-100 bg-zinc-50/70 px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900/60">
            <div className="mb-1 text-xs font-semibold text-zinc-500">符号与约定</div>
            <ul className="list-disc space-y-1 pl-5 text-xs leading-6 text-zinc-600 dark:text-zinc-300">
              {symbols.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        )}
        <ol className="list-decimal space-y-1 px-4 py-3 pl-9 text-sm leading-7 text-zinc-600 dark:text-zinc-300">
          {steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </div>
    </InfoBlock>
  );
}

function ActionButton({
  icon: Icon,
  label,
  loadingText,
  loading,
  disabled,
  onClick,
  primary = false,
}: {
  icon: typeof Play;
  label: string;
  loadingText: string;
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
      {loading ? loadingText : label}
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
  const text = value === null || value === undefined ? "未提供" : typeof value === "string" ? value : formatResultValue(value);
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

function formatResultValue(value: unknown): string {
  if (isRecord(value) && Array.isArray(value.args) && isRecord(value.kwargs)) {
    const kwargs = Object.keys(value.kwargs);
    if (value.args.length === 1 && kwargs.length === 0) {
      return formatValue(value.args[0], true);
    }
    const parts = [`args=${formatValue(value.args, true)}`];
    if (kwargs.length > 0) {
      parts.push(`kwargs=${formatValue(value.kwargs, true)}`);
    }
    return parts.join(" ");
  }
  return formatValue(value, true);
}
