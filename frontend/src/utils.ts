export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

function compactJson(value: unknown): string {
  return JSON.stringify(value);
}

export function formatValue(value: unknown, compact = false): string {
  if (value && typeof value === "object" && "__type__" in value) {
    const typed = value as { __type__: string; data?: unknown };
    const name = typed.__type__ === "tensor" ? "Tensor" : "ndarray";
    return `${name}(${compact ? compactJson(typed.data) : JSON.stringify(typed.data, null, 2)})`;
  }
  return compact ? compactJson(value) : JSON.stringify(value, null, 2);
}

export function statusClass(status: string): string {
  if (status === "通过") return "text-emerald-700 bg-emerald-50 dark:text-emerald-300 dark:bg-emerald-950/40";
  if (status === "答案错误") return "text-amber-700 bg-amber-50 dark:text-amber-300 dark:bg-amber-950/40";
  if (status.includes("错误") || status.includes("超出")) {
    return "text-berry bg-rose-50 dark:text-rose-200 dark:bg-rose-950/40";
  }
  return "text-zinc-700 bg-zinc-100 dark:text-zinc-200 dark:bg-zinc-800";
}

export function difficultyClass(difficulty: string): string {
  if (difficulty === "简单") return "text-emerald-700 bg-emerald-50 dark:text-emerald-300 dark:bg-emerald-950/40";
  if (difficulty === "中等") return "text-saffron bg-amber-50 dark:text-amber-300 dark:bg-amber-950/40";
  return "text-berry bg-rose-50 dark:text-rose-200 dark:bg-rose-950/40";
}

export function parseUserCodeLine(traceback?: string | null): number | null {
  if (!traceback) return null;
  const match = traceback.match(/user_code\.py", line (\d+)/);
  return match ? Number(match[1]) : null;
}
