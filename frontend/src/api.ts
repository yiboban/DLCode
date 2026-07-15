import type {
  DraftResponse,
  JudgeResponse,
  ProblemDetail,
  ProblemsResponse,
  StatisticsResponse,
  SubmissionDetail,
  SubmissionItem,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let message = "请求失败";
    try {
      const body = (await response.json()) as { message?: unknown; detail?: unknown };
      const rawMessage = body.message ?? body.detail;
      if (typeof rawMessage === "string") {
        message = rawMessage;
      } else if (rawMessage) {
        message = JSON.stringify(rawMessage, null, 2);
      }
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export interface ProblemQuery {
  search?: string;
  category?: string;
  difficulty?: string;
  company?: string;
  status?: string;
  sort_by?: string;
  order?: string;
}

function queryString(query: ProblemQuery): string {
  const params = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value) params.set(key, value);
  });
  const text = params.toString();
  return text ? `?${text}` : "";
}

export const api = {
  problems: (query: ProblemQuery = {}) => request<ProblemsResponse>(`/api/problems${queryString(query)}`),
  problem: (slug: string) => request<ProblemDetail>(`/api/problems/${slug}`),
  statistics: () => request<StatisticsResponse>("/api/statistics"),
  draft: (problemId: number) => request<DraftResponse>(`/api/drafts/${problemId}`),
  saveDraft: (problemId: number, code: string) =>
    request<DraftResponse>(`/api/drafts/${problemId}`, {
      method: "PUT",
      body: JSON.stringify({ code }),
    }),
  run: (problemId: number, code: string, custom_tests?: unknown[]) =>
    request<JudgeResponse>("/api/run", {
      method: "POST",
      body: JSON.stringify({ problem_id: problemId, code, custom_tests }),
    }),
  submit: (problemId: number, code: string) =>
    request<JudgeResponse>("/api/submit", {
      method: "POST",
      body: JSON.stringify({ problem_id: problemId, code }),
    }),
  submissions: () => request<SubmissionItem[]>("/api/submissions"),
  submission: (id: number) => request<SubmissionDetail>(`/api/submissions/${id}`),
};
