export type Difficulty = "简单" | "中等" | "困难";

export interface ProblemListItem {
  id: number;
  slug: string;
  title: string;
  difficulty: Difficulty;
  categories: string[];
  company_tags: string[];
  source_note: string;
  acceptance_rate: number;
  completed: boolean;
  submit_count: number;
  pass_count: number;
}

export interface ProblemsResponse {
  items: ProblemListItem[];
  total: number;
  categories: string[];
  companies: string[];
}

export interface ProblemDetail extends ProblemListItem {
  description: string;
  function_name: string;
  function_signature: string;
  starter_code: string;
  explanation: string;
  constraints: string[];
  examples: Array<Record<string, unknown>>;
  public_tests: Array<Record<string, unknown>>;
  time_limit: number;
  memory_limit: number;
}

export interface DraftResponse {
  problem_id: number;
  code: string;
  updated_at: string | null;
}

export interface TestCaseResult {
  passed: boolean;
  input: unknown;
  expected_output: unknown;
  actual_output: unknown;
  stdout: string;
  stderr: string;
  runtime_ms: number;
  error_type: string | null;
  error_message: string | null;
  traceback: string | null;
}

export interface JudgeResponse {
  status: string;
  status_code: string;
  passed_tests: number;
  total_tests: number;
  runtime_ms: number;
  results: TestCaseResult[];
  first_error: TestCaseResult | null;
}

export interface SubmissionItem {
  id: number;
  problem_id: number;
  problem_slug: string;
  problem_title: string;
  status: string;
  language: string;
  runtime_ms: number;
  passed_tests: number;
  total_tests: number;
  created_at: string;
}

export interface SubmissionDetail extends SubmissionItem {
  code: string;
  error_sample: TestCaseResult | null;
  result_detail: JudgeResponse | null;
}

export interface StatisticsResponse {
  total_problems: number;
  completed_problems: number;
  by_difficulty: Record<Difficulty, { total: number; completed: number }>;
  recent_submissions: SubmissionItem[];
  categories: string[];
}
