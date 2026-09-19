export interface User { id: number; username: string; display_name: string }
export interface Point { id: number; name: string }
export interface Material {
  id: number; original_filename: string; status: string; knowledge_points_json: string;
  chunk_count: number; error_message: string | null;
}
export interface Question {
  id: number; material_id: number; primary_knowledge_point_id: number;
  question_type: string; question: string; options: { id: string; text: string }[];
  difficulty_level: number; question_summary: string;
}
export interface AgentTraceEvent {
  phase: 'observe' | 'decide' | 'act'; step: number; action?: string; summary: string;
}
export interface AgentResult {
  goal: string; message: string; tool_steps: number; trace: AgentTraceEvent[];
}
export interface SourceRef { chunk_id: number; page: number | null }
export interface GraphNode { id: string; label: string; category: string; description: string }
export interface GraphEdge { source: string; target: string; relation: string; evidence: string }
export interface KnowledgeGraph { nodes: GraphNode[]; edges: GraphEdge[]; source_refs: SourceRef[] }
export type AgentOutput =
  | { type: 'practice'; question: Question; reused?: boolean }
  | { type: 'qa'; answer: string; source_refs: SourceRef[]; reused?: boolean }
  | { type: 'knowledge_graph'; graph: KnowledgeGraph; reused?: boolean };
export interface Attempt {
  id: number; question_id: number; attempt_no: number; status: string;
  started_at: string; submitted_at: string | null; updated_at: string;
  is_manually_paused: number; is_page_hidden: number; active_duration_ms: number;
  progress_seq: number; score: number | null; feedback_summary: string | null;
  answer_json: string | null; submission_key: string | null; question_summary?: string;
  hint_count?: number; answer_view_count?: number; grading_confidence: string;
}
export interface Hint {
  id: number; hint_type: string; hint_level: number; status: string; hint_summary: string | null;
  request_key: string; request_text: string | null;
}
export interface ErrorRecord {
  id: number; description: string; evidence_summary: string; source: string;
  confidence: string; review_status: string;
}
export interface Detail { attempt: Attempt; question: Question | null; hints: Hint[]; errors: ErrorRecord[] }
export interface Progress {
  progress_seq: number; active_duration_ms: number; is_manually_paused: number; is_page_hidden: number;
}
export class RequestError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}
export async function call<T>(path: string, method = 'GET', data?: unknown): Promise<T> {
  const form = data instanceof FormData;
  const response = await fetch(`${import.meta.env.VITE_API_URL || ''}/api${path}`, {
    method, credentials: 'include',
    headers: { 'X-CtrlPass-Request': '1', ...(form ? {} : { 'Content-Type': 'application/json' }) },
    body: data === undefined ? undefined : form ? data : JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) {
    if (response.status === 401 && !path.startsWith('/auth/')) window.dispatchEvent(new Event('ctrlpass-session-expired'));
    throw new RequestError(result.error || '请求失败，请稍后重试。', response.status);
  }
  return result;
}
export const requestKey = () => `${Date.now().toString(36)}-${crypto.getRandomValues(new Uint32Array(3)).join('-')}`;
export const messageOf = (error: unknown) => error instanceof Error ? error.message : '请求失败，请重试。';
export const labels: Record<string, string> = {
  uploaded: '已上传', processing: '处理中', ready: '可使用', failed: '处理失败',
  in_progress: '作答中', pending_grading: '待判分', completed: '已完成', abandoned: '已放弃', expired: '题目过期', grading_failed: '判分失败',
  single_choice: '单选', multiple_choice: '多选', fill_blank: '填空', short_answer: '简答',
  concept: '概念提醒', approach: '思路引导', step: '步骤提示', answer: '查看答案',
  high: '高', medium: '中', low: '低', unknown: '未知', user: '用户自述', rule: '规则判断', model: '模型推断',
  pending: '待确认', confirmed: '已确认', rejected: '已否定', requested: '请求中', generated: '已生成', viewed: '已展示',
};
