// ===== 共享类型（组件与 hooks 共用）=====
// 前后端接口的数据形状在这里定义一次，谁用谁 import，改名改形状只动这一处。

// 引用卡片：后端 SSE sources 事件的数据（id 编号 + 来源文件名 + 命中的切片原文）
export type Source = { id: number; filename: string; snippet: string };
// 步骤9：一次问答的轻量指标（meta 事件，全员可见）
export type Meta = { latency: number; tokens: number; cost: number; calls: number };
// 步骤9：单次模型调用的埋点 span
export type LlmSpan = {
  purpose?: string; latency?: number; prompt_tokens?: number;
  completion_tokens?: number; cost?: number; degraded?: boolean; reason?: string;
};
// 步骤9：完整链路（trace 事件，仅亮哥可见）= 汇总 + 两类 span 明细
export type Trace = {
  summary: { calls: number; total_latency: number; prompt_tokens: number;
             completion_tokens: number; total_cost: number; degraded: boolean };
  llm_spans: LlmSpan[];
  node_spans: Record<string, unknown>[];
};

// 一条消息：AI 消息可携带引用卡片 + 可观测性数据
export type Msg = { role: string; content: string; sources?: Source[]; meta?: Meta; trace?: Trace };
// 知识库文件（后端 /api/files 返回：文件名 + 切片数 + 公私标记，后端才是真相之源）
export type KbFile = { filename: string; chunks: number; private: boolean };
// 一次会话：id 唯一标识，title 用首条提问生成，消息和创建时间一起存
export type Conversation = { id: string; title: string; messages: Msg[]; createdAt: number };
