// ===== 共享类型（12.15 模块化拆分：从 page.tsx 抽出，组件与 hooks 共用）=====
// 前后端接口的数据形状在这里定义一次，谁用谁 import，改名改形状只动这一处

// 引用卡片：后端 SSE sources 事件的数据（id 编号 + 来源文件名 + 命中的切片原文）
export type Source = { id: number; filename: string; snippet: string };
// 一条消息：role 区分 user/assistant，AI 消息可携带引用卡片
export type Msg = { role: string; content: string; sources?: Source[] };
// 知识库文件（后端 /api/files 返回：文件名 + 切片数，后端才是真相之源）
export type KbFile = { filename: string; chunks: number };
// 一次会话：id 唯一标识，title 用首条提问生成，消息和创建时间一起存
export type Conversation = { id: string; title: string; messages: Msg[]; createdAt: number };
