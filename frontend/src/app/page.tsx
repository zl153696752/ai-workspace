"use client";
// ===== 页面入口（12.15 模块化拆分后，这个文件只剩两类事）=====
// ① 业务编排：sendMessage（SSE 流式消费）、停止生成、新对话；② 组装组件树
// 会话持久化在 hooks/useConversations，知识库文件管理在 hooks/useKnowledgeFiles，
// 界面在 components/（Sidebar / Welcome / MessageList / ChatInput），共享类型在 types.ts
import { useRef, useState } from "react";
import ChatInput from "@/components/ChatInput";
import MessageList from "@/components/MessageList";
import Sidebar from "@/components/Sidebar";
import Welcome from "@/components/Welcome";
import { useConversations } from "@/hooks/useConversations";
import { useKnowledgeFiles } from "@/hooks/useKnowledgeFiles";
import type { Conversation, Msg } from "@/types";
import { API_BASE } from "@/lib/api";

export default function Home() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null); // 停止生成用：中断 fetch 流
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 会话数据（恢复/回写/删除都在 hook 里）+ 知识库文件（清单/上传/删除/下载都在 hook 里）
  const { conversations, setConversations, activeId, setActiveId, messages, patchLastMsg, deleteConversation } =
    useConversations();
  const { files, uploading, handleUpload, deleteFile, downloadFile } = useKnowledgeFiles();

  const sendMessage = async (text?: string) => {
    const content = (text ?? input).trim();
    if (!content || loading) return;

    const userMsg = { role: "user", content };
    // 第 3 期：欢迎页发出的第一句话负责"开新会话"，标题取问题前 20 字
    let convId = activeId;
    let history: Msg[];
    if (!convId) {
      convId = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
      const conv: Conversation = {
        id: convId,
        title: content.length > 20 ? content.slice(0, 20) + "…" : content,
        messages: [userMsg],
        createdAt: Date.now(),
      };
      setConversations(prev => [conv, ...prev]);
      setActiveId(convId);
      history = [userMsg];
    } else {
      const current = conversations.find(c => c.id === convId);
      history = [...(current?.messages ?? []), userMsg];
      setConversations(prev =>
        prev.map(c => (c.id === convId ? { ...c, messages: [...c.messages, userMsg] } : c))
      );
    }
    setInput("");
    setLoading(true);

    // 第 4 期：AbortController——“停止生成”就是中断这个信号，已流出的内容保留
    const controller = new AbortController();
    abortRef.current = controller;

    let aiContent = ""; // 提升到 try 外：catch 里要根据已生成量决定错误提示的写法

    // 占位气泡必须在 fetch 前出生：后端决策调用在响应返回前就跑完了（最漫长的 1~3 秒），
    // 等 fetch 结束再建气泡，等待期屏幕上一片空白，跳动点永远看不到（异常路径由 catch/finally 兜底）
    setConversations(prev =>
      prev.map(c => (c.id === convId ? { ...c, messages: [...c.messages, { role: "assistant", content: "" }] } : c))
    );

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history }), // 发包含本句话的完整历史
        signal: controller.signal,
      });

      const reader = response.body?.getReader();
      if (!response.ok || !reader) {
        // 后端非流式报错（如请求体不合法）：解析 detail 提示用户，不静默卡死
        let detail = "服务暂时不可用，请稍后再试";
        try {
          const err = await response.json();
          if (err?.detail) detail = typeof err.detail === "string" ? err.detail : "请求参数有误";
        } catch {
          // 响应体不是 JSON 就用默认提示
        }
        throw new Error(detail);
      }
      const decoder = new TextDecoder();
      let buffer = ""; // SSE 缓冲区：网络分包和消息边界不对齐，必须攒够一条完整消息再解析（占位气泡已在 fetch 前创建）

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }); // stream:true 防中文被拦腰截成乱码

        // 一条完整的 SSE 消息以空行 "\n\n" 结束；最后一段可能不完整，留在 buffer 里等下一轮
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const evt of events) {
          if (!evt.trim()) continue;
          let name = "";
          let data = "";
          for (const line of evt.split("\n")) {
            if (line.startsWith("event:")) name = line.slice(6).trim();
            else if (line.startsWith("data:")) data = line.slice(5).trim();
          }
          if (!data) continue;
          const payload = JSON.parse(data);

          if (name === "tool") {
            // 第 7 步：模型决定查知识库时，先给用户一个明确的状态提示，替代干等（正文首 token 一到就被干净替换）
            if (payload.called) {
              patchLastMsg(convId, m => ({ ...m, content: "🔍 正在检索知识库…" }));
            }
          } else if (name === "sources") {
            // 来源事件：写进当前 AI 消息的 sources 字段（卡片先于回答渲染出来）
            patchLastMsg(convId, m => ({ ...m, sources: payload }));
          } else if (name === "token") {
            aiContent += payload.content;
            patchLastMsg(convId, m => ({ ...m, content: aiContent }));
          } else if (name === "error") {
            // 后端发来的模型故障事件：已流出的内容保留，追加一行醒目提示，用户不会不明所以卡住
            aiContent += `\n\n> ⚠️ ${payload.message ?? "生成中断，请稍后再试"}`;
            patchLastMsg(convId, m => ({ ...m, content: aiContent }));
          }
          // done 事件：无需处理，流结束后下面会重置 loading
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // 用户点了停止：已流出的内容保留，静默结束，不提示错误
      } else {
        // 网络失败/后端没开/后端报错：把错误写进气泡，不留一个卡死的空消息；
        // 占位气泡此时必然已存在（fetch 前创建），错误文案直接填进空泡；“新建消息”分支仅作双保险
        const msgText = err instanceof Error && err.message ? err.message : "无法连接后端服务，请确认后端已启动";
        const appendText = aiContent ? `\n\n> ⚠️ ${msgText}` : `⚠️ ${msgText}`;
        setConversations(prev =>
          prev.map(c => {
            if (c.id !== convId) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last && last.role === "assistant") {
              msgs[msgs.length - 1] = { ...last, content: last.content ? last.content + appendText : appendText };
            } else {
              msgs.push({ role: "assistant", content: appendText });
            }
            return { ...c, messages: msgs };
          })
        );
      }
    } finally {
      abortRef.current = null;
      // 清理空占位消息（一个字都没到就被停掉时，不留空气泡；出错时上面已填了错误文案，不会被误删）
      setConversations(prev =>
        prev.map(c =>
          c.id === convId
            ? {
                ...c,
                messages: c.messages.filter(
                  (m, idx) => !(idx === c.messages.length - 1 && m.role === "assistant" && m.content === "")
                ),
              }
            : c
        )
      );
      setLoading(false);
    }
  };

  // 第 4 期：停止生成（中断请求，已流出的内容保留）
  const stopGeneration = () => {
    abortRef.current?.abort();
  };

  // 新对话：只是切回“未选中”状态，历史会话都还在列表里
  const newConversation = () => {
    setActiveId(null);
    setInput("");
    textareaRef.current?.focus();
  };

  return (
    <div className="flex h-screen bg-white text-gray-800">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        loading={loading}
        files={files}
        uploading={uploading}
        onNewConversation={newConversation}
        onSelectConversation={setActiveId}
        onDeleteConversation={deleteConversation}
        onUploadFile={handleUpload}
        onDownloadFile={downloadFile}
        onDeleteFile={deleteFile}
      />

      <main className="flex-1 flex flex-col min-w-0">
        {messages.length === 0 ? (
          /* --- 欢迎页（默认新对话：给面试官的功能广告位，三大能力一目了然） --- */
          <Welcome onSend={sendMessage} />
        ) : (
          /* --- 对话流（气泡 / Markdown / 引用卡片 / 复制 / 贴底滚动都在组件内） --- */
          <MessageList messages={messages} loading={loading} activeId={activeId} />
        )}

        {/* --- 底部输入框 --- */}
        <ChatInput
          input={input}
          setInput={setInput}
          loading={loading}
          onSend={() => sendMessage()}
          onStop={stopGeneration}
          textareaRef={textareaRef}
        />
      </main>
    </div>
  );
}
