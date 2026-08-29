"use client";
import { useEffect, useRef, useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Bot,
  CalendarDays,
  Check,
  Copy,
  FileText,
  Hash,
  MessageSquare,
  Paperclip,
  Plus,
  SendHorizonal,
  Sparkles,
  Square,
  Trash2,
  UtensilsCrossed,
} from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkBreaks from "remark-breaks";

type Source = { id: number; filename: string; snippet: string };
type Msg = { role: string; content: string; sources?: Source[] };
// 知识库文件（后端 /api/files 返回：文件名 + 切片数，后端才是真相之源）
type KbFile = { filename: string; chunks: number };
// 一次会话：id 唯一标识，title 用首条提问生成，消息和创建时间一起存
type Conversation = { id: string; title: string; messages: Msg[]; createdAt: number };

// localStorage 键名（第 3 期：会话持久化，刷新页面不丢）
const LS_CONVS = "ai-workspace:conversations";
const LS_ACTIVE = "ai-workspace:active";

// 欢迎页推荐问题：点击直接发送，面试官零门槛体验核心功能
const SUGGESTIONS: { icon: LucideIcon; text: string }[] = [
  { icon: CalendarDays, text: "年假几天？" },
  { icon: Hash, text: "公司代号是什么？" },
  { icon: Bot, text: "介绍下你自己" },
  { icon: UtensilsCrossed, text: "加班餐补怎么算？" },
];

// AI 回答的 Markdown 排版（第 2 期：富文本渲染；用户消息仍用纯文本，不渲染）
const mdComponents: Components = {
  p: ({ children }) => <p className="mb-2.5 leading-7 last:mb-0">{children}</p>,
  h1: ({ children }) => <h1 className="mt-4 mb-2 text-lg font-semibold first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mt-4 mb-2 text-base font-semibold first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mt-3 mb-1.5 text-[15px] font-semibold first:mt-0">{children}</h3>,
  ul: ({ children }) => <ul className="mb-2.5 pl-5 space-y-1 list-disc">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2.5 pl-5 space-y-1 list-decimal">{children}</ol>,
  li: ({ children }) => <li className="leading-7">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-gray-900">{children}</strong>,
  blockquote: ({ children }) => (
    <blockquote className="my-2.5 border-l-2 border-[#4d6bfe] pl-3 text-gray-500">{children}</blockquote>
  ),
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-[#4d6bfe] underline underline-offset-2">
      {children}
    </a>
  ),
  // 代码块：深色底卡片，等宽字体，横向溢出可滚动
  pre: ({ children }) => (
    <pre className="my-2.5 p-3 rounded-lg bg-[#1f2430] text-gray-100 text-[13px] leading-6 overflow-x-auto">{children}</pre>
  ),
  // 行内代码：无语言标记且无换行的 code 视为行内，灰底蓝字；否则交给 pre 处理
  code: ({ className, children }) =>
    !className && !String(children).includes("\n") ? (
      <code className="px-1.5 py-0.5 rounded bg-gray-100 text-[#4d6bfe] text-[13px]">{children}</code>
    ) : (
      <code className={className}>{children}</code>
    ),
};

export default function Home() {
  const [input, setInput] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null); // null = 未开始的新对话（欢迎页）
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState<KbFile[]>([]); // 知识库文件清单（来自后端，非本地记录）
  const [uploading, setUploading] = useState(false);
  const [hydrated, setHydrated] = useState(false); // 从 localStorage 读取完成后才允许回写，防止空数据覆盖
  const [copiedKey, setCopiedKey] = useState<string | null>(null); // 刚复制成功的 AI 消息（显示“已复制”反馈用）
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true); // 是否贴底：用户往上滚去看旧内容时置为 false，输出就不劫持滚动
  const abortRef = useRef<AbortController | null>(null); // 停止生成用：中断 fetch 流
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // ===== 第 3 期：从 localStorage 恢复会话（只在首次挂载执行一次）=====
  useEffect(() => {
    try {
      const convs = localStorage.getItem(LS_CONVS);
      if (convs) setConversations(JSON.parse(convs));
      const active = localStorage.getItem(LS_ACTIVE);
      if (active) setActiveId(active === "null" ? null : active);
    } catch {
      // 数据损坏就当没有，从空白开始（不阻断页面）
    }
    setHydrated(true);
  }, []);

  // ===== 第 3 期：状态变化即写回 localStorage（hydrated 前不回写）=====
  useEffect(() => {
    if (hydrated) localStorage.setItem(LS_CONVS, JSON.stringify(conversations));
  }, [conversations, hydrated]);
  useEffect(() => {
    if (hydrated) localStorage.setItem(LS_ACTIVE, String(activeId));
  }, [activeId, hydrated]);

  // 当前激活会话（派生值，不额外存状态，避免数据不一致）
  const active = conversations.find(c => c.id === activeId) ?? null;
  const messages = active?.messages ?? [];

  // 新消息到达时自动滚动到底部（仅当用户贴底时；往上翻看时不劫持滚动，第 4 期）
  useEffect(() => {
    if (stickRef.current) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 切换会话时重置为贴底（从新对话的最新内容开始看）
  useEffect(() => {
    stickRef.current = true;
  }, [activeId]);

  // 滚动监听：距底部 80px 以内算“贴底”（第 4 期）
  const handleScroll = () => {
    const el = scrollRef.current;
    if (el) stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  // 更新指定会话的最后一条消息（流式渲染用：token 逐段追加到最后一条 AI 消息）
  const patchLastMsg = (convId: string, patch: (m: Msg) => Msg) => {
    setConversations(prev =>
      prev.map(c => {
        if (c.id !== convId || c.messages.length === 0) return c;
        const msgs = [...c.messages];
        msgs[msgs.length - 1] = patch(msgs[msgs.length - 1]);
        return { ...c, messages: msgs };
      })
    );
  };

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

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
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
      let buffer = ""; // SSE 缓冲区：网络分包和消息边界不对齐，必须攒够一条完整消息再解析

      // 先占位一条空 AI 消息（流式内容逐段填进去）
      setConversations(prev =>
        prev.map(c => (c.id === convId ? { ...c, messages: [...c.messages, { role: "assistant", content: "" }] } : c))
      );

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

          if (name === "sources") {
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
        // 已有 AI 占位消息就追加提示行（不丢已生成内容），还没建占位（fetch 就失败）就新建一条错误消息，
        // 绝不能直接 patch 最后一条——那可能是用户消息，会把错误文案糊到用户头上
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

  // 第 4 期：复制 AI 回答（成功后显示“已复制”反馈 2 秒）
  const copyMessage = async (text: string, key: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 2000);
    } catch {
      // 剪贴板不可用（如非安全上下文）时静默失败，不打断用户
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // 大小预检：超 5MB 当场提示，不等网络往返（后端还有同样规则的最终裁决）
    if (file.size > 5 * 1024 * 1024) {
      alert("文件太大（超过 5MB），请压缩或拆分后再上传");
      e.target.value = "";
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file); // "file" 这个名字要和后端 UploadFile 的参数名一致

    try {
      const res = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData, // ⚠️ 注意：发 FormData 千万不要手动设置 Content-Type！
      });

      if (!res.ok) {
        // 后端返回的错误信息，比如“不支持的文件类型”；响应体异常时兜底默认文案，绝不静默
        let detail = "上传失败，请稍后再试";
        try {
          const err = await res.json();
          if (err?.detail) detail = err.detail;
        } catch {
          // 响应体不是 JSON（如裸 500 文本）就用默认提示
        }
        alert(detail);
      } else {
        await loadFiles(); // 上传成功后重新拉后端清单（含新文件的切片数）
      }
    } catch {
      alert("上传失败：无法连接后端服务，请确认后端已启动");
    } finally {
      setUploading(false);
      e.target.value = ""; // 清空 input，允许再次选择同一个文件（无论成败都要清，否则选同一文件无反应）
    }
  };

  // ===== 知识库文件管理（后端为真相之源：清单从 Chroma 元数据聚合）=====
  const loadFiles = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/files");
      if (res.ok) setFiles((await res.json()).files);
    } catch {
      // 后端没启动时清单保持为空，不阻断页面
    }
  };

  useEffect(() => {
    loadFiles(); // 首次挂载拉一次真实清单（替代以前的本地记录）
  }, []);

  const deleteFile = async (filename: string) => {
    if (!window.confirm(`确定把「${filename}」从知识库删除吗？相关切片会一并删除。`)) return;
    try {
      const res = await fetch(`http://localhost:8000/api/files/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail);
        return;
      }
      await loadFiles(); // 删除后刷新清单
    } catch {
      alert("删除失败：无法连接后端服务");
    }
  };

  // 新对话：只是切回"未选中"状态，历史会话都还在列表里
  const newConversation = () => {
    setActiveId(null);
    setInput("");
    textareaRef.current?.focus();
  };

  // 删除会话（阻止冒泡，避免触发切换；二次确认防误触）
  const deleteConversation = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm("确定删除这条对话吗？删除后无法恢复。")) return;
    setConversations(prev => prev.filter(c => c.id !== id));
    if (activeId === id) setActiveId(null);
  };

  // 输入框自适应高度（最多 160px，再多就滚动）
  const resizeTextarea = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  };

  return (
    <div className="flex h-screen bg-white text-gray-800">
      {/* ===== 侧栏 ===== */}
      <aside className="w-[240px] shrink-0 bg-[#f7f8fa] border-r border-gray-200/70 flex flex-col">
        <div className="p-3 space-y-3">
          {/* logo */}
          <div className="flex items-center gap-2 px-2 py-1.5">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-[#4d6bfe] to-[#7c93ff] flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold text-[15px]">AI Workspace</span>
          </div>

          {/* 新对话 */}
          <button
            onClick={newConversation}
            disabled={loading}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-[#4d6bfe] text-white text-sm font-medium hover:bg-[#3d5bf0] transition-colors disabled:opacity-40"
          >
            <Plus className="w-4 h-4" />
            新对话
          </button>

          {/* 知识库 */}
          <div>
            <div className="px-2 pb-1.5 text-xs text-gray-400">知识库</div>
            <label
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border border-dashed text-sm transition-colors ${
                uploading
                  ? "border-gray-200 text-gray-400 cursor-wait"
                  : "border-gray-300 text-gray-500 cursor-pointer hover:border-[#4d6bfe] hover:text-[#4d6bfe]"
              }`}
            >
              <Paperclip className="w-4 h-4" />
              {uploading ? "上传中..." : "上传文档"}
              <input
                type="file"
                accept=".txt,.md,.pdf"
                onChange={handleUpload}
                disabled={uploading}
                className="hidden"
              />
            </label>

            <div className="mt-2 space-y-1">
              {files.length === 0 ? (
                <div className="px-2 text-xs text-gray-300">暂无文档</div>
              ) : (
                files.map(f => (
                  <div
                    key={f.filename}
                    className="group/file flex items-center gap-2 px-2 py-1.5 rounded-md text-[13px] text-gray-600 bg-white border border-gray-200/60"
                  >
                    <FileText className="w-3.5 h-3.5 text-[#4d6bfe] shrink-0" />
                    <span className="flex-1 truncate" title={f.filename}>
                      {f.filename}
                    </span>
                    <span className="text-[10px] text-gray-300 shrink-0">{f.chunks}片</span>
                    <button
                      onClick={() => deleteFile(f.filename)}
                      title="从知识库删除该文档"
                      className="opacity-0 group-hover/file:opacity-100 p-0.5 rounded text-gray-300 hover:text-red-500 transition-all shrink-0"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* ===== 第 3 期：会话列表（按时间倒序，新会话在上）===== */}
        <div className="flex-1 overflow-y-auto px-3 pb-2">
          <div className="px-2 pb-1.5 text-xs text-gray-400">对话记录</div>
          {conversations.length === 0 ? (
            <div className="px-2 text-xs text-gray-300">暂无对话</div>
          ) : (
            <div className="space-y-0.5">
              {conversations.map(c => (
                <div
                  key={c.id}
                  onClick={() => setActiveId(c.id)}
                  className={`group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer text-[13px] transition-colors ${
                    c.id === activeId
                      ? "bg-white text-gray-800 border border-gray-200/70 shadow-sm"
                      : "text-gray-500 hover:bg-white/60 border border-transparent"
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5 shrink-0 text-gray-400" />
                  <span className="flex-1 truncate" title={c.title}>
                    {c.title}
                  </span>
                  <button
                    onClick={e => deleteConversation(c.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-gray-300 hover:text-red-500 transition-all"
                    title="删除该对话"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="p-4 text-[11px] text-gray-300">
          RAG · FastAPI · Chroma · DeepSeek
        </div>
      </aside>

      {/* ===== 主区域 ===== */}
      <main className="flex-1 flex flex-col min-w-0">
        {messages.length === 0 ? (
          /* --- 欢迎页 --- */
          <div className="flex-1 flex flex-col items-center justify-center px-6 pb-20">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#4d6bfe] to-[#7c93ff] flex items-center justify-center shadow-lg shadow-blue-100">
              <Sparkles className="w-7 h-7 text-white" />
            </div>
            <h1 className="mt-5 text-[22px] font-medium">嗨，我是知识库助手</h1>
            <p className="mt-2 text-sm text-gray-400">
              在左侧上传企业文档后向我提问，回答将标注来源
            </p>
            <div className="grid grid-cols-2 gap-3 mt-9 w-full max-w-[560px]">
              {SUGGESTIONS.map(({ icon: Icon, text }) => (
                <button
                  key={text}
                  onClick={() => sendMessage(text)}
                  className="flex items-center gap-2.5 px-4 py-3.5 rounded-xl border border-gray-200 text-sm text-gray-600 text-left hover:border-[#4d6bfe] hover:text-[#4d6bfe] hover:bg-[#f5f7ff] transition-colors"
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {text}
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* --- 对话流 --- */
          <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto">
            <div className="max-w-[768px] mx-auto px-4 py-8 space-y-7">
              {messages.map((msg, i) =>
                msg.role === "user" ? (
                  <div key={i} className="flex justify-end">
                    <div className="max-w-[80%] px-4 py-2.5 rounded-2xl bg-[#e8f1ff] text-sm whitespace-pre-wrap">
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  <div key={i} className="group/msg flex gap-3">
                    {/* AI 头像 */}
                    <div className="w-7 h-7 shrink-0 rounded-full bg-gradient-to-br from-[#4d6bfe] to-[#7c93ff] flex items-center justify-center mt-0.5">
                      <Sparkles className="w-3.5 h-3.5 text-white" />
                    </div>
                    <div className="flex-1 min-w-0">
                      {msg.content === "" && loading ? (
                        /* 正在思考：三个跳动的点 */
                        <div className="flex gap-1.5 py-2">
                          {[0, 1, 2].map(n => (
                            <span
                              key={n}
                              className="w-1.5 h-1.5 rounded-full bg-[#4d6bfe] animate-bounce"
                              style={{ animationDelay: `${n * 0.15}s` }}
                            />
                          ))}
                        </div>
                      ) : (
                        /* Markdown 渲染：加粗/列表/代码块等语法正常显示（第 2 期）；
                           remarkBreaks：模型输出的单个换行也当换行处理，不粘连成一段 */
                        <div className="text-sm text-gray-800">
                          <ReactMarkdown remarkPlugins={[remarkBreaks]} components={mdComponents}>
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      )}

                      {/* 引用卡片 */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-3 space-y-1.5 max-w-xl">
                          {msg.sources.map(s => (
                            <details
                              key={s.id}
                              className="group rounded-lg border border-gray-100 bg-gray-50/70 px-3 py-2 text-xs"
                            >
                              <summary className="cursor-pointer select-none text-gray-500 flex items-center gap-1.5 hover:text-[#4d6bfe] transition-colors">
                                <FileText className="w-3.5 h-3.5 shrink-0" />
                                <span className="font-medium text-[#4d6bfe]">[{s.id}]</span>
                                <span className="truncate">{s.filename}</span>
                              </summary>
                              <div className="mt-2 pl-5 text-gray-500 leading-5 whitespace-pre-wrap">
                                {s.snippet}
                              </div>
                            </details>
                          ))}
                        </div>
                      )}

                      {/* 第 4 期：复制按钮（悬停消息浮现；正在流式输出的那条不显示）*/}
                      {msg.content && !(loading && i === messages.length - 1) && (
                        <button
                          onClick={() => copyMessage(msg.content, `${activeId}-${i}`)}
                          title="复制回答"
                          className="mt-1.5 flex items-center gap-1 text-[11px] text-gray-300 opacity-0 group-hover/msg:opacity-100 hover:text-[#4d6bfe] transition-all"
                        >
                          {copiedKey === `${activeId}-${i}` ? (
                            <>
                              <Check className="w-3.5 h-3.5" /> 已复制
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5" /> 复制
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                )
              )}
              <div ref={bottomRef} />
            </div>
          </div>
        )}

        {/* --- 底部输入框 --- */}
        <div className="px-4 pb-5 pt-1">
          <div className="max-w-[768px] mx-auto rounded-2xl border border-gray-200 bg-white shadow-sm focus-within:border-[#4d6bfe] transition-colors">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={e => {
                setInput(e.target.value);
                resizeTextarea();
              }}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="输入消息，Enter 发送，Shift + Enter 换行"
              className="w-full resize-none px-4 pt-3 pb-1 bg-transparent outline-none text-sm max-h-40"
            />
            <div className="flex items-center justify-between px-3 pb-2">
              <span className="text-[11px] text-gray-300 pl-1">
                基于知识库回答 · 引用可查证
              </span>
              {/* 第 4 期：生成中发送按钮变停止按钮（方块图标）*/}
              {loading ? (
                <button
                  onClick={stopGeneration}
                  title="停止生成"
                  className="w-8 h-8 rounded-full bg-[#4d6bfe] text-white flex items-center justify-center hover:bg-[#3d5bf0] transition-colors"
                >
                  <Square className="w-3 h-3 fill-current" />
                </button>
              ) : (
                <button
                  onClick={() => sendMessage()}
                  disabled={!input.trim()}
                  title="发送"
                  className="w-8 h-8 rounded-full bg-[#4d6bfe] text-white flex items-center justify-center hover:bg-[#3d5bf0] disabled:bg-gray-200 disabled:text-gray-400 transition-colors"
                >
                  <SendHorizonal className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
