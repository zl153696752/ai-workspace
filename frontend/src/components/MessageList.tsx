"use client";
// ===== 对话流组件（12.15 模块化拆分：从 page.tsx 抽出）=====
// 消息气泡、Markdown 渲染、引用卡片、复制按钮都在这；贴底滚动逻辑也归它管
//（滚动是这个组件自己的事：监听、状态、refs 全在内部，页面不用关心）
import { useEffect, useRef, useState } from "react";
import { Check, Copy, FileText, Sparkles } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkBreaks from "remark-breaks";
import type { Msg } from "@/types";

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

type MessageListProps = {
  messages: Msg[];
  loading: boolean;
  activeId: string | null; // 复制反馈的 key 前缀（切会话后旧反馈自动失效）
};

export default function MessageList({ messages, loading, activeId }: MessageListProps) {
  const [copiedKey, setCopiedKey] = useState<string | null>(null); // 刚复制成功的 AI 消息（显示“已复制”反馈用）
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true); // 是否贴底：用户往上滚去看旧内容时置为 false，输出就不劫持滚动

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

  return (
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
  );
}
