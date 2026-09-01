"use client";
// ===== 底部输入区组件（12.15 模块化拆分：从 page.tsx 抽出）=====
// 文本框自适应高度、Enter 发送 / Shift+Enter 换行、生成中变停止按钮；
// input 状态由页面持有（发送后清空是页面的事），这里只管输入交互
import { SendHorizonal, Square } from "lucide-react";

type ChatInputProps = {
  input: string;
  setInput: (v: string) => void;
  loading: boolean;
  onSend: () => void;
  onStop: () => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>; // 页面持有：新对话时要聚焦输入框
};

export default function ChatInput({ input, setInput, loading, onSend, onStop, textareaRef }: ChatInputProps) {
  // 输入框自适应高度（最多 160px，再多就滚动）
  const resizeTextarea = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  };

  return (
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
              onSend();
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
              onClick={onStop}
              title="停止生成"
              className="w-8 h-8 rounded-full bg-[#4d6bfe] text-white flex items-center justify-center hover:bg-[#3d5bf0] transition-colors"
            >
              <Square className="w-3 h-3 fill-current" />
            </button>
          ) : (
            <button
              onClick={onSend}
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
  );
}
