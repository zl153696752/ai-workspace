"use client";
// ===== 侧栏组件（12.15 模块化拆分：从 page.tsx 抽出）=====
// 纯展示 + 回调上抛：数据和业务逻辑都在页面/hooks 手里，这里只负责画和转发点击
import {
  Download,
  FileText,
  MessageSquare,
  Paperclip,
  Plus,
  Sparkles,
  Trash2,
} from "lucide-react";
import type { Conversation, KbFile } from "@/types";

type SidebarProps = {
  conversations: Conversation[];
  activeId: string | null;
  loading: boolean;
  files: KbFile[];
  uploading: boolean;
  onNewConversation: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onUploadFile: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onDownloadFile: (filename: string) => void;
  onDeleteFile: (filename: string) => void;
};

export default function Sidebar({
  conversations,
  activeId,
  loading,
  files,
  uploading,
  onNewConversation,
  onSelectConversation,
  onDeleteConversation,
  onUploadFile,
  onDownloadFile,
  onDeleteFile,
}: SidebarProps) {
  return (
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
          onClick={onNewConversation}
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
              onChange={onUploadFile}
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
                    onClick={() => onDownloadFile(f.filename)}
                    title="下载原文件"
                    className="opacity-0 group-hover/file:opacity-100 p-0.5 rounded text-gray-300 hover:text-[#4d6bfe] transition-all shrink-0"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => onDeleteFile(f.filename)}
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
                onClick={() => onSelectConversation(c.id)}
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
                  onClick={e => {
                    e.stopPropagation(); // 阻止冒泡，避免触发切换会话
                    onDeleteConversation(c.id);
                  }}
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
  );
}
