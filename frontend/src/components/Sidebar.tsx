"use client";
// ===== 侧栏组件 =====
// 展示 + 回调上抛为主；仅"登录口令输入""私人上传勾选"两处是本地 UI 状态（步骤4c），不必上抛给页面。
import { useState } from "react";   // 步骤4c：登录口令、私人勾选两处本地状态
import {
  Activity,
  Brain,
  Download,
  FileText,
  Lock,
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
  isLiang: boolean;   // 步骤4c：是否亮哥——决定画不画删除按钮、私人勾选、登录框还是退出
  onNewConversation: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onUploadFile: (e: React.ChangeEvent<HTMLInputElement>, isPrivate: boolean) => void;   // 步骤4c：多带一个"是否私人"
  onDownloadFile: (filename: string) => void;
  onDeleteFile: (filename: string) => void;
  onLogin: (password: string) => void;   // 步骤4c：登录（口令上抛给 page.tsx 换票）
  onLogout: () => void;                  // 步骤4c：退出（前端删票）
  onOpenMemories: () => void;
  onOpenMetrics: () => void;
};

export default function Sidebar({
  conversations,
  activeId,
  loading,
  files,
  uploading,
  isLiang,
  onNewConversation,
  onSelectConversation,
  onDeleteConversation,
  onUploadFile,
  onDownloadFile,
  onDeleteFile,
  onLogin,
  onLogout,
  onOpenMemories,
  onOpenMetrics
}: SidebarProps) {
  // 步骤4c：两处纯本地 UI 状态——登录口令输入、上传时的"私人"勾选（都不用上抛给页面）
  const [pw, setPw] = useState("");
  const [uploadPrivate, setUploadPrivate] = useState(false);
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
              onChange={e => onUploadFile(e, uploadPrivate)}
              disabled={uploading}
              className="hidden"
            />
          </label>

          {/* 私人上传勾选：仅亮哥可见。游客传的一律公共（后端也会强制），故前端干脆不给游客这个选项 */}
          {isLiang && (
            <label className="flex items-center gap-1.5 px-3 pt-1.5 text-xs text-gray-500 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={uploadPrivate}
                onChange={e => setUploadPrivate(e.target.checked)}
                className="accent-[#4d6bfe]"
              />
              <Lock className="w-3 h-3" />
              传为私人（仅自己可见）
            </label>
          )}

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
                  {/* 私人徽章：只有亮哥的清单里会出现 private=true 的文件（游客的私人片后端已过滤） */}
                  {f.private && (
                    <span
                      className="shrink-0 flex items-center gap-0.5 text-[10px] px-1 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-200/70"
                      title="私人文档：仅亮哥可见、可下载"
                    >
                      <Lock className="w-2.5 h-2.5" /> 私
                    </span>
                  )}
                  <span className="text-[10px] text-gray-300 shrink-0">{f.chunks}片</span>
                  <button
                    onClick={() => onDownloadFile(f.filename)}
                    title="下载原文件"
                    className="opacity-0 group-hover/file:opacity-100 p-0.5 rounded text-gray-300 hover:text-[#4d6bfe] transition-all shrink-0"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                  {/* 删除按钮：仅亮哥可见（游客界面直接不画＝体验层；后端 require_liang 的 403 才是安全层，两层独立） */}
                  {isLiang && (
                    <button
                      onClick={() => onDeleteFile(f.filename)}
                      title="从知识库删除该文档"
                      className="opacity-0 group-hover/file:opacity-100 p-0.5 rounded text-gray-300 hover:text-red-500 transition-all shrink-0"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* ===== 会话列表（按时间倒序，新会话在上）===== */}
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

      {/* ===== 底部：身份区（游客显示登录框，亮哥显示"已登录 + 退出"）===== */}
      <div className="p-3 border-t border-gray-200/70 space-y-2">
        {isLiang ? (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between px-1">
              <span className="flex items-center gap-1 text-xs text-gray-500">
                <Lock className="w-3 h-3 text-[#4d6bfe]" /> 已登录 · 亮哥
              </span>
              <button
                onClick={() => {
                  onLogout();
                  setPw("");
                }}
                className="text-xs text-gray-400 hover:text-red-500 transition-colors"
              >
                退出
              </button>
            </div>
            {/* 步骤11：记忆治理入口（仅亮哥）——点开看清系统记了啥、一键删错记 */}
            <button
              onClick={onOpenMemories}
              className="w-full flex items-center gap-1.5 px-2 py-1.5 rounded-md text-xs text-gray-500 border border-transparent hover:bg-white hover:text-[#4d6bfe] hover:border-gray-200/70 transition-colors"
            >
              <Brain className="w-3.5 h-3.5" /> 我的长期记忆
            </button>
            {/* 步骤12·12B：质量看板入口（仅亮哥）——运行时成本/降级/检索命中/漏标一屏看全 */}
            <button
              onClick={onOpenMetrics}
              className="w-full flex items-center gap-1.5 px-2 py-1.5 rounded-md text-xs text-gray-500 border border-transparent hover:bg-white hover:text-[#4d6bfe] hover:border-gray-200/70 transition-colors"
            >
              <Activity className="w-3.5 h-3.5" /> 质量看板
            </button>
          </div>
        ) : (
          <div className="space-y-1.5">
            <input
              type="password"
              value={pw}
              onChange={e => setPw(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter") {
                  onLogin(pw);
                  setPw("");
                }
              }}
              placeholder="亮哥口令"
              className="w-full px-2 py-1.5 text-xs rounded-md border border-gray-300 focus:outline-none focus:border-[#4d6bfe]"
            />
            <button
              onClick={() => {
                onLogin(pw);
                setPw("");
              }}
              className="w-full px-2 py-1.5 text-xs rounded-md bg-[#4d6bfe] text-white font-medium hover:bg-[#3d5bf0] transition-colors"
            >
              登录
            </button>
          </div>
        )}
        <div className="text-[11px] text-gray-300 text-center">
          LangGraph · RAG · FastAPI · Chroma · DeepSeek
        </div>
      </div>
    </aside>
  );
}
