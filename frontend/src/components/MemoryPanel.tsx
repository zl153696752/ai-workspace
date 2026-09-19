"use client";
// ===== 步骤11：亮哥的记忆治理面板（模态）=====
// 治理入口：看清系统记了亮哥啥（含低置信的，一条不藏）+ 一键删错记/过时。
// 数据全来自后端 GET /api/memories（真相之源），删除走 DELETE /api/memories/{id}。
import { Brain, Trash2, X } from "lucide-react";
import { useMemories } from "@/hooks/useMemories";

// type 值 → 中文标签 + 配色（未知类型兜底灰色）
const TYPE_LABEL: Record<string, { text: string; cls: string }> = {
  preference: { text: "偏好", cls: "bg-blue-50 text-blue-600 border-blue-200/70" },
  fact: { text: "事实", cls: "bg-emerald-50 text-emerald-600 border-emerald-200/70" },
};

export default function MemoryPanel({ onClose }: { onClose: () => void }) {
  const { memories, loading, deleteMemory } = useMemories();

  return (
    // 遮罩层：点空白处关闭
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={onClose}>
      {/* 面板本体：阻止冒泡，点内部不关闭 */}
      <div
        className="w-[560px] max-w-[92vw] max-h-[80vh] flex flex-col bg-white rounded-xl shadow-xl border border-gray-200"
        onClick={e => e.stopPropagation()}
      >
        {/* 头部 */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-[#4d6bfe]" />
            <span className="font-semibold text-[15px]">我的长期记忆</span>
            <span className="text-xs text-gray-400">（{memories.length} 条）</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
            title="关闭"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* 列表 */}
        <div className="flex-1 overflow-y-auto px-5 py-3 space-y-2">
          {loading ? (
            <div className="text-sm text-gray-400 text-center py-10">加载中…</div>
          ) : memories.length === 0 ? (
            <div className="text-sm text-gray-400 text-center py-10">
              暂无记忆。多和牛来聊聊你的偏好，它会慢慢记住。
            </div>
          ) : (
            memories.map(m => {
              const t = TYPE_LABEL[m.type] ?? { text: m.type, cls: "bg-gray-50 text-gray-500 border-gray-200" };
              return (
                <div
                  key={m.id}
                  className="group rounded-lg border border-gray-200/70 px-3 py-2.5 hover:border-gray-300 transition-colors"
                >
                  <div className="flex items-start gap-2">
                    <span className={`shrink-0 text-[10px] px-1.5 py-0.5 rounded border ${t.cls}`}>{t.text}</span>
                    <span className="flex-1 text-[13px] text-gray-700 leading-relaxed">{m.content}</span>
                    <button
                      onClick={() => deleteMemory(m.id)}
                      title="删除这条记忆"
                      className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-gray-300 hover:text-red-500 transition-all shrink-0"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  {/* 元信息行：主题 / 置信度 / 未注入徽章 / 更新时间 */}
                  <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-gray-400">
                    <span>主题：{m.topic}</span>
                    <span>置信度：{(m.confidence * 100).toFixed(0)}%</span>
                    {m.confidence < 0.6 && (
                      <span
                        className="px-1.5 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-200/70"
                        title="置信度低于 60%，不会注入回答（记了但不生效）"
                      >
                        未注入
                      </span>
                    )}
                    <span>更新：{m.updated_at}</span>
                  </div>

                  {/* 来源原话：可回溯核查，错记一眼看穿 */}
                  {m.source && (
                    <div className="mt-1 text-[11px] text-gray-400 italic" title="从这句原话抽取的">
                      来源：“{m.source}”
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* 底部说明 */}
        <div className="px-5 py-2.5 border-t border-gray-100 text-[11px] text-gray-400 leading-relaxed">
          回答时这些记忆作为“仅供参考”的背景注入；发现错记或过时，直接删除即可。置信度低于 60% 的不会注入。
        </div>
      </div>
    </div>
  );
}