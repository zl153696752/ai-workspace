// ===== 步骤11：长期记忆治理 hook =====
// 复刻 useKnowledgeFiles 的模式：后端是真相之源，前端不做本地记录。
// 只管跟 GET/DELETE /api/memories 打交道：加载清单、删除单项。
import { useEffect, useState } from "react";
import type { Memory } from "@/types";
import { API_BASE } from "@/lib/api";
import { authHeaders } from "@/lib/auth";   // 带票：亮哥过 require_liang，游客被后端 403

export function useMemories() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(false);

  const loadMemories = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/memories`, { headers: authHeaders() });
      if (res.ok) setMemories((await res.json()).memories);
      // 非 ok（如游客 403）就保持空，不阻断面板
    } catch {
      // 后端没启动时清单保持为空
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMemories();   // 面板挂载即拉一次（每次打开都是新鲜数据）
  }, []);

  const deleteMemory = async (id: number) => {
    if (!window.confirm("确定删除这条记忆吗？删除后牛来不再参考它。")) return;
    try {
      const res = await fetch(`${API_BASE}/api/memories/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail ?? "删除失败");   // 404「记忆不存在或无权删除」从这里直达
        return;
      }
      setMemories(prev => prev.filter(m => m.id !== id));   // 乐观移除，无需重拉整个清单
    } catch {
      alert("删除失败：无法连接后端服务");
    }
  };

  return { memories, loading, deleteMemory };
}