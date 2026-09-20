// ===== 步骤12·12B：质量看板数据 hook =====
// 封装 GET /api/metrics（亮哥专属），对称 useMemories 的写法：返回 { 数据, loading, error, 刷新 }。
import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";
import { authHeaders } from "@/lib/auth";
import type { Metrics } from "@/types";

export function useMetrics() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/metrics`, { headers: authHeaders() });
      if (!res.ok) {
        throw new Error(res.status === 403 ? "仅亮哥可见，请先在侧栏登录" : `加载失败（HTTP ${res.status}）`);
      }
      setMetrics(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "无法连接后端服务，请确认后端已启动");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);   // 面板一挂载就拉一次

  return { metrics, loading, error, refresh: load };
}