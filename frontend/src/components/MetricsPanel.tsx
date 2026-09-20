"use client";
// ===== 步骤12·12B：在线质量看板（模态，亮哥专属）=====
// 数据来自后端 GET /api/metrics（o1 聚合 traces 表的运行时数据）。骨架复刻 MemoryPanel，风格一致。
// 🔴 纯 Tailwind 手搓：数字卡 + 进度条 + 分布条，不引图表库（o1 给的是聚合快照、非时序，够用且清爽）。
import { Activity, X, RefreshCw, AlertTriangle } from "lucide-react";
import { useMetrics } from "@/hooks/useMetrics";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const barW = (v: number | null | undefined) => `${Math.max(0, Math.min(1, v ?? 0)) * 100}%`;

// 核心数字卡
function Stat({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div className="rounded-lg border border-gray-200/70 bg-white px-3 py-2.5">
      <div className="text-[11px] text-gray-400">{label}</div>
      <div className={`mt-0.5 text-xl font-semibold ${accent ?? "text-gray-800"}`}>{value}</div>
      {sub && <div className="text-[10px] text-gray-300 mt-0.5">{sub}</div>}
    </div>
  );
}

// 带进度条的率
function RateBar({ label, value, color }: { label: string; value: number | null; color: string }) {
  return (
    <div>
      <div className="flex items-center justify-between text-[12px] mb-1">
        <span className="text-gray-500">{label}</span>
        <span className="font-medium text-gray-700">{pct(value)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-gray-100 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: barW(value) }} />
      </div>
    </div>
  );
}

// 分布条（意图/scope/身份）
function Dist({ title, data }: { title: string; data: Record<string, number> }) {
  const entries = Object.entries(data);
  const max = Math.max(1, ...entries.map(([, c]) => c));
  return (
    <div>
      <div className="text-[11px] text-gray-400 mb-1.5">{title}</div>
      <div className="space-y-1">
        {entries.length === 0 ? (
          <div className="text-[11px] text-gray-300">无数据</div>
        ) : entries.map(([k, c]) => (
          <div key={k} className="flex items-center gap-2 text-[12px]">
            <span className="w-20 shrink-0 truncate text-gray-500" title={k}>{k || "—"}</span>
            <div className="flex-1 h-1.5 rounded-full bg-gray-100 overflow-hidden">
              <div className="h-full rounded-full bg-[#4d6bfe]/70" style={{ width: `${(c / max) * 100}%` }} />
            </div>
            <span className="w-6 text-right text-gray-400">{c}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function MetricsPanel({ onClose }: { onClose: () => void }) {
  const { metrics, loading, error, refresh } = useMetrics();

  return (
    // 遮罩层：点空白关闭（同 MemoryPanel）
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={onClose}>
      <div
        className="w-[680px] max-w-[94vw] max-h-[86vh] flex flex-col bg-white rounded-xl shadow-xl border border-gray-200"
        onClick={e => e.stopPropagation()}
      >
        {/* 头部 */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#4d6bfe]" />
            <span className="font-semibold text-[15px]">质量看板</span>
            {metrics && (
              <span className="text-xs text-gray-400">
                （{metrics.range.days ? `近${metrics.range.days}天` : "全部"} · {metrics.range.total_requests} 次请求）
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button onClick={refresh} title="刷新" className="p-1 rounded text-gray-400 hover:text-[#4d6bfe] hover:bg-gray-100 transition-colors">
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            </button>
            <button onClick={onClose} title="关闭" className="p-1 rounded text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* 内容 */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
          {loading ? (
            <div className="text-sm text-gray-400 text-center py-10">加载中…</div>
          ) : error ? (
            <div className="text-sm text-center py-10 text-amber-600 flex flex-col items-center gap-2">
              <AlertTriangle className="w-5 h-5" /> {error}
            </div>
          ) : !metrics || metrics.range.total_requests === 0 ? (
            <div className="text-sm text-gray-400 text-center py-10">
              还没有真实请求数据。先跟牛来聊几句，trace 攒起来这里就亮了。
            </div>
          ) : (
            <>
              {/* 核心数字卡 */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                <Stat label="总请求" value={String(metrics.range.total_requests)} sub={`均 ${metrics.avg_calls} 次调用/请求`} />
                <Stat label="总成本" value={`¥${metrics.cost.total.toFixed(4)}`} sub={`均 ¥${metrics.cost.avg_per_request.toFixed(4)}/次`} accent="text-emerald-600" />
                <Stat label="平均延迟" value={`${metrics.latency.avg.toFixed(2)}s`} sub={`最慢 ${metrics.latency.max.toFixed(2)}s`} />
                <Stat label="降级率" value={pct(metrics.degrade_rate)} sub="触发兜底比例" accent={metrics.degrade_rate > 0.05 ? "text-red-500" : "text-gray-800"} />
              </div>

              {/* 检索质量 + 漏标 */}
              <div className="rounded-lg border border-gray-200/70 p-3.5 space-y-3">
                <div className="text-[13px] font-medium text-gray-700">检索质量（子查询级）</div>
                <RateBar label="命中率（质检判相关）" value={metrics.retrieval.hit_rate} color="bg-emerald-500" />
                <RateBar label="空手率（判无资料）" value={metrics.retrieval.empty_rate} color="bg-gray-400" />
                <RateBar label="重查率（触发纠正重写）" value={metrics.retrieval.retry_rate} color="bg-amber-500" />
                <div className="pt-2 border-t border-gray-100">
                  <RateBar label={`漏标率（检索到却没标 [n]，${metrics.citation.miss}/${metrics.citation.checked}）`} value={metrics.citation.miss_rate} color="bg-red-400" />
                </div>
              </div>

              {/* 分布 */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Dist title="意图分布" data={metrics.distribution.intents} />
                <Dist title="作用域分布" data={metrics.distribution.scopes} />
                <Dist title="身份分布" data={metrics.distribution.identity} />
              </div>

              {/* 各环节成本 */}
              <div className="rounded-lg border border-gray-200/70 p-3.5">
                <div className="text-[13px] font-medium text-gray-700 mb-2">各环节成本 / 延迟（谁最烧钱）</div>
                <div className="space-y-1.5">
                  {Object.entries(metrics.cost_by_purpose).sort((a, b) => b[1].cost - a[1].cost).map(([p, d]) => (
                    <div key={p} className="flex items-center gap-2 text-[12px]">
                      <span className="w-24 shrink-0 truncate text-gray-500">{p}</span>
                      <span className="text-gray-400 w-12 shrink-0">{d.calls}次</span>
                      <span className="text-emerald-600 w-20 shrink-0">¥{d.cost.toFixed(4)}</span>
                      <span className="text-gray-400 flex-1 text-right">{d.latency.toFixed(2)}s</span>
                    </div>
                  ))}
                  {Object.keys(metrics.cost_by_purpose).length === 0 && <div className="text-[11px] text-gray-300">无数据</div>}
                </div>
              </div>
            </>
          )}
        </div>

        {/* 底部说明 */}
        <div className="px-5 py-2.5 border-t border-gray-100 text-[11px] text-gray-400 leading-relaxed">
          数据来自每次真实请求的 trace 聚合（成本 / 延迟 / 降级 / 检索质检 / 漏标），仅亮哥可见。与离线评估互补：离线是期末考、这是行车记录仪。
        </div>
      </div>
    </div>
  );
}