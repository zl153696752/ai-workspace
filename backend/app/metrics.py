"""
步骤12 · 12B o1：在线监控聚合——把 traces 表里每次真实请求的运行时数据，聚合成质量看板要的统计指标。

🔴 只读聚合、不新增采集：步骤9 的可观测性已经把成本/延迟/降级/检索质检 grade/重查/漏标全埋进 traces 表了，
   这里只是"把躺着的数据算成看板指标"。跟 12A 离线评估互补——离线是"期末考试"(固定题库、有标准答案)，
   在线是"行车记录仪"(真实请求、无标准答案，靠运行时信号看健康度)。

数据分三处，对应三种聚合方式：
   ① traces 表的【列】(cost/latency/degraded/intents/scope/identity) → 直接 SQL 聚合；
   ② node_spans【JSON】(检索 grade、crag_retry 重查、citation_check 漏标) → 逐行解析统计；
   ③ llm_spans【JSON】(每次模型调用的 purpose/token/cost/latency) → 按环节归并。
"""
import json
from .db import _cursor   # 复用 db.py 的统一连接管理（正常 commit、异常 rollback、必 close），不自己开连接


def _spans(raw):
    """把存成 JSON 文本的 span 列安全解析回 list；空/损坏都返回 []，绝不让看板因一条脏数据崩掉。"""
    try:
        data = json.loads(raw) if raw else []
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _rate(numerator, denominator):
    """算比例；分母为 0 返回 None（前端显示"—"，而不是崩、也不是显示假的 0）。"""
    return round(numerator / denominator, 4) if denominator else None


def collect_metrics(days: int | None = None) -> dict:
    """聚合质量看板的全部指标。days=None 聚合历史全部；传 N 则只聚合最近 N 天。"""
    where = "WHERE ts >= datetime('now','localtime',?)" if days else ""
    params = (f"-{int(days)} days",) if days else ()

    with _cursor() as conn:
        # ---------- ① 列级 SQL 聚合（成本/延迟/降级/调用次数/token）----------
        agg = conn.execute(f"""
            SELECT COUNT(*)                               AS n,
                   AVG(degraded)                          AS degrade_rate,
                   AVG(total_latency)                     AS avg_latency,
                   MAX(total_latency)                     AS max_latency,
                   AVG(calls)                             AS avg_calls,
                   SUM(total_cost)                        AS sum_cost,
                   AVG(total_cost)                        AS avg_cost,
                   SUM(prompt_tokens + completion_tokens) AS sum_tokens
            FROM traces {where}
        """, params).fetchone()

        def _dist(column):
            """按某列 GROUP BY 出分布（column 是代码写死的列名，非用户输入，无注入风险）。"""
            rows = conn.execute(
                f"SELECT {column} AS k, COUNT(*) AS c FROM traces {where} GROUP BY {column} ORDER BY c DESC",
                params).fetchall()
            return {(r["k"] or "未分类"): r["c"] for r in rows}

        intents, scopes, identity = _dist("intents"), _dist("scope"), _dist("identity")

        # ---------- ②③ 逐行解析两类 span ----------
        rows = conn.execute(f"SELECT node_spans, llm_spans FROM traces {where}", params).fetchall()

    n_retrieve = n_correct = n_incorrect = n_unavailable = n_retry = 0
    n_cite_checked = n_citation_miss = 0
    purpose = {}
    for r in rows:
        for s in _spans(r["node_spans"]):
            # 检索质检 grade（每个 retrieve 子查询记一条）→ 命中率/空手率
            if s.get("node") == "retrieve" and "grade" in s:
                n_retrieve += 1
                g = s.get("grade")
                if g == "correct":
                    n_correct += 1
                elif g == "incorrect":
                    n_incorrect += 1
                elif g == "unavailable":
                    n_unavailable += 1
            if s.get("event") == "crag_retry":          # Self-CRAG 纠正性重查触发
                n_retry += 1
            if s.get("event") == "citation_check":      # 漏标：检索到资料却没标任何 [n]
                n_cite_checked += 1
                if not s.get("cited"):
                    n_citation_miss += 1
        for s in _spans(r["llm_spans"]):                # 各环节成本/延迟归并
            p = s.get("purpose", "未知")
            d = purpose.setdefault(p, {"calls": 0, "cost": 0.0, "latency": 0.0})
            d["calls"] += 1
            d["cost"] = round(d["cost"] + (s.get("cost") or 0.0), 6)
            d["latency"] = round(d["latency"] + (s.get("latency") or 0.0), 3)

    return {
        "range": {"days": days, "total_requests": agg["n"] or 0},
        "cost": {
            "total": round(agg["sum_cost"] or 0.0, 4),
            "avg_per_request": round(agg["avg_cost"] or 0.0, 6),
            "total_tokens": agg["sum_tokens"] or 0,
        },
        "latency": {"avg": round(agg["avg_latency"] or 0.0, 3), "max": round(agg["max_latency"] or 0.0, 3)},
        "degrade_rate": round(agg["degrade_rate"] or 0.0, 4),
        "avg_calls": round(agg["avg_calls"] or 0.0, 2),
        "retrieval": {
            "subqueries": n_retrieve,
            "hit_rate": _rate(n_correct, n_retrieve),         # 检索命中率（子查询级）
            "empty_rate": _rate(n_unavailable, n_retrieve),   # 空手率（质检判无资料）
            "retry_rate": _rate(n_retry, n_retrieve),         # 重查触发率
            "grade_dist": {"correct": n_correct, "incorrect": n_incorrect, "unavailable": n_unavailable},
        },
        "citation": {
            "checked": n_cite_checked, "miss": n_citation_miss,
            "miss_rate": _rate(n_citation_miss, n_cite_checked),   # 漏标率
        },
        "distribution": {"intents": intents, "scopes": scopes, "identity": identity},
        "cost_by_purpose": purpose,
    }