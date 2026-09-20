"""
步骤12 · e2：检索层评估——拿 e1 的 61 道题考"检索系统"，看它能不能翻到正确资料。

大白话：e1 出好了考卷（eval_set.json，每题标了"标准答案出自哪一段"=gold_sources）。
       e2 让检索系统去答这些题，检查它翻回来的资料有没有命中我们标的"正确出处"。

判分方式（关键）：
- 每题拿 question 跑线上同款检索 _retrieve_once，拿回最多 3 段资料（TOP_K=3）。
- 把每段资料的 (文件名, 章节) 跟 gold_sources 的 (文件名, 章节) 比，命中就算"翻对了"。
- 放宽到"章节"级（不纠结章节内第几段 part）：避免"资料印在同章不同段"时被冤枉。

三个指标（都是 0~1，越大越好）：
- Recall@3：3 段结果里命中标准出处的比例（有没有翻到）。
- MRR：第一段命中的排名倒数（翻对得早不早；第1段=1.0，第2段=0.5，第3段≈0.33）。
- NDCG@3：综合"翻对没有 + 排得靠不靠前"的排序质量分。

怎么跑（backend 目录下）：python eval/eval_retrieval.py
跑完把终端输出（尤其 MISS 明细）发我，我判断哪些"被冤枉"、哪些"真翻错"。
"""
import os
import sys
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag import _retrieve_once, TOP_K       # 线上同款检索（召回→融合→精排）
from app.config import collection

EVAL_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")


def _key(meta: dict) -> tuple:
    """从切片元数据取"匹配键"= (文件名, 章节)。放宽到章节级，忽略 part（段号）。"""
    return (meta.get("filename", ""), meta.get("section", ""))


def _recall(retrieved: list, gold: set) -> float:
    """3 段结果里命中标准出处的比例。gold 通常 1 条，故每题非 0 即 1。"""
    if not gold:
        return 0.0
    return sum(1 for g in gold if g in retrieved) / len(gold)


def _mrr(retrieved: list, gold: set) -> float:
    """第一段命中的排名倒数：第1段=1.0，第2段=0.5，第3段≈0.33，没命中=0。"""
    for i, key in enumerate(retrieved):
        if key in gold:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg(retrieved: list, gold: set, k: int = 3) -> float:
    """排序质量：命中且排得越靠前，分越高。"""
    dcg = sum(1.0 / math.log2(i + 2) for i, key in enumerate(retrieved[:k]) if key in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def evaluate():
    if collection.count() == 0:
        print("[检索评估] 知识库是空的，先跑 python seed.py 灌语料再评估。")
        return
    with open(EVAL_SET, encoding="utf-8") as f:
        eval_set = json.load(f)

    rows = []
    for item in eval_set:
        q = item["question"]
        gold = set(_key(g) for g in item["gold_sources"])
        hits = _retrieve_once(q, allow_private=True)  # 亮哥身份：public+private 题都能检索到
        # 按 (文件名,章节) 去重、保留首次名次：同一章的多段切片只算"翻到这一章一次"。
        # 不去重的话，同一章的多段会在 NDCG 分子里重复加分，导致 NDCG > 1（就是刚才 1.015 的 bug）。
        retrieved, seen = [], set()
        for _score, _doc, meta in hits:
            key = _key(meta)
            if key in seen:
                continue
            seen.add(key)
            retrieved.append(key)
        r, m, n = _recall(retrieved, gold), _mrr(retrieved, gold), _ndcg(retrieved, gold)
        rows.append({"id": item["id"], "q": q, "gold": gold, "retrieved": retrieved,
                     "recall": r, "mrr": m, "ndcg": n, "hit": r > 0})

    total = len(rows)
    avg_r = sum(x["recall"] for x in rows) / total
    avg_m = sum(x["mrr"] for x in rows) / total
    avg_n = sum(x["ndcg"] for x in rows) / total
    n_hit = sum(1 for x in rows if x["hit"])

    print(f"\n{'=' * 60}")
    print(f"[检索评估] 共 {total} 题   命中 {n_hit}   未命中 {total - n_hit}")
    print(f"  Recall@{TOP_K} = {avg_r:.3f}   （翻到正确资料的比例）")
    print(f"  MRR         = {avg_m:.3f}   （翻对得早不早）")
    print(f"  NDCG@{TOP_K}     = {avg_n:.3f}   （排序质量）")
    print(f"{'=' * 60}")

    miss = [x for x in rows if not x["hit"]]
    if miss:
        print(f"\n[检索评估] 未命中 {len(miss)} 题明细（发我看，判断被冤枉还是真翻错）：")
        for x in miss:
            print(f"  {x['id']} | {x['q'][:32]}")
            print(f"        标准出处: {sorted(x['gold'])}")
            print(f"        实际翻到: {x['retrieved']}")
    else:
        print("\n[检索评估] 全部命中，检索层满分。")

    # ===== 步骤12 · e4：结果落地成 JSON，供报告聚合脚本 eval_report.py 读取（e4 不重跑、零 token）=====
    out_dir = os.path.join(os.path.dirname(EVAL_SET), "results")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "layer": "retrieval",
        "kb_chunks": collection.count(),
        "total": total,
        "hit": n_hit,
        "top_k": TOP_K,
        "metrics": {"recall": round(avg_r, 4), "mrr": round(avg_m, 4), "ndcg": round(avg_n, 4)},
        "misses": [{"id": x["id"], "q": x["q"], "gold": sorted(x["gold"]), "retrieved": x["retrieved"]}
                   for x in miss],
    }
    with open(os.path.join(out_dir, "retrieval.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[检索评估] 结果已落地 → eval/results/retrieval.json（供 e4 报告聚合）")

if __name__ == "__main__":
    evaluate()