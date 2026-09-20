"""
步骤12 · e4：报告聚合——读 e2/e3 落地的结果 JSON，拼成人类可读的 markdown 报告 + 机器可读的 summary.json。

🔴 关键设计：e4【不重跑任何评估】（重跑 e3 走完整图又慢又烧 token），只读现成结果文件：
   - eval/results/retrieval.json（e2 检索层落地）
   - eval/results/generation.json（e3 生成层落地）
   谁没跑就跳过谁、报告里标"未评估"。你什么时候重跑 e2/e3，再跑一次 e4，报告自动更新。

产出：
   - eval/results/report.md    ：给人看的报告（可贴文档 / 给面试官看）
   - eval/results/summary.json ：给机器看的结构化汇总（未来在线面板可复用）

怎么跑（backend 目录下）：python eval/eval_report.py
"""
import os
import json
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")
_RETRIEVAL = os.path.join(_RESULTS, "retrieval.json")
_GENERATION = os.path.join(_RESULTS, "generation.json")


def _load(path):
    """读结果 JSON；不存在或损坏就返回 None（报告里对应部分标"未评估"）。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _grade(value, reverse=False):
    """给指标打红绿灯。reverse=True 表示"越低越好"（如幻觉率）。"""
    if value is None:
        return "—"
    if reverse:
        return "✅" if value <= 0.05 else ("⚠️" if value <= 0.2 else "❌")
    return "✅" if value >= 0.9 else ("⚠️" if value >= 0.7 else "❌")


def _fmt(value):
    return "未评估" if value is None else f"{value:.3f}"


def build_conclusion(retr, gen):
    """根据指标自动生成几条结论（红灯项优先提示）。"""
    out = []
    if retr:
        m = retr.get("metrics", {})
        if m.get("recall") is not None and m["recall"] >= 0.999:
            out.append("- **检索层召回满分**：正确资料都能翻到；但当前知识库切片数少，Recall 满分是客观必然，"
                       "重点看 MRR/NDCG 的排序质量，别把满分当“检索无敌”。")
        if m.get("ndcg") is not None and m["ndcg"] < 0.9:
            out.append(f"- **检索排序有优化空间**：NDCG={m['ndcg']:.3f} < 0.9，正确资料没稳定排在最前，可查改写/精排。")
        if not retr.get("misses"):
            out.append("- 无未命中题，检索层暂无需修正。")
    if gen:
        p, n = gen.get("positive", {}), gen.get("negative", {})
        hr = n.get("hallucination_rate")
        if hr is not None and hr <= 0.05:
            out.append(f"- **负样本幻觉率 {_fmt(hr)}**：库里没有的题基本都老实拒答，“该拒绝时能拒绝”是企业级 RAG 的关键素质。")
        f = p.get("faithfulness")
        if f is not None and f < 0.9:
            out.append(f"- **忠实度 {_fmt(f)} 偏低**：低分多集中在 both scope（公司规定+国家标准）题——"
                       "先确认是 judge 口径问题还是牛来真编造，再决定改提示词还是改 judge。")
        c = p.get("correctness")
        if c is not None and c >= 0.9:
            out.append(f"- **正确性 {_fmt(c)}**：绝大多数题答得对、切题。")
    if not retr and not gen:
        out.append("_两层都未评估，先跑 e2 / e3 再回来生成报告。_")
    return out


def build_report(retr, gen):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kb = (retr or gen or {}).get("kb_chunks")
    lines = [
        "# 牛来 RAG 质量评估报告",
        "",
        f"> 生成时间：{now}  ",
        f"> 知识库规模：{kb if kb is not None else '未知'} 个切片  ",
        "> 评估套件：步骤12 离线评估（e2 检索层 + e3 生成层 + e1b 负样本）",
        "",
        "## 一、总览",
        "",
        "| 层 | 指标 | 得分 | 评级 |",
        "| --- | --- | --- | --- |",
    ]
    if retr:
        m, k = retr.get("metrics", {}), retr.get("top_k", 3)
        lines.append(f"| 检索层 | Recall@{k} | {_fmt(m.get('recall'))} | {_grade(m.get('recall'))} |")
        lines.append(f"| 检索层 | MRR | {_fmt(m.get('mrr'))} | {_grade(m.get('mrr'))} |")
        lines.append(f"| 检索层 | NDCG@{k} | {_fmt(m.get('ndcg'))} | {_grade(m.get('ndcg'))} |")
    else:
        lines.append("| 检索层 | — | 未评估 | — |")
    if gen:
        p, n = gen.get("positive", {}), gen.get("negative", {})
        lines.append(f"| 生成层 | 忠实度 faithfulness | {_fmt(p.get('faithfulness'))} | {_grade(p.get('faithfulness'))} |")
        lines.append(f"| 生成层 | 正确性 correctness | {_fmt(p.get('correctness'))} | {_grade(p.get('correctness'))} |")
        hr = n.get("hallucination_rate")
        lines.append(f"| 生成层 | 负样本幻觉率 | {_fmt(hr)} | {_grade(hr, reverse=True)} |")
    else:
        lines.append("| 生成层 | — | 未评估 | — |")
    lines.append("")

    # ---------- 二、检索层详情 ----------
    lines += ["## 二、检索层详情（e2）", ""]
    if retr:
        m, k = retr.get("metrics", {}), retr.get("top_k", 3)
        lines.append(f"- 考题数：{retr.get('total')}，命中 {retr.get('hit')}，未命中 {retr.get('total', 0) - retr.get('hit', 0)}")
        lines.append(f"- Recall@{k} = {_fmt(m.get('recall'))}（结果里翻到正确资料的比例）")
        lines.append(f"- MRR = {_fmt(m.get('mrr'))}（翻对得早不早）")
        lines.append(f"- NDCG@{k} = {_fmt(m.get('ndcg'))}（排序质量）")
        lines.append("")
        misses = retr.get("misses", [])
        if misses:
            lines.append(f"**未命中 {len(misses)} 题明细：**")
            lines.append("")
            for x in misses:
                lines.append(f"- `{x['id']}` {x['q']}")
                lines.append(f"  - 标准出处：{x['gold']}")
                lines.append(f"  - 实际翻到：{x['retrieved']}")
        else:
            lines.append("**全部命中，检索层满分。**")
            lines.append("")
            lines.append("> 注：小库（切片数少）+ TOP_3 下 Recall 满分是客观必然，真正的区分度看 MRR/NDCG。")
    else:
        lines.append("_未跑 e2（检索层评估），或结果文件缺失。跑 `python eval/eval_retrieval.py` 后重试。_")
    lines.append("")

    # ---------- 三、生成层详情 ----------
    lines += ["## 三、生成层详情（e3）", ""]
    if gen:
        p, n = gen.get("positive", {}), gen.get("negative", {})
        lines += ["### 正样本（走完整图 + LLM-as-judge）", ""]
        lines.append(f"- 考题数：{p.get('count')}")
        lines.append(f"- 忠实度 faithfulness = {_fmt(p.get('faithfulness'))}（有没有编造资料外的公司事实）")
        lines.append(f"- 正确性 correctness = {_fmt(p.get('correctness'))}（跟标准答案一不一致）")
        lines.append("")
        low = p.get("low_score", [])
        if low:
            lines.append(f"**低分 {len(low)} 题明细：**")
            lines.append("")
            for x in low:
                lines.append(f"- `{x['id']}` 忠实={x['faith']} 正确={x['corr']} | {x['q']}")
                lines.append(f"  - 考官理由：{x.get('reason', '')}")
        else:
            lines.append("**正样本全部满分。**")
        lines += ["", "### 负样本（库里没有的题，测幻觉）", ""]
        lines.append(f"- 考题数：{n.get('count')}")
        lines.append(f"- 幻觉率 = {_fmt(n.get('hallucination_rate'))}（越低越好；0 表示全部老实拒答）")
        bad = n.get("hallucinated", [])
        if bad:
            lines.append("")
            lines.append(f"**幻觉 {len(bad)} 题明细：**")
            lines.append("")
            for x in bad:
                lines.append(f"- `{x['id']}` {x['q']}")
                lines.append(f"  - 考官理由：{x.get('reason', '')}")
    else:
        lines.append("_未跑 e3（生成层评估），或结果文件缺失。跑 `python eval/eval_generation.py` 后重试。_")
    lines.append("")

    # ---------- 四、结论与建议 ----------
    lines += ["## 四、结论与建议", ""]
    lines += build_conclusion(retr, gen)
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(_RESULTS, exist_ok=True)
    retr, gen = _load(_RETRIEVAL), _load(_GENERATION)
    if not retr and not gen:
        print("[报告] 没找到任何评估结果（eval/results/ 下 retrieval.json、generation.json 都不存在）。")
        print("      先跑 python eval/eval_retrieval.py（免费快）和/或 python eval/eval_generation.py（走完整图、烧 token）。")
        return

    with open(os.path.join(_RESULTS, "report.md"), "w", encoding="utf-8") as f:
        f.write(build_report(retr, gen))
    with open(os.path.join(_RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "retrieval": retr, "generation": gen}, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("[报告] 已生成：")
    print("  - Markdown 报告：eval/results/report.md")
    print("  - 结构化汇总  ：eval/results/summary.json")
    print(f"{'=' * 60}")
    if retr:
        m = retr["metrics"]
        print(f"[检索层] Recall@{retr.get('top_k', 3)}={m['recall']:.3f}  MRR={m['mrr']:.3f}  NDCG={m['ndcg']:.3f}")
    if gen:
        p, n = gen["positive"], gen["negative"]
        print(f"[生成层] 忠实={p['faithfulness']:.3f}  正确={p['correctness']:.3f}  幻觉率={_fmt(n.get('hallucination_rate'))}")


if __name__ == "__main__":
    main()