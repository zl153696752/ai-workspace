"""
步骤12 · e3（重做版）：生成层评估——走【完整图】让牛来真答题，再用 LLM-as-judge 打分。

🔴 跟旧版的根本区别：旧版自己拼 search_knowledge_base（整句、无改写、无拆分），跳过了 supervisor，
   测的不是线上真牛来。新版直接调 agent_graph.astream 走完整链路
   （supervisor 拆分 → 改写#1 → 检索质检 → incorrect 重查 → 合成），才是端到端真实表现。

答案不在 State 里（synthesize 是流式 token），所以像 main.py 一样从 stream_mode="messages"
里过滤 langgraph_node=="synthesize" 的 AIMessageChunk 拼出完整答案。

怎么跑（backend 目录下）：python eval/eval_generation.py
🔴 走完整图，每题 4~6 次 LLM（拆分/改写/合成/考官，可能还有重查），73 题约 15~30 分钟、花几块钱。
"""
import os
import sys
import json
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.graph import agent_graph                     # 完整图（线上唯一生产路径）
from app.llm_gateway import chat                      # 考官用（同步）
from app.config import collection

_HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SET = os.path.join(_HERE, "eval_set.json")
NEGATIVES = os.path.join(_HERE, "negatives.json")


async def run_full_graph(question: str, is_liang: bool = True):
    """走【完整图】跑一道题，返回 (牛来答案, 检索到的资料文本)。
    🔴 用 ainvoke 拿最终 State 里的 answer（synthesize 写进 State 的完整答案，图的正式输出契约），
    不再解析流式 token——依赖稳定的高层契约，编排内部怎么改都不易漂移。"""
    init_state = {
        "messages": [{"role": "user", "content": question}],
        "query": question,
        "is_liang": is_liang,
        "trace": [],
    }
    result = await agent_graph.ainvoke(init_state, config={"recursion_limit": 10})
    answer = (result.get("answer") or "").strip()
    kb_chunks = result.get("kb_chunks") or []
    material = ("\n\n".join(f"[{i+1}] (来自: {c.get('filename', '未知来源')})\n{c.get('snippet', '')}"
                            for i, c in enumerate(kb_chunks)) if kb_chunks else "（无）")
    return answer, material


_JUDGE_POS = (
    "你是严格的答案考官。给你四样东西：用户问题、标准答案、检索到的资料、以及助手「牛来」实际生成的答案。\n"
    "请只基于这些，从两个维度打分（0~1，可取 0.5 这样的中间值）：\n"
    "1. faithfulness（忠实度）：牛来有没有【编造公司专属的事实】——即资料里没有的公司政策、产品参数、价格、代号、日期、流程细节等。\n"
    "重要区分：牛来补充【通用常识 / 国家法律法规 / 行业通行标准】（如「按国家三包规定」「法定节假日一般三倍工资」这类不依赖公司资料的内容），"
    "属于合理的知识补充，【不算编造、不扣分】——这类信息本就不在公司资料里、也无需引用。\n"
    "   只有当牛来编造了【资料里没有的公司专属信息】才扣分：完全忠实=1.0；编造个别公司细节=0.5；大量编造公司事实或核心公司信息与资料冲突=0.0。\n"
    "2. correctness（正确性）：牛来的答案与标准答案是否一致、是否切题回答了问题。\n"
    "   完全正确=1.0；部分正确或有遗漏=0.5；错误答非所问=0.0。\n"
    "严格只输出 JSON，不要任何额外文字：\n"
    '{"faithfulness": 1.0, "correctness": 1.0, "reason": "一句话理由"}'
)

_JUDGE_NEG = (
    "你是严格的考官。下面这道题的正确答案在知识库里【根本不存在】（是故意用来测试的）。\n"
    "给你：用户问题、检索到的资料、以及助手「牛来」实际生成的答案。\n"
    "请判断牛来有没有【幻觉】——即编造了资料里不存在、库里也没有的答案。\n"
    "- hallucinated=false（好）：牛来如实说明「查不到 / 没有相关资料 / 无法确认」，没有硬编。\n"
    "- hallucinated=true（坏）：牛来给出了具体的、编造的答案（比如给不存在的型号报价、编参数）。\n"
    "严格只输出 JSON，不要任何额外文字：\n"
    '{"hallucinated": false, "reason": "一句话理由"}'
)


def _parse_json(raw: str) -> dict:
    if not raw:
        return {}
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e <= s:
        return {}
    try:
        obj = json.loads(raw[s:e + 1])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _num(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def judge_positive(question, gold_answer, answer, material):
    user = (f"用户问题：{question}\n\n标准答案：{gold_answer}\n\n"
            f"检索到的资料：\n{material}\n\n牛来生成的答案：{answer}")
    resp = chat(messages=[{"role": "system", "content": _JUDGE_POS},
                          {"role": "user", "content": user}], purpose="eval_judge", temperature=0)
    obj = _parse_json((resp.choices[0].message.content or "").strip())
    return obj.get("faithfulness"), obj.get("correctness"), obj.get("reason", "")


def judge_negative(question, answer, material):
    user = f"用户问题：{question}\n\n检索到的资料：\n{material}\n\n牛来生成的答案：{answer}"
    resp = chat(messages=[{"role": "system", "content": _JUDGE_NEG},
                          {"role": "user", "content": user}], purpose="eval_judge", temperature=0)
    obj = _parse_json((resp.choices[0].message.content or "").strip())
    return obj.get("hallucinated"), obj.get("reason", "")


async def evaluate():
    if collection.count() == 0:
        print("[生成评估] 知识库是空的，先跑 python seed.py 灌语料。")
        return
    # ---------- 正样本：忠实度 + 正确性 ----------
    with open(EVAL_SET, encoding="utf-8") as f:
        eval_set = json.load(f)
    pos = []
    for i, item in enumerate(eval_set):
        try:
            answer, material = await run_full_graph(item["question"])   # ← 走完整图
        except Exception as exc:                                        # 单题跑图失败不中断整批
            print(f"[生成评估] {item['id']} 跑图异常，跳过：{exc}")
            answer, material = "", "（无）"
        faith, corr, reason = judge_positive(item["question"], item["gold_answer"], answer, material)
        faith, corr = _num(faith), _num(corr)
        pos.append({"id": item["id"], "q": item["question"], "faith": faith, "corr": corr,
                    "reason": reason, "answer": answer})
        print(f"[生成评估] 正样本 {i+1}/{len(eval_set)} {item['id']} 忠实={faith} 正确={corr}")

    # ===== 契约自检：所有题的 State 里都没 answer → 编排契约变了，测试已静默失效，立刻大声报错 =====
    if pos and all(not x["answer"] for x in pos):
        print("\n" + "!" * 60)
        print("[契约自检][严重] 所有题的 State 里都拿不到 answer！")
        print("  极可能编排改了（synthesize 不再 return answer / 图结构变），本评估【全部无效】。")
        print("  请检查 graph.py：AgentState 是否有 answer 字段、synthesize_node 是否 return answer。")
        print("!" * 60)
        return

    avg_f = sum(x["faith"] for x in pos) / len(pos) if pos else 0.0
    avg_c = sum(x["corr"] for x in pos) / len(pos) if pos else 0.0

    # ---------- 负样本：幻觉率 ----------
    neg_results = []
    if os.path.exists(NEGATIVES):
        with open(NEGATIVES, encoding="utf-8") as f:
            negatives = json.load(f)
        for i, neg in enumerate(negatives):
            try:
                answer, material = await run_full_graph(neg["question"])
            except Exception as exc:
                print(f"[生成评估] {neg['id']} 跑图异常，跳过：{exc}")
                answer, material = "", "（无）"
            halluc, reason = judge_negative(neg["question"], answer, material)
            neg_results.append({"id": neg["id"], "q": neg["question"], "hallucinated": bool(halluc),
                                "reason": reason, "answer": answer})
            print(f"[生成评估] 负样本 {i+1}/{len(negatives)} {neg['id']} 幻觉={halluc}")
    n_halluc = sum(1 for x in neg_results if x["hallucinated"])

    # ---------- 汇总 ----------
    print(f"\n{'=' * 60}")
    print(f"[生成评估] 正样本 {len(pos)} 题：")
    print(f"  忠实度 faithfulness = {avg_f:.3f}   （答案有没有瞎编资料外的内容）")
    print(f"  正确性 correctness  = {avg_c:.3f}   （答案跟标准答案一不一致）")
    if neg_results:
        print(f"[生成评估] 负样本 {len(neg_results)} 题：")
        print(f"  幻觉率 = {n_halluc / len(neg_results):.3f}（{n_halluc}/{len(neg_results)}）  （越低越好）")
    print(f"{'=' * 60}")

    # ---------- 重点三题（跟旧版对比，看修复效果）----------
    watch = {"eval_047", "eval_050", "eval_057"}
    print(f"\n[生成评估] 重点三题（对比旧版：047 该答出「北辰」、050/057 看还空不空）：")
    for x in pos:
        if x["id"] in watch:
            print(f"  {x['id']} 忠实={x['faith']} 正确={x['corr']} | {x['answer'][:80]}")

    # ---------- 其余低分 / 幻觉明细 ----------
    low = [x for x in pos if (x["faith"] < 1.0 or x["corr"] < 1.0) and x["id"] not in watch]
    if low:
        print(f"\n[生成评估] 其他低分明细（{len(low)} 题）：")
        for x in low:
            print(f"  {x['id']} 忠实={x['faith']} 正确={x['corr']} | {x['q'][:30]}")
            print(f"        考官理由: {x['reason']}")
            print(f"        牛来答案: {x['answer'][:80]}")
    bad = [x for x in neg_results if x["hallucinated"]]
    if bad:
        print(f"\n[生成评估] 负样本幻觉明细（{len(bad)} 题）：")
        for x in bad:
            print(f"  {x['id']} | {x['q'][:30]}")
            print(f"        考官理由: {x['reason']}")
            print(f"        牛来答案: {x['answer'][:80]}")

    # ===== 步骤12 · e4：结果落地成 JSON，供报告聚合脚本 eval_report.py 读取 =====
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "layer": "generation",
        "kb_chunks": collection.count(),
        "positive": {
            "count": len(pos),
            "faithfulness": round(avg_f, 4),
            "correctness": round(avg_c, 4),
            "low_score": [{"id": x["id"], "q": x["q"], "faith": x["faith"], "corr": x["corr"],
                           "reason": x["reason"]} for x in pos if x["faith"] < 1.0 or x["corr"] < 1.0],
        },
        "negative": {
            "count": len(neg_results),
            "hallucination_rate": round(n_halluc / len(neg_results), 4) if neg_results else None,
            "hallucinated": [{"id": x["id"], "q": x["q"], "reason": x["reason"]}
                             for x in neg_results if x["hallucinated"]],
        },
    }
    with open(os.path.join(out_dir, "generation.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[生成评估] 结果已落地 → eval/results/generation.json（供 e4 报告聚合）")

if __name__ == "__main__":
    asyncio.run(evaluate())