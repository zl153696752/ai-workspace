"""
步骤12 · e1b：负样本生成器——出"牛来知识库里根本查不到答案"的问题。

为什么必需：① 现有 61 题全是"库里有答案"的正样本，测不出检索"该拒绝时会不会硬凑"（瞎命中）；
           ② e5 标定 Self-CRAG 阈值，必须有"不相关"的负样本当低分参照，否则找不到分界线。

存到 negatives.json（单独文件，不动 eval_set.json、不改 e2）。
怎么跑（backend 目录下）：python eval/build_negatives.py
🔴 跑完务必人工抽检：LLM 可能生成"其实库里有"的题（那就不算负样本），要删掉。
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import collection
from app.llm_gateway import chat

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_DIR = os.path.join(BACKEND_DIR, "seed_docs")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "negatives.json")

_NEG_SYSTEM = (
    "你是评估集出题官，这次要出【负样本】——即牛来数码知识库里【根本查不到答案】的问题，"
    "用来测试检索系统会不会「瞎命中」（库里没有却硬凑一个资料返回）。\n\n"
    "我会给你知识库覆盖的主题范围。请生成 12 个「看起来跟牛来数码相关、但资料里确实没有答案」的问题，类型参考：\n"
    "1. 不存在的型号或产品（如 N7 Ultra、N7 mini、牛来平板）；\n"
    "2. 资料没提及的属性（如防水等级、5G 频段、系统版本、机身材质、屏幕供应商之外的参数）；\n"
    "3. 超出资料范围的公司信息（如股票代码、CEO 姓名、融资情况、年营收）。\n\n"
    "🔴 铁律：务必确保这些问题在上面列出的资料主题里【找不到确切答案】。"
    "若某个问题资料里其实有（比如问发布时间、已知型号的价格），就绝对不要出。\n\n"
    "严格只输出 JSON 数组，不要任何额外文字：\n"
    '[{"question": "...", "why_absent": "一句话说明为什么库里查不到"}]'
)


def _parse_neg(raw: str) -> list:
    """解析 LLM 返回的负样本 JSON 数组，不合格的丢弃。"""
    if not raw:
        return []
    s, e = raw.find("["), raw.rfind("]")
    if s < 0 or e <= s:
        return []
    try:
        arr = json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for item in arr if isinstance(arr, list) else []:
        if isinstance(item, dict) and str(item.get("question", "")).strip():
            out.append({"question": str(item["question"]).strip(),
                        "why_absent": str(item.get("why_absent", "")).strip()})
    return out


def _topic_map() -> str:
    """聚合库里 seed 语料的「主题地图」（文件名 → 章节列表），喂给 LLM 当"库里有啥"的参照。"""
    seed_names = set()
    for sub in ("public", "private"):
        d = os.path.join(SEED_DIR, sub)
        if os.path.isdir(d):
            seed_names.update(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    data = collection.get(include=["metadatas"])
    m = {}
    for meta in data["metadatas"]:
        if not meta or meta.get("filename") not in seed_names:
            continue
        m.setdefault(meta.get("filename", ""), set()).add(meta.get("section", "") or "(开头简介)")
    return "\n".join(f"- {fn}：{', '.join(sorted(secs))}" for fn, secs in m.items())


def build():
    topic_map = _topic_map()
    resp = chat(messages=[
        {"role": "system", "content": _NEG_SYSTEM},
        {"role": "user", "content": f"知识库覆盖的主题范围：\n{topic_map}"},
    ], purpose="eval_neg", temperature=0.7)
    raw = (resp.choices[0].message.content or "").strip()   # chat() 返回 response 对象，取 content
    negs = _parse_neg(raw)

    out = [{"id": f"neg_{i+1:03d}", "question": n["question"], "why_absent": n["why_absent"],
            "expected": "absent", "gold_sources": []} for i, n in enumerate(negs)]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[负样本] 生成 {len(out)} 道，已写入 {OUT_PATH}")
    print("[负样本] 请抽检：逐条确认「库里真查不到」（看 why_absent）；若某条其实库里有，删掉它。")


if __name__ == "__main__":
    build()