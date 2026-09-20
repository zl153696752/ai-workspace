"""
步骤12 · e1：离线评估集生成器（开发期工具，不进运行时、不打包进 ModelScope）

干什么：读知识库语料 → 用 LLM 自动出问答题（问题 + 参考答案）→ 标注金标准来源
       → 产出 eval_set.json，供 e2（检索层评估）、e3（生成层评估）使用。

关键设计：
1. 数据源从 Chroma 库读切片（collection.get），不重新切文件——保证 gold_sources 的
   (filename, section, part) 一定对齐库里真实切片，e2 才能拿它跟检索结果比对。
2. 只处理 seed_docs 语料，过滤掉 uploads 杂物（无金标准价值）。
3. 提示词强制"真实用户口吻"改写、不照抄原文——防泄题（措辞太贴原文会让检索虚高）。
4. 对 N7/N7 Pro/N7s 这种相似型号，要求问题明确指向具体某个，覆盖语料埋的区分度陷阱。

怎么跑（在 backend 目录下）：python eval/build_eval_set.py
会逐片调 LLM（约 20~40 次），花点小钱、耗时 1~3 分钟。跑完务必人工抽检 eval_set.json。
"""
import os
import sys
import json

# 让脚本能 import app 包：把 backend 根目录（本文件的上一级）加进模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import collection          # 复用全局向量库单例
from app.llm_gateway import chat           # 复用统一模型入口（带重试/降级/埋点）

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_DIR = os.path.join(BACKEND_DIR, "seed_docs")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")

# ===== 出题官提示词 =====
_QA_SYSTEM = (
    "你是知识库评估集的出题官。给你一段知识库切片正文，请针对它出 1~2 道问答题，"
    "用于评估检索系统能否把用户的真实问题正确路由到这段资料。\n\n"
    "出题要求：\n"
    "1. 用真实用户的口吻提问、用你自己的话组织，【绝不照抄原文句子】——照抄会让检索虚高、测不出真实能力。"
    "例如原文是「牛来 N7 Pro 电池容量 5000mAh」，别问「牛来 N7 Pro 的电池容量是多少」（太贴原文），"
    "可以问「N7 Pro 续航咋样、电池多大」（更口语）。\n"
    "2. 若这段涉及相似但不同的对象（如 N7 / N7 Pro / N7s），问题必须明确指向其中某一个，答案也对应那一个，绝不混淆。\n"
    "3. gold_answer 必须是这段正文里能直接找到的准确事实，简洁。\n"
    "4. 若这段正文信息太少、不足以出一道有意义的题，返回空数组 []。\n\n"
    "严格只输出 JSON 数组，不要任何额外文字：\n"
    '[{"question": "...", "gold_answer": "..."}]'
)


def _parse_qa(raw: str) -> list[dict]:
    """把 LLM 返回的 JSON 数组解析成 [{question, gold_answer}, ...]，不合格则丢弃（防御性，绝不裸抛）。"""
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
        if isinstance(item, dict) and str(item.get("question", "")).strip() and str(item.get("gold_answer", "")).strip():
            out.append({"question": str(item["question"]).strip(), "gold_answer": str(item["gold_answer"]).strip()})
    return out


def _seed_filenames() -> set[str]:
    """收集 seed_docs/public 和 private 下的文件名，用于过滤掉 uploads 杂物。"""
    names = set()
    for sub in ("public", "private"):
        d = os.path.join(SEED_DIR, sub)
        if os.path.isdir(d):
            names.update(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    return names


def build():
    seed_names = _seed_filenames()
    data = collection.get(include=["documents", "metadatas"])   # 读全库切片正文 + 元数据
    docs, metas = data["documents"], data["metadatas"]

    # 只留 seed 语料的切片，并按 (filename, section, part) 去重（防库里同片重复出题）
    seen, chunks = set(), []
    for d, m in zip(docs, metas):
        if not m or m.get("filename") not in seed_names:
            continue
        key = (m.get("filename"), m.get("section"), m.get("part"))
        if key in seen:
            continue
        seen.add(key)
        chunks.append((d, m))
    print(f"[评估集] 库内切片 {len(docs)} 片，其中种子语料去重后 {len(chunks)} 片，开始逐片出题...")

    eval_set = []
    for i, (doc, meta) in enumerate(chunks):
        if not doc or not doc.strip():
            continue
        try:
            resp = chat(messages=[
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user", "content": f"切片正文：\n{doc}"},
            ], purpose="eval_gen", temperature=0.3)
            raw = (resp.choices[0].message.content or "").strip()  # chat() 返回 response 对象，取 content 才是字符串
        except Exception as exc:  # 单片失败降级跳过，不中断整批
            print(f"[评估集][降级] 第 {i+1}/{len(chunks)} 片出题失败：{exc}")
            continue
        for qa in _parse_qa(raw):
            eval_set.append({
                "id": f"eval_{len(eval_set)+1:03d}",
                "question": qa["question"],
                "gold_answer": qa["gold_answer"],
                "gold_sources": [{                               # 金标准来源 = 取材那片，脚本自动填
                    "filename": meta.get("filename", ""),
                    "section": meta.get("section", ""),
                    "part": meta.get("part", 0),
                }],
                "private": bool(meta.get("private", False)),
                "category": meta.get("filename", ""),
            })
        print(f"[评估集] 第 {i+1}/{len(chunks)} 片 → 累计 {len(eval_set)} 题")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)
    print(f"\n[评估集] 完成：共 {len(eval_set)} 道题，已写入 {OUT_PATH}")
    print("[评估集] 请打开 eval_set.json 人工抽检：删掉不合理的题、修正错答案——这是整个评估可信的地基。")


if __name__ == "__main__":
    build()