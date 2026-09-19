# ===== 步骤11：记忆官（长期记忆抽取）=====
# 从对话里抽取亮哥本人的【偏好 + 稳定事实】，保守写入 memories 表（复用 s1 的 upsert_memory）。
# 🔴 第一铁律：宁可漏记，不可错记——提示词、解析、入库三道关卡层层设防，拿不准一律不抽。
# 🔴 定位：记忆官是【横切基础设施】，不是图里的第 5 个 Worker——由 main.py 在回答后【异步】调用，绝不进主链路、不阻塞 SSE。
import json
import re

from .llm_gateway import chat   # 复用 supervisor 同款 LLM 入口（自带 token 埋点 + 降级留痕）
from .db import upsert_memory   # s1 建的去重更新写入

# 只认这两类记忆（砍掉对话摘要/承诺——那是错记重灾区）
_VALID_TYPES = ("preference", "fact")

# 🔴 敏感信息硬过滤（双保险之二：提示词已要求不记，这里再用正则拦一道，防模型不听话）
# content 或 source 命中任一模式 → 整条丢弃，绝不入库
_SENSITIVE_RE = re.compile(
    r"(密码|passwd|password|api[\s_-]?key|secret|token|私钥|密钥|sk-[A-Za-z0-9]"
    r"|身份证|银行卡|\d{17}[\dXx]|1[3-9]\d{9})",
    re.IGNORECASE,
)


def build_extract_system() -> str:
    """记忆官提示词（参照 build_intent_system 风格：角色 + 记什么 + 铁律 + 输出格式 + 示例）。
    🔴 全篇围绕「宁可漏记不可错记」：明确主体、只偏好+事实、拿不准不抽、带原文、不记敏感。"""
    return (
        "你是【记忆官】。从对话里抽取值得长期记住的、关于用户【亮哥本人】的信息，供以后个性化回答。"
        "第一铁律：宁可漏记，绝不错记——拿不准的一律不抽。\n\n"
        "一、只记这两类：\n"
        "- preference 偏好：稳定的喜好/习惯/风格（如：偏好详细讲解并多举例、常用 PyCharm）。\n"
        "- fact 稳定事实：不太会变的身份/背景（如：后端出身、正在准备 AI 方向面试）。\n\n"
        "二、绝不记（任一命中就丢弃，输出里不要出现）：\n"
        "1. 主体不是亮哥本人的：别人的偏好、假设句（如果我是、假如）、反问、举例里的第三方。\n"
        "2. 易过时的：对话摘要（刚才聊了X）、临时状态（现在有点累）、待办承诺（等下要改X）、带时效的词（明天、这次）。\n"
        "3. 敏感信息：密码、API key、token、密钥、身份证号、银行卡号、手机号——一律不记，哪怕用户明说。\n"
        "4. 本轮的提问本身：用户问的知识（如：年假几天）不是记忆，别把提问当成偏好。\n"
        "5. 拿不准的：任何模棱两可、需要你推断的，一律不记；只记用户明确陈述的。\n\n"
        "三、每条记忆给出这些字段：\n"
        "- type：preference 或 fact。\n"
        "- content：描述性陈述句，主语用「亮哥」，不用指令口吻（写「亮哥偏好详细讲解」，别写「要详细讲解」）。\n"
        "- confidence：0~1 置信度。明确直白的陈述给 0.85 以上；稍有推断给 0.6~0.8；低于 0.6 的干脆别输出。\n"
        "- source：支撑这条记忆的用户原话片段，用于回溯核查。\n"
        "- topic：主题键（如：回答风格、技术背景、开发工具），同一主题只保留一条，用于去重更新。\n\n"
        "四、输出格式：只输出一个 JSON 数组，不要任何解释、不要 markdown 围栏。每个元素形如 "
        "{\"type\":\"preference\",\"content\":\"亮哥偏好详细讲解\",\"confidence\":0.9,\"source\":\"回答要详细点\",\"topic\":\"回答风格\"}。"
        "没有任何值得记的就输出 []。\n\n"
        "示例：\n"
        "用户：我是做后端的，最近在看 AI → [{\"type\":\"fact\",\"content\":\"亮哥后端出身\",\"confidence\":0.9,\"source\":\"我是做后地的\",\"topic\":\"技术背景\"},{\"type\":\"fact\",\"content\":\"亮哥正在学习 AI 方向\",\"confidence\":0.85,\"source\":\"最近在看 AI\",\"topic\":\"当前学习方向\"}]\n"
        "用户：回答能不能详细点，多举例子 → [{\"type\":\"preference\",\"content\":\"亮哥偏好详细讲解并多举例\",\"confidence\":0.95,\"source\":\"回答能不能详细点，多举例子\",\"topic\":\"回答风格\"}]\n"
        "用户：我朋友喜欢简短回答 → []  （主体不是亮哥本人）\n"
        "用户：我的密码是 abc123 → []  （敏感信息，绝不记）\n"
        "用户：公司年假有几天？ → []  （这是提问，不是偏好或事实）\n"
        "用户：帮我写首关于秋天的诗 → []  （闲聊创作，无长期价值）"
    )


def _parse_memories(raw: str) -> list:
    """健壮解析记忆官输出的 JSON 数组（照抄 _parse_subtasks 的容错套路）。
    取第一个 '[' 到最后一个 ']' 再 json.loads；逐个校验清洗；全解析不出来返回 []。"""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        mtype = str(item.get("type", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        source = str(item.get("source", "")).strip()
        topic = str(item.get("topic", "")).strip()
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        # 清洗① 缺关键字段（非法 type / 空 content / 空 topic）→ 丢弃，无法安全入库
        if mtype not in _VALID_TYPES or not content or not topic:
            continue
        # 清洗② 低置信 → 丢弃（源头就卡死，不给污染回答的机会）
        if confidence < 0.6:
            continue
        # 清洗③ 敏感信息硬过滤（content 或 source 命中即整条丢，双保险之二）
        if _SENSITIVE_RE.search(content) or _SENSITIVE_RE.search(source):
            print(f"[记忆官][拦截] 疑似敏感记忆，已丢弃：topic={topic!r}")
            continue
        out.append({"type": mtype, "content": content,
                    "confidence": round(min(confidence, 1.0), 2),
                    "source": source, "topic": topic})
    return out


def extract_and_store(query: str, *, user: str = "liang") -> int:
    """记忆官主流程：拿用户这句话 → LLM 抽取 → 解析清洗 → 去重更新入库。返回本次写入/更新了几条。
    🔴 只对亮哥调用（游客不记，由调用方 main.py 保证）；全程 try/except 兜底，抽取失败绝不影响主回答。"""
    try:
        resp = chat(
            messages=[{"role": "system", "content": build_extract_system()},
                      {"role": "user", "content": query}],
            purpose="memory",
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # 抽取炸了就静默放弃这一轮（宁可不记，也绝不因记忆官拖累主流程）
        print(f"[记忆官] 抽取异常，本轮不记：{e}")
        return 0

    memories = _parse_memories(raw)
    for m in memories:
        upsert_memory(user=user, type=m["type"], content=m["content"],
                      confidence=m["confidence"], source=m["source"], topic=m["topic"])
    if memories:
        print(f"[记忆官] 写入/更新 {len(memories)} 条："
              f"{[(m['topic'], m['content']) for m in memories]}（原始={raw!r}）")
    return len(memories)