# ===== 模型调用统一收口层（步骤9 阶段A）=====
# 为什么要这层：对大模型的调用原本散在多处（supervisor 意图分类、两次查询改写、合成、工具 worker），
# 各写各的、没有统一的重试/超时/降级，也没地方统一记 token/成本/延迟。
# 收口 = 把所有调用赶进这一个模块，在一处统一做「韧性（重试/超时/降级）+ 可观测性第一档（埋点）」。
#
# 🔴 双轨设计（本文件先实现轨①，轨② TokenMeter 在阶段 A-2 补进来）：
#   轨① 裸 OpenAI SDK 的 client（supervisor / 改写×2）→ 走本模块 chat()，手动埋点；
#   轨② LangChain 的 lc_llm（合成 / 工具 worker ReAct）→ 挂 callback 自动埋点。
import time

# tenacity：声明式重试库。只对「瞬时故障」重试，退避策略可配。
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# OpenAI SDK 的异常类型：只重试超时/连接/限流/5xx 这类「重试可能好」的；
# 参数错、认证错这类重试也没用，不在名单里 → 直接抛，不浪费三次重试。
from openai import APITimeoutError, APIConnectionError, RateLimitError, InternalServerError

from .config import client   # 复用全局裸 client（已连好 DeepSeek），不新建连接

import contextvars

# 请求级 span 收集器：一次问答的所有模型调用 span 都归到这里。
# 用 contextvars 而非全局变量——异步/多线程下按「当前请求」隔离，并发不串。
_current_spans = contextvars.ContextVar("llm_spans", default=None)


def start_trace():
    """一次问答开始时调用：开一个空收集器。"""
    _current_spans.set([])


def get_trace_spans() -> list:
    """一次问答结束时调用：取出本次收集到的全部 span。"""
    return _current_spans.get() or []

# ===== DeepSeek 单价（元 / 百万 token）：成本 = token 数 × 单价 =====
# 🔴 下面是示例值，动手前按 deepseek-v4-flash 官方实际单价改这两个数——改这一处，全项目成本都跟着对。
PRICE_IN = 2.0     # 输入（prompt）单价
PRICE_OUT = 8.0    # 输出（completion）单价


def _cost(usage) -> float:
    """把 token 用量换算成成本（元）。usage 是裸 SDK 的 resp.usage 对象。"""
    if not usage:
        return 0.0
    return (usage.prompt_tokens * PRICE_IN + usage.completion_tokens * PRICE_OUT) / 1_000_000


def _record(span: dict):
    print(f"[LLM埋点] {span}")        # 开发期观察，保留
    sink = _current_spans.get()
    if sink is not None:              # 在请求上下文里才归集（启动暖机等无收集器时跳过）
        sink.append(span)


# 只对瞬时故障重试；最多3次；指数退避 1s→2s→4s（封顶8s）；reraise=True 表示耗尽后抛【原异常】，
# 交调用方走各自兜底（supervisor 回落 kb、改写退回原句），而不是被 tenacity 包成 RetryError。
@retry(
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _call_with_retry(**kwargs):
    # timeout=30：单次最多等30秒，防模型卡死把整个请求拖挂（超时本身也是一种降级触发）。
    return client.chat.completions.create(timeout=30, **kwargs)


def chat(*, messages, purpose, temperature=None, model="deepseek-v4-flash"):
    """【轨① 唯一入口】所有裸 client 的模型调用都走这里。

    messages    —— OpenAI 格式消息数组（和原来 client.chat.completions.create 的 messages 完全一样）
    purpose     —— 这次调用是干嘛的（"supervisor"/"rewrite"/...），埋点靠它区分环节
    temperature —— 可选；分类传 0（稳定），改写不传走默认
    返回：原始 response 对象——调用方照旧用 resp.choices[0].message.content，改动最小。
    """
    t0 = time.time()
    try:
        resp = _call_with_retry(
            model=model,
            messages=messages,
            **({"temperature": temperature} if temperature is not None else {}),
        )
        usage = getattr(resp, "usage", None)
        _record({"purpose": purpose, "latency": round(time.time() - t0, 3),
                 "prompt_tokens": getattr(usage, "prompt_tokens", None),
                 "completion_tokens": getattr(usage, "completion_tokens", None),
                 "cost": round(_cost(usage), 6)})
        return resp
    except Exception as e:
        # 降级留痕：彻底失败也记一笔再抛，绝不静默吞（各调用方的 except 兜底逻辑保持不动）
        _record({"purpose": purpose, "degraded": True, "reason": str(e),
                 "latency": round(time.time() - t0, 3)})
        raise


# ===== 轨②：LangChain lc_llm 的回调计量器（A-2）=====
# 为什么需要它：合成和工具 worker 走的是 ChatOpenAI（工具 worker 还被 create_react_agent 包在里面反复调），
# 调用点在框架内部、一次问答可能调好几次，没法像裸 client 那样在每个调用点手动记。
# callback 就是「总机计费器」——不管内部调几次，每次都触发 on_chat_model_start / on_llm_end。
from langchain_core.callbacks import BaseCallbackHandler


class TokenMeter(BaseCallbackHandler):
    def __init__(self, purpose="lc"):
        self.purpose = purpose
        self._t0 = None

    # ChatOpenAI 是 chat model，触发的是 on_chat_model_start；on_llm_start 一并留着兜底
    def on_chat_model_start(self, serialized, messages, **kw):
        self._t0 = time.time()

    def on_llm_start(self, serialized, prompts, **kw):
        self._t0 = time.time()

    def on_llm_end(self, response, **kw):
        latency = round(time.time() - self._t0, 3) if self._t0 else None
        usage = self._extract_usage(response)
        _record({"purpose": self.purpose, "latency": latency,
                 "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                 "completion_tokens": usage.get("completion_tokens") if usage else None,
                 "cost": round(self._cost_dict(usage), 6)})
        self._t0 = None

    def on_llm_error(self, error, **kw):
        _record({"purpose": self.purpose, "degraded": True, "reason": str(error)})
        self._t0 = None

    @staticmethod
    def _extract_usage(response):
        """从 LLMResult 多路抠 token 用量（流式与否、版本不同，放的位置不一样，逐个试）。"""
        llm_output = getattr(response, "llm_output", None) or {}
        tu = llm_output.get("token_usage") or llm_output.get("usage")   # 路1：非流式常见位置
        if tu:
            return {"prompt_tokens": tu.get("prompt_tokens"),
                    "completion_tokens": tu.get("completion_tokens")}
        try:                                                            # 路2：较新版/流式，挂在 message.usage_metadata
            meta = getattr(response.generations[0][0].message, "usage_metadata", None)
            if meta:
                return {"prompt_tokens": meta.get("input_tokens"),
                        "completion_tokens": meta.get("output_tokens")}
        except Exception:
            pass
        return None

    @staticmethod
    def _cost_dict(usage) -> float:
        if not usage:
            return 0.0
        return (usage.get("prompt_tokens", 0) * PRICE_IN
                + usage.get("completion_tokens", 0) * PRICE_OUT) / 1_000_000