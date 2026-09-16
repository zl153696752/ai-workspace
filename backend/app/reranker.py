# ===== 重排模型：bge-reranker-base（ONNX 本地推理，Cross-Encoder）=====
# 作用：给「查询-候选切片」逐对打相关性分，用于混合检索后的精排（替代旧的 dist<1.1 距离硬阈值）。
# 与嵌入模型的区别（Bi-Encoder vs Cross-Encoder）：
#   嵌入模型把 query、doc 各自转向量再比距离，快但粗；Reranker 把两者拼一起送进模型做全交叉注意力，
#   输出一个相关性 logit（越大越相关），准但慢——所以只对粗召回后的少量候选做，形成「先粗排再精排」。
#
# 🔴 架构差异（和 embeddings_bge.py 的 BGE 中文模型不同，别照搬）：
#   bge-reranker-base 基于 XLM-RoBERTa（XLMRobertaForSequenceClassification）：
#   ① 输入没有 token_type_ids；② 输出是单个 logit（num_labels=1），不是一个向量。
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# 用 __file__ 推导模型路径，换机器/换盘符/进容器都能跑（与 embeddings_bge.py 一致）
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "bge-reranker-base")
MODEL_PATH = os.path.join(_MODEL_DIR, "model_quantized.onnx")
TOKENIZER_PATH = os.path.join(_MODEL_DIR, "tokenizer.json")

_MAX_LEN = 512                 # 模型上限 512 token，query+passage 拼接可能超长，必须截断
_MIN_MODEL_BYTES = 1024 * 1024  # 小于 1MB 视为「没下全/下成 LFS 指针」，判定不可用（正常 279MB 远大于此）


def _model_files_ok() -> bool:
    """模型资产是否就位且不是损坏的指针文件。缺失/过小时返回 False，触发降级。"""
    if not (os.path.isfile(MODEL_PATH) and os.path.isfile(TOKENIZER_PATH)):
        return False
    # 先按大小拦一道：指针文件只有几百字节，直接拿去 InferenceSession 会报难懂的错
    return os.path.getsize(MODEL_PATH) >= _MIN_MODEL_BYTES


class _Reranker:
    """bge-reranker-base 的 ONNX 推理封装（启动加载一次 + 缺失时优雅降级）。"""

    def __init__(self):
        self.available = False          # 模型是否可用；不可用时 rerank 走降级分支
        self.sess = None
        self.tk = None
        self.in_names = set()
        if not _model_files_ok():
            print(f"[Reranker] 未找到可用模型（{MODEL_PATH}），精排降级为原序返回。"
                  f"下载方式见 docs/企业级改造方案.md 步骤 1。")
            return
        try:
            # 显式 CPU：容器无 GPU，避免 onnxruntime 先探测 CUDA 再回退白耗启动时间
            self.sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
            self.tk = Tokenizer.from_file(TOKENIZER_PATH)
            self.tk.enable_truncation(max_length=_MAX_LEN)
            # 探测模型实际需要哪些输入：XLM-R 只要 input_ids + attention_mask，不含 token_type_ids
            self.in_names = set(i.name for i in self.sess.get_inputs())
            self.available = True
            print("[Reranker] bge-reranker-base 已加载，精排可用。")
        except Exception as e:
            # 加载失败不拖垮服务：记一笔并降级（available 保持 False）
            print(f"[Reranker] 加载失败，精排降级为原序返回：{e}")
            self.available = False


# 模块级单例：模型只在首次导入时加载一次，后续请求复用（与 MCP 工具缓存同理）
_reranker = _Reranker()


def score_pair(query: str, passage: str) -> float:
    """给单个 (query, passage) 对打相关性分。模型不可用时返回 0.0（对所有对一视同仁，等于不改变原序）。"""
    if not _reranker.available:
        return 0.0
    # tk.encode(a, b) 按 XLM-R 模板拼成 <s> query </s></s> passage </s>，这就是 Cross-Encoder 的输入
    e = _reranker.tk.encode(query, passage)
    feed = {
        "input_ids": np.array([e.ids], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask], dtype=np.int64),
    }
    if "token_type_ids" in _reranker.in_names:   # 防御：万一导出模型带这个输入才喂
        feed["token_type_ids"] = np.array([e.type_ids], dtype=np.int64)
    # 输出 logits 形状 [batch=1, num_labels=1]，取那个标量；reshape(-1)[0] 对形状变化更稳
    out = _reranker.sess.run(None, feed)[0]
    return float(np.asarray(out).reshape(-1)[0])


def is_available() -> bool:
    """精排模型是否可用。不可用时上层（Self-CRAG）应跳过「按分数判相关」改走降级，别拿 0.0 当真实分数用。"""
    return _reranker.available


def rerank_with_scores(query: str, candidates: list, top_k: int = 4) -> list:
    """对候选精排，返回 [(相关性分数, 正文, 元数据), ...]（按分数降序，最多 top_k 条）。

    与 rerank 的区别：保留每片的 cross-encoder 相关性 logit，供 Self-CRAG 逐片判定相关与否。
    分数含义：bge-reranker-base 输出单个 logit，>0≈相关、<0≈不相关（越大越相关）。
    模型不可用时：分数统一记 0.0、按原序返回前 top_k（降级，绝不报错、绝不返回空）。
    """
    if not candidates:
        return []
    if not _reranker.available:
        return [(0.0, doc, meta) for doc, meta in candidates[:top_k]]
    scored = [(score_pair(query, doc), doc, meta) for doc, meta in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)   # 分数越大越相关，降序
    return scored[:top_k]


def rerank(query: str, candidates: list, top_k: int = 4) -> list:
    """对候选切片按相关性精排，返回 [(正文, 元数据), ...]（丢弃分数版，向后兼容旧调用与文件末尾自测）。"""
    return [(doc, meta) for _score, doc, meta in rerank_with_scores(query, candidates, top_k)]


# ===== 单独验证打分能力（步骤 1 自测：直接运行本文件即可，不依赖服务）=====
if __name__ == "__main__":
    q = "公司加班餐补的发放标准是多少？"
    passages = [
        ("工作日加班晚于 19 点，每餐补贴 30 元。", {"filename": "公司制度.txt"}),   # 相关，应最高分
        ("入职满一年享 5 天年假，满五年享 10 天。", {"filename": "公司制度.txt"}),   # 不相关（年假）
        ("今天天气不错，适合出去散步。", {"filename": "闲聊.txt"}),                  # 完全不相关
    ]
    print(f"查询：{q}")
    print(f"Reranker 可用：{_reranker.available}")
    for doc, _meta in passages:
        print(f"  分数 {score_pair(q, doc):+.4f}  ← {doc}")
    print("精排结果（top 2）：")
    for doc, _meta in rerank(q, passages, top_k=2):
        print(f"  {doc}")