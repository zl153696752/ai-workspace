# ===== 中文嵌入模型：BGE-small-zh-v1.5（ONNX 本地推理）=====
# 为什么要自己写这个文件，而不是继续用 Chroma 自带的那个：
#   Chroma 默认的 all-MiniLM-L6-v2 是英文模型，词表是 30522 个英文 wordpiece。
#   实测一份 300 字的中文切片，它有 167 个 token（66%）被替换成同一个 [UNK] 占位符，
#   向量区分度直接塌缩——这是「上传成功但检索不到」的第一层根因（完整复盘见 13.6.1 ②）。
#   BGE-small-zh-v1.5 用的是 21128 个字的中文 BERT 词表，同一份切片只有 1 个 [UNK]。
#
# 为什么不用 Chroma 内置的 34 个 EmbeddingFunction 里的任何一个（逐个构造实测过）：
#   SentenceTransformer / Text2Vec 要装 torch，镜像涨 800MB~2GB；Bm25 要装 fastembed；
#   ChromaBm25 虽然零依赖，但它的分词器就是 text.lower().split()，只按空格切，
#     中文整句变成一个 token，再被 token_max_length=40 一刀丢掉；
#   HuggingFace 那个是调远程 HTTP API，要 token 还要运行时联网，国内基本不通；
#   OpenAI 那个用不上——DeepSeek 官方没有 embedding 接口。
#   自己写这一个只用 onnxruntime + tokenizers + numpy，三个都已经在 requirements.txt 里 → 零新增依赖。
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
# register_embedding_function 为什么是必须的，见下面类定义上方那段注释
from chromadb.utils.embedding_functions import register_embedding_function

# 模型文件跟着代码一起进镜像（Dockerfile:29 的 COPY backend/app ./backend/app 已经覆盖）。
# 用 __file__ 推导路径而不是写死 "E:/ai-workspace/..."，和 config.py:37 同一个理由：
# 项目挪到任何机器、任何盘符、任何容器路径都能正常跑。
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "bge-small-zh-v1.5")
MODEL_PATH = os.path.join(_MODEL_DIR, "model_quantized.onnx")
TOKENIZER_PATH = os.path.join(_MODEL_DIR, "tokenizer.json")


# 🔴 这个装饰器不是可选的，是必须的。
# Chroma 会把「这张表用的是哪个 embedding 函数」按名字持久化进 collection 配置。
# 不注册的话，任何一次【没有显式传 embedding_function】的 get_or_create_collection，
# 建表时不报错，一直等到 query() 才炸：
#   ValueError: Embedding function bge_small_zh_v15_onnx not found.
#               Add @register_embedding_function decorator to the class definition.
# 注册之后 Chroma 能按名字自己把类找回来，这条路就通了（注册前后两种情况都实测过）。
# 它做的事极其简单：把类塞进一个全局字典 known_embedding_functions[name()]，
# 导入时执行一次，没有别的副作用。
@register_embedding_function
class BgeZhEF(EmbeddingFunction):
    """把一段中文文本变成 512 维单位向量。"""

    def __init__(self):
        # providers 显式写 CPU：创空间的容器没有 GPU，不写的话 onnxruntime 会先去找 CUDA 再退回来，白花启动时间
        self.sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        self.tk = Tokenizer.from_file(TOKENIZER_PATH)
        # 模型上限 512 token。不开截断的话，超长文本会让 sess.run 直接报维度不匹配。
        # 512 对我们 CHUNK_SIZE=300 的切片余量很足（实测 300 字中文 ≈ 261 token）。
        # 对比：Chroma 给 MiniLM 写死的是 256，300 字的切片只剩 4 个 token 余量，
        # CHUNK_SIZE 一旦调到 305 以上就会静默丢掉每片尾部，不报错、不告警。
        self.tk.enable_truncation(max_length=512)
        # 有的 ONNX 导出不要 token_type_ids，先问清楚模型到底要哪几个输入，缺什么补什么
        self.in_names = set(i.name for i in self.sess.get_inputs())

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            e = self.tk.encode(text)
            # onnxruntime 只吃 numpy 数组，且 input_ids 必须是 int64；
            # shape 要带 batch 维，所以外面套一层 [] 变成 [1, seq]
            feed = {
                "input_ids": np.array([e.ids], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask], dtype=np.int64),
            }
            if "token_type_ids" in self.in_names:
                feed["token_type_ids"] = np.array([e.type_ids], dtype=np.int64)
            # 输出 last_hidden_state 的 shape 是 [batch, seq, 512]。
            # 🔴 [0][0][0] 取的是第 0 个样本、第 0 个 token（也就是 [CLS]）那一条 512 维向量。
            #    这叫 CLS pooling，是 BGE 官方指定的取法。
            #    Chroma 自带的 MiniLM 用的是 mean pooling（对所有 token 做注意力加权平均）——
            #    两种取法不能混用，换模型时必须跟着换，否则向量的语义完全不是一回事。
            v = self.sess.run(None, feed)[0][0][0]
            # L2 归一化成单位向量：这样「平方欧氏距离」和「余弦相似度」一一对应，
            # 距离越小越相似，rag.py 里那个 dist < 1.1 的阈值才有稳定含义。
            # max(..., 1e-12) 是防全零向量除零，Chroma 自带的实现也是这么兜的。
            out.append((v / max(float(np.linalg.norm(v)), 1e-12)).tolist())
        return out

    # ===== 下面四个方法是 Chroma 的「持久化协议」，缺一个就会在重开库时报错 =====
    @staticmethod
    def name() -> str:
        # 这个名字会被写进 collection 配置。一旦定了就别改：
        # 改了以后旧库认不出新名字，会报 Embedding function conflict
        return "bge_small_zh_v15_onnx"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        # Chroma 从磁盘读回配置时，靠这个方法按名字把实例重建出来
        return BgeZhEF()

    @staticmethod
    def validate_config(config):
        return None
