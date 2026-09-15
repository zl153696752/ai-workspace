# ===== 中文嵌入模型：BGE-small-zh-v1.5（ONNX 本地推理）=====
# 作用：把中文文本转成 512 维向量，供 Chroma 检索使用。
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

# 用 __file__ 推导模型路径，保证换机器/换盘符/进容器都能跑
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "bge-small-zh-v1.5")
MODEL_PATH = os.path.join(_MODEL_DIR, "model_quantized.onnx")
TOKENIZER_PATH = os.path.join(_MODEL_DIR, "tokenizer.json")


# 🔴 必须注册：Chroma 按名字把 EF 持久化进 collection，重开库时靠它按名字找回本类；
# 不注册的话 query() 会报 "Embedding function ... not found"。
@register_embedding_function
class BgeZhEF(EmbeddingFunction):
    """把一段中文文本变成 512 维单位向量。"""

    def __init__(self):
        # 显式指定 CPU：容器无 GPU，避免 onnxruntime 先探测 CUDA 再回退，白耗启动时间
        self.sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        self.tk = Tokenizer.from_file(TOKENIZER_PATH)
        # 模型上限 512 token，必须开截断，否则超长文本会让 sess.run 报维度不匹配
        self.tk.enable_truncation(max_length=512)
        # 有的 ONNX 导出不要 token_type_ids，先探测模型实际需要哪些输入
        self.in_names = set(i.name for i in self.sess.get_inputs())

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            e = self.tk.encode(text)
            # onnxruntime 只吃 numpy，input_ids 必须 int64；外层套 [] 补出 batch 维 → [1, seq]
            feed = {
                "input_ids": np.array([e.ids], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask], dtype=np.int64),
            }
            if "token_type_ids" in self.in_names:
                feed["token_type_ids"] = np.array([e.type_ids], dtype=np.int64)
            # 🔴 输出 shape 为 [batch, seq, 512]，[0][0][0] 取 [CLS] 那条向量（CLS pooling，BGE 官方指定）
            v = self.sess.run(None, feed)[0][0][0]
            # L2 归一化成单位向量：让欧氏距离与余弦相似度对应；max(...,1e-12) 防全零除零
            out.append((v / max(float(np.linalg.norm(v)), 1e-12)).tolist())
        return out

    # ===== 下面四个方法是 Chroma 的持久化协议，缺一个重开库就会报错 =====
    @staticmethod
    def name() -> str:
        # 这个名字会写进 collection 配置，定了就别改，否则旧库认不出会报 conflict
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
