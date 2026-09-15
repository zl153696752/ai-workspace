# ===== 知识库检索服务（RAG 的底层能力层）=====
# RAG = 检索增强生成：先从企业文档检索出相关原文，再塞进提示词让模型基于资料作答，避免瞎编且能标注出处。
# 本文件集中三个工具函数：向量检索、文字提取、长文切片。HTTP 路由和 Agent 工具都复用它们。
import io                    # 提供内存中的“假文件”对象 BytesIO
from pypdf import PdfReader  # 解析 PDF、提取文字

# 相对导入：app 是一个包，包内互相引用要用 . 开头，写成 from config import ... 会报 ModuleNotFoundError
from .config import collection, CHUNK_SIZE, CHUNK_OVERLAP


def search_knowledge_base(query: str) -> list:
    """在知识库做语义检索，返回与 query 最相关的切片（RAG 的检索核心）。

    参数 query：检索语句，最好是贴近文档措辞的完整句子（向量检索对措辞敏感）。
    返回：[(切片正文, 元数据), ...]，最多 3 条按相关度降序；元数据含 filename 供前端引用卡片用。
          库空或都不够相关时返回 []，调用方据此走“如实告知没有资料”分支。
    """
    # 库空时提前返回：对空集合做 query，Chroma 会抛异常
    if collection.count() == 0:
        return []
    # collection.query：把文本转向量（Chroma 内部自动完成）再找最近的 n_results 条
    results = collection.query(
        query_texts=[query],   # 传文本即可，参数是列表（支持一次查多个，本项目只查一个）
        n_results=3,           # 取最相近 3 条：多了塞无关内容干扰模型，少了可能漏关键信息
        include=["documents", "distances", "metadatas"]  # 返回切片正文、距离（越小越相似）、元数据
    )
    # 因查询传的是列表，每个字段都是二层列表，取 [0] 才是第一个问题的结果；zip 按位置配对三字段。
    # 🔴 距离阈值 1.1：collection 用默认 l2（平方欧氏距离）空间，对归一化向量有 l2²=2×余弦距离，
    #    数值是网上余弦经验值的两倍，别直接套。换嵌入模型必须重新标定此阈值，否则会误杀相关切片。
    # 列表推导式一次完成：配对字段 + 过滤空白内容 + 过滤超阈值的不相关切片
    return [
        (doc, meta) for doc, dist, meta in zip(
            results["documents"][0], results["distances"][0], results["metadatas"][0])
        if doc and doc.strip() and dist < 1.1   # 三条件全满足才留：doc 非空、去空白后有字、距离 < 阈值
    ]


def extract_text(content: bytes, ext: str) -> str:
    """把上传文件的原始字节转成纯文字（向量库只能存文字，存不了二进制）。

    参数 content：文件二进制内容（来自内存，不依赖磁盘）；ext：后缀，决定解析方式。
    返回：纯文字字符串；提不出内容时返回空串或纯空白串，由调用方判断并拒收。
    """
    if ext == ".pdf":
        # PDF 是二进制，需用专门库解析。BytesIO 把内存字节包成“文件对象”，
        # 因为 PdfReader 只接受路径或文件对象、不接受裸字节（上传是先解析后落盘，磁盘上还没这文件）。
        reader = PdfReader(io.BytesIO(content))
        # 逐页提取再拼接；extract_text() 对空白/纯图片页可能返回 None，用 or "" 兜底避免 join 报 TypeError
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    # .txt / .md 是纯文本，直接按 UTF-8 解码；errors="ignore" 跳过非法字节（如 GBK 文件），宁可少几字也不让接口崩
    return content.decode("utf-8", errors="ignore")


def split_text(text: str) -> list:
    """把长文切成若干小片供向量化入库（每片一条向量，检索才能定位到具体段落）。

    切法是“滑动窗口”：每片最多 CHUNK_SIZE 字，相邻片重叠 CHUNK_OVERLAP 字。
    以 300/50 为例：第1片 0~300、第2片 250~550……每轮起点净进 250，必然走到文末不死循环。
    参数 text：全文纯文字；返回：切片字符串列表。
    """
    # 先把换行压成空格：避免切片边界落在标题/列表的排版结构上把内容拆成表达不完整的两半
    text = text.replace("\n", " ")
    chunks = []
    start = 0                        # 当前切片起始字符位置
    while start < len(text):         # 起点走到文末就结束
        end = start + CHUNK_SIZE     # 右边界（text[a:b] 不含 b，正好取 CHUNK_SIZE 字）
        chunk = text[start:end]      # 到文末时 end 超出总长，Python 自动截到末尾不报错
        if chunk.strip():            # 跳过纯空白片：转向量没意义还白占存储
            chunks.append(chunk)
        # 下一片起点回退 CHUNK_OVERLAP 字（这就是“重叠”）：保证任一句不会因硬切丢失完整表达
        start = end - CHUNK_OVERLAP
    return chunks
