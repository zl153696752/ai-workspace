# ===== 知识库检索服务（RAG 的底层能力层）=====
# RAG = Retrieval-Augmented Generation（检索增强生成），是本项目的主干技术：
#   先从企业文档里“检索”出相关原文，再把原文塞进提示词，让模型“基于资料生成”回答。
#   好处：模型不依赖自己的训练记忆（它不可能知道你公司的制度），因此不会乱编，还能标注出处供用户核对。
#
# 所有“跟文档打交道”的底层能力集中在本文件：向量检索、文字提取、长文切片。
# 这三个都是普通的工具函数：只依赖 config.py 里的 collection 和切片参数，不关心调用者是谁——
# HTTP 路由（上传接口）调它们，Agent 工具（模型决定要查知识库时）也调它们，能力只写一份。
import io                    # 标准库：提供内存中的“假文件”对象（BytesIO）
from pypdf import PdfReader  # 解析 PDF、提取其中文字的第三方库

# 相对导入（开头的点表示“同一个包内”）：从隔壁的 config.py 拿向量表对象和两个切片参数。
# 后端是用 uvicorn app.main:app 启动的，app 是一个包，所以包内模块互相引用要用 . 开头，
# 写成 from config import ... 会报 ModuleNotFoundError。
from .config import collection, CHUNK_SIZE, CHUNK_OVERLAP


def search_knowledge_base(query: str) -> list:
    """在知识库里做语义检索，返回与 query 最相关的文档切片。这是整个 RAG 系统的检索核心。

    参数 query：检索语句。最好是贴近文档措辞的完整句子（如“餐补发放标准是什么”），
                而不是关键词堆砌或带指代的短句（如“那它呢”）——向量检索对措辞敏感。
    返回：[(切片正文, 元数据字典), ...]，最多 3 条，按相关度从高到低。
          元数据里存着 filename（原文件名）等信息，前端的引用卡片要用。
          库里没数据、或检索到的都不够相关时，返回空列表 []
          （调用方据此走“如实告知没有资料”的分支，而不是让模型硬答）。
    """
    # 库是空的就提前返回：对空集合做 query，Chroma 会直接抛异常，这里先挡一道
    if collection.count() == 0:
        return []
    # collection.query = 把 query 文本转成向量，再到库里找距离最近的 n_results 条切片。
    # 注意：文本转向量（embedding）是 Chroma 内部自动完成的，我们只管传文本，不用自己调向量模型。
    results = collection.query(
        query_texts=[query],   # 传文本而不是向量；参数是列表，支持一次查多个问题（本项目只查一个）
        n_results=3,           # 取最相近的 3 条：多了会塞进无关内容干扰模型，少了可能漏掉关键信息
        include=["documents", "distances", "metadatas"]  # 指定要返回哪些字段：切片正文、距离（越小越相似）、元数据
    )
    # 返回结构说明：因为查询传的是列表（可以一次查多个问题），所以每个字段都是二层列表：
    #   results["documents"] = [[第1个问题的命中切片...]]，取 [0] 才是第一个问题的结果。
    # zip(...) 把三个字段按位置配对：第1片的正文 + 第1片的距离 + 第1片的元数据 组成一组。
    #
    # 距离阈值 1.1 是第二道防线（第一道是模型自己判断“这问题该不该查库”）：
    #   Chroma 默认的距离度量里，0 = 完全一致，数值越大越不相关。
    #   实测数据：相关问题的距离普遍在 1.1 以下，无关问题普遍在 1.3 以上，1.1 是两堆数据的分界线。
    #   有了它，即使模型决定要查库，捞回来的无关切片也不会被塞进提示词污染回答（双保险）。
    # 下面这个列表推导式一次完成三件事：配对字段、过滤空白内容、过滤超阈值的不相关切片
    return [
        (doc, meta) for doc, dist, meta in zip(
            results["documents"][0], results["distances"][0], results["metadatas"][0])
        if doc and doc.strip() and dist < 1.1   # doc 非空、去掉空白后还有字、且距离小于阈值，三个条件全满足才留下
    ]


def extract_text(content: bytes, ext: str) -> str:
    """把上传文件的原始字节内容转成纯文字（向量库只能存文字，存不了二进制）。

    参数 content：文件的二进制内容（bytes 类型），直接来自内存，不依赖磁盘上的文件
          ext：文件后缀（".txt" / ".md" / ".pdf"），决定用哪种解析方式
    返回：提取出的纯文字字符串（提不出内容时返回空串或只有空白的串，由调用方判断并拒收）
    """
    if ext == ".pdf":
        # PDF 是二进制格式，文字被封装在页面结构里，必须用专门的库解析。
        # io.BytesIO(content)：把内存里的字节包装成一个“像文件一样的对象”。
        # 为什么要多这一道：PdfReader 只接受文件路径或文件对象，不接受裸字节——
        # 包装后就不用先把文件写到磁盘再读一遍（上传流程是“先解析后落盘”，此时磁盘上还没这个文件）。
        reader = PdfReader(io.BytesIO(content))
        # 逐页提取文字再用换行拼接。page.extract_text() 对某些页面（如空白页、纯图片页）可能返回 None，
        # 用 or "" 兜底成空串，否则 join 遇到 None 会直接报 TypeError。
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    # .txt / .md 本身就是纯文本，直接按 UTF-8 解码即可。
    # errors="ignore"：遇到不符合 UTF-8 规则的字节（比如文件其实是 GBK 编码）直接跳过，
    # 而不是抛异常让整个上传失败——宁可少几个乱码字符，也要保证接口不崩。
    return content.decode("utf-8", errors="ignore")


def split_text(text: str) -> list:
    """把长文切成若干小片，供向量化入库（每片存一条向量，检索时才能定位到具体段落）。

    切法是“滑动窗口”：每片最多 CHUNK_SIZE 字，相邻两片重叠 CHUNK_OVERLAP 字。
    以默认值 300 / 50 为例：
        第 1 片 = 第 0~300 字，第 2 片 = 第 250~550 字，第 3 片 = 第 500~800 字……
        （每轮起点净前进 250 字，所以循环必然能走到文末结束，不会死循环）
    参数 text：已提取出的全文纯文字
    返回：切片字符串组成的列表，每一片都会被单独转成向量存进 Chroma
    """
    # 先把换行符全部压成空格：文档里的标题、列表靠换行排版，
    # 如果保留换行，切片边界正好落在排版结构上时会把一个列表拆成两半，两半都表达不完整，
    # 检索质量下降。压成空格后文字连成一片，切在哪儿都不影响语义完整性。
    text = text.replace("\n", " ")
    chunks = []
    start = 0                        # 当前切片的起始字符位置
    while start < len(text):         # 起点走到文末就结束
        end = start + CHUNK_SIZE     # 切片右边界（Python 的 text[a:b] 不含 b，正好取 CHUNK_SIZE 个字）
        chunk = text[start:end]      # 取出这一片；到文末时 end 会超出总长度，Python 自动截到末尾，不报错
        if chunk.strip():            # 跳过纯空白片段（比如文件末尾一堆空格）：空白片转向量没有意义，还白占存储
            chunks.append(chunk)
        # 下一片的起点不是 end，而是往回退 CHUNK_OVERLAP 个字——这就是“重叠”的实现方式。
        # 后退保证上一片末尾那 50 个字会在下一片开头再出现一次，任何一句话都不会因为硬切而丢失完整表达。
        start = end - CHUNK_OVERLAP
    return chunks
