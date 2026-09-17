# ===== 知识库检索服务（RAG 的底层能力层）=====
# RAG = 检索增强生成：先从企业文档检索出相关原文，再塞进提示词让模型基于资料作答，避免瞎编且能标注出处。
# 本文件集中三个工具函数：向量检索、文字提取、长文切片。HTTP 路由和 Agent 工具都复用它们。
import io                    # 提供内存中的"假文件"对象 BytesIO
import re                    # 结构化切片要用正则识别标题、按句切分（2a 加的，保留）
import jieba                 # 中文分词：BM25 是词袋模型，中文必须先切成词
from pypdf import PdfReader  # 解析 PDF、提取文字
from rank_bm25 import BM25Okapi  # BM25 关键词打分（经典 Okapi 变体）

# 相对导入：app 是一个包，包内互相引用要用 . 开头，写成 from config import ... 会报 ModuleNotFoundError
from .config import collection, CHUNK_SIZE, CHUNK_OVERLAP
from .reranker import rerank_with_scores, is_available  # 精排（带分数版供质检判定）+ 可用性探测（缺失时降级）


# ===== 混合检索参数（2b）=====
RECALL_N = 10   # 每路召回条数：宽松多取，给融合和精排留足候选（旧的只取 3 太窄，相关的常被丢在第 4、5 名）
FUSE_N   = 8    # RRF 融合后送入精排的候选数
TOP_K    = 3    # 精排后最终返回条数（对齐旧版 top-3，前端引用卡片数量不变）
_RRF_K   = 60   # RRF 平滑常数，经验默认值：越大越弱化头部名次的碾压，60 是通用取值
# ===== Self-CRAG 质检参数（步骤3）=====
# 相关性阈值：精排 cross-encoder 的 logit，>0≈相关、<0≈不相关（bge-reranker-base 的语义）。
# 逐片判定：分数 ≥ 阈值的片才算「相关资料」保留，低于阈值的片丢弃（引用卡片因此更干净）。
# 🔴 0.0 是原理性默认值（logit 过 0 = sigmoid 概率过 0.5），真实最优阈值需步骤10 用评估集标定，别拍脑袋改。
_CRAG_SCORE_THRESHOLD = 0.0


def _vector_recall(query: str, n: int) -> list:
    """向量召回：Chroma 语义检索，返回按距离升序的 [(id, 正文, 元数据), ...]。

    只负责「宽松多召回 + 排名」，不再用距离硬阈值砍（相关与否交给后面 RRF + 精排判定）。
    任何异常都吞掉返回 []，让检索退化为「只有关键词那一路」，不拖垮整个查询。
    """
    try:
        res = collection.query(
            query_texts=[query],
            n_results=min(n, collection.count()),   # 库里不足 n 条时取实际条数，避免 Chroma 告警
            include=["documents", "metadatas"]
        )
        return [
            (cid, doc, meta)
            for cid, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
            if doc and doc.strip()
        ]
    except Exception as e:
        print(f"[检索] 向量召回失败，本路降级为空：{e}")
        return []


# BM25 索引缓存：全量重建成本高（要拉全库 + 逐片分词），用「切片总数」当版本戳缓存，
# 只有增删文档导致总数变化时才重建。容器场景 seed 在启动时灌库、运行中极少变，命中率高。
_bm25_cache = {"count": -1, "index": None, "rows": []}


def _get_bm25_index():
    """返回 (BM25 索引, [(id, 正文, 元数据), ...])；库空或构建失败返回 (None, [])。"""
    try:
        total = collection.count()
    except Exception as e:
        print(f"[检索] 读取切片总数失败，关键词召回降级：{e}")
        return None, []
    if total == 0:
        return None, []
    if _bm25_cache["count"] == total and _bm25_cache["index"] is not None:
        return _bm25_cache["index"], _bm25_cache["rows"]   # 总数没变，复用缓存
    try:
        got = collection.get(include=["documents", "metadatas"])   # 小库全量拉取；大库需换专用检索服务
        rows = [(cid, doc, meta) for cid, doc, meta
                in zip(got["ids"], got["documents"], got["metadatas"]) if doc and doc.strip()]
        tokenized = [jieba.lcut(doc) for _cid, doc, _meta in rows]  # BM25 吃「分好词的 token 列表」
        _bm25_cache.update(count=total, index=BM25Okapi(tokenized), rows=rows)
        return _bm25_cache["index"], rows
    except Exception as e:
        print(f"[检索] BM25 索引构建失败，关键词召回降级：{e}")
        return None, []


def _keyword_recall(query: str, n: int) -> list:
    """关键词召回：jieba 把 query 分词，用 BM25 对全库打分，取 top-n 的 [(id, 正文, 元数据), ...]。"""
    index, rows = _get_bm25_index()
    if index is None:
        return []
    try:
        scores = index.get_scores(jieba.lcut(query))
        order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:n]  # 分数降序取前 n 的下标
        return [rows[i] for i in order if scores[i] > 0]   # 分数 0 = 一个关键词都没命中，不召回
    except Exception as e:
        print(f"[检索] 关键词召回失败，本路降级为空：{e}")
        return []


def _rrf_fuse(vec_hits: list, kw_hits: list, k: int, top_n: int) -> list:
    """RRF（倒数排名融合）：把向量、关键词两份「名次」合成一份，返回 [(正文, 元数据), ...]。

    为什么用名次不用分数：向量给的是距离、BM25 给的关键词分，量纲相反又不可比，直接相加没意义；
    名次天然可比（第 1 名就是各自最相关的），故 RRF 只看名次、免归一化、免调权重。
    公式 score(d)=Σ 1/(k+rank)：同一片被两路都召回 → 两项相加 → 共识者自然浮到最前，去重与融合一步到位。
    """
    fused = {}   # id -> [rrf分, 正文, 元数据]
    for hits in (vec_hits, kw_hits):
        for rank, (cid, doc, meta) in enumerate(hits, start=1):   # rank 从 1 开始
            if cid in fused:
                fused[cid][0] += 1.0 / (k + rank)   # 已在另一路出现 → 累加（这就是「共识加分」）
            else:
                fused[cid] = [1.0 / (k + rank), doc, meta]
    ordered = sorted(fused.values(), key=lambda x: x[0], reverse=True)   # 按 RRF 分降序
    return [(doc, meta) for _score, doc, meta in ordered[:top_n]]


def _filter_private(hits: list, allow_private: bool) -> list:
    """身份过滤：allow_private=False（游客）时剔除标了 private 的切片，True（亮哥）原样放行。

    🔴 放在【召回层】过滤（不是精排后）：这样游客拿到的是「最好的公共结果」，
    不会出现「私人片先占了 top-k 名额、再被丢弃，导致公共结果变少」的降级。
    meta.get("private", False)：老数据 / seed 底料没有 private 字段，默认当公共——它们本就该对所有人可见。
    """
    if allow_private:
        return hits
    return [(cid, doc, meta) for cid, doc, meta in hits if not meta.get("private", False)]


def _retrieve_once(query: str, allow_private: bool = False) -> list:
    """跑一遍完整混合检索：向量召回 + BM25 召回 → RRF 融合 → 精排，返回 [(相关性分数, 正文, 元数据), ...]。

    这是「单次检索」的原子操作。步骤5 的检索质检 worker 判定不相关后，会用重写过的查询再调它一次（有界重查）。
    精排保留每片分数，正是为了让上层拿分数当 Self-CRAG 的「质检员」。
    """
    # 两路各自独立召回（各自内部已 try/except，一路炸了另一路照常）
    vec_hits = _filter_private(_vector_recall(query, RECALL_N), allow_private)
    kw_hits = _filter_private(_keyword_recall(query, RECALL_N), allow_private)
    # RRF 融合去重，取前 FUSE_N 条作为精排候选
    candidates = _rrf_fuse(vec_hits, kw_hits, _RRF_K, FUSE_N)
    if not candidates:
        return []
    # 精排收窄到 TOP_K，并保留每片相关性分数；reranker 缺失时它内部降级为「按融合原序、分数记 0.0」
    return rerank_with_scores(query, candidates, top_k=TOP_K)


def _grade_and_filter(scored_hits: list) -> tuple:
    """Self-CRAG 的「检索质检员」：用精排分数给结果定级、并逐片过滤掉不相关的。

    返回 (保留的 [(正文, 元数据), ...], 级别)，级别 ∈ {"correct", "incorrect", "unavailable"}：
      correct     —— 至少一片分数 ≥ 阈值，资料可用（低分片已被过滤，引用卡片更干净）；
      incorrect   —— 全部低于阈值，判定「没检索到相关资料」；
      unavailable —— 精排模型缺失、拿不到真实分数，无法判定 → 信任检索原序（降级，不过滤）。
    巧思：复用精排的 cross-encoder 分数当质检员，不必再叫一个模型判相关性（省延迟省成本，也避免「核查模型自己也会错」）。
    """
    if not scored_hits:
        return [], "incorrect"          # 一条都没召回，等同「不相关」
    if not is_available():
        # 降级：没有可信分数（全 0.0），阈值过滤会把所有片误杀，故原样放行、交给后续提示词规则兜底
        return [(doc, meta) for _score, doc, meta in scored_hits], "unavailable"
    kept = [(doc, meta) for score, doc, meta in scored_hits if score >= _CRAG_SCORE_THRESHOLD]
    return kept, ("correct" if kept else "incorrect")


def search_knowledge_base(query: str, allow_private: bool = False) -> list:
    """混合检索 + Self-CRAG 质检，返回 [(正文, 元数据), ...]。

    流程：单次检索（召回→融合→精排）→ 用精排分数逐片质检 →
      ① correct / unavailable：返回过滤后的资料（模型缺失时降级信任原序）；
      ② incorrect：返回 []（如实「无资料」，调用方据此走「告知没有资料」分支）。
    「不相关时有界重写重查一次」不在这里做，留给步骤5 的检索质检 worker（那里有意图路由，
    只对真·知识库查询触发重写重查，避免拖累闲聊）——见改造方案步骤3/5。

    返回结构与旧版一致，上游（main.py / agents.py / mcp_server.py）无需改动；
    检索层任何异常都在内部降级为 []，绝不把堆栈抛给用户。
    """
    try:
        if collection.count() == 0:
            return []   # 库空：对空集合做 query，Chroma 会抛异常，提前拦掉
        kept, grade = _grade_and_filter(_retrieve_once(query, allow_private))
        if grade == "incorrect":
            print(f"[Self-CRAG] 检索结果全部低于相关性阈值，判定无资料：{query!r}")
            return []
        return kept   # correct：过滤后的相关资料；unavailable：降级信任原序
    except Exception as e:
        print(f"[检索] 混合检索异常，降级为「无资料」：{e}")
        return []


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


def _split_long_block(block: str) -> list:
    """把单个超长块按句子切成 ≤CHUNK_SIZE 的片，相邻片重叠 CHUNK_OVERLAP 字。
    优先在句末标点（。！？；）或换行处断句，实在没标点才退化为定长滑窗，
    保证再长的段落也不会一刀切在句子中间（这正是旧「定长硬切」最大的毛病）。
    """
    # (?<=...) 零宽断言：在句末标点「之后」切开，标点本身留在前一句末尾
    sentences = [s for s in (x.strip() for x in re.split(r"(?<=[。！？；\n])", block)) if s]
    pieces, cur = [], ""
    for s in sentences:
        if len(s) > CHUNK_SIZE:                      # 单句就超长（无标点长串）→ 定长滑窗兜底
            if cur.strip():
                pieces.append(cur.strip()); cur = ""
            start = 0
            while start < len(s):
                pieces.append(s[start:start + CHUNK_SIZE])
                start += CHUNK_SIZE - CHUNK_OVERLAP
            continue
        if cur and len(cur) + len(s) > CHUNK_SIZE:   # 攒够一片就结算
            pieces.append(cur.strip())
            cur = cur[-CHUNK_OVERLAP:] + s           # 重叠：带上上一片末尾，边界句不丢上下文
        else:
            cur += s
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def split_text(text: str) -> list:
    """结构化切片：按标题/段落切、尊重语义边界，每片携带结构元数据。

    与旧「压平换行 + 定长硬切」的区别：旧法把所有换行压成空格再每 300 字一刀，
    会把标题、列表、段落拦腰截断，切出的片语义不完整（半句话 / 混两个话题），
    向量与关键词检索都受累。新法：
      ① 先按空行把全文切成自然块，markdown 标题识别为「小节名」；
      ② 过小的相邻同节块合并到接近 CHUNK_SIZE，超大的块交给 _split_long_block 按句切；
      ③ 每片记录所属小节（section）与片序号（part），供引用卡片 / 后续质检使用。

    参数 text：全文纯文字。
    返回：[(切片正文, {"section": 小节标题, "part": 片序号}), ...]
    """
    # 1) 统一换行符 + 去每行行尾空白（保留段落结构，不再压平换行）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(ln.rstrip() for ln in text.split("\n")).strip()
    if not text:
        return []
    # 让 markdown 标题独占一块（前后补空行），便于下一步把它单独识别成小节名
    text = re.sub(r"(?m)^(#{1,6}\s+.*)$", r"\n\1\n", text)

    # 2) 按空行切自然块；纯标题块记为随后内容的小节名，本身不单独成片
    blocks, current_section = [], ""
    for blk in re.split(r"\n\s*\n", text):
        blk = blk.strip()
        if not blk:
            continue
        m = re.match(r"^#{1,6}\s+(.*)$", blk)
        if m and "\n" not in blk:
            current_section = m.group(1).strip()
            continue
        blocks.append((current_section, blk))

    # 3) 合并过小块 / 拆分超大块，使每片尽量接近但不超过 CHUNK_SIZE
    chunks, buf, buf_section = [], "", ""
    for section, blk in blocks:
        if len(blk) > CHUNK_SIZE:                    # 单块超长：先结算缓冲区，再按句拆
            if buf.strip():
                chunks.append((buf.strip(), buf_section)); buf, buf_section = "", ""
            for piece in _split_long_block(blk):
                chunks.append((piece, section))
            continue
        # 跨小节、或累加会超 CHUNK_SIZE → 先结算当前缓冲区，再另起一片
        if buf and (section != buf_section or len(buf) + len(blk) + 1 > CHUNK_SIZE):
            chunks.append((buf.strip(), buf_section))
            buf, buf_section = blk, section
        else:
            buf = f"{buf}\n{blk}" if buf else blk    # 同节且没超 → 继续往缓冲区攒
            buf_section = section
    if buf.strip():
        chunks.append((buf.strip(), buf_section))

    # 4) 附 part 序号，组装成 (正文, 元数据) 返回
    return [(txt, {"section": sec, "part": i}) for i, (txt, sec) in enumerate(chunks)]
