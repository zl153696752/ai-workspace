from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import UploadFile, File, HTTPException
import hashlib  # 新增：算内容指纹
import chromadb
from pypdf import PdfReader
import io
import json

load_dotenv()

app = FastAPI(title="AI Workspace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")  # backend/uploads/ 目录
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 目录不存在就自动创建，存在也不报错
ALLOWED_EXT = [".txt", ".md", ".pdf"]  # 只允许上传这三种格式

# ===== 向量数据库 Chroma（数据持久化到 backend/chroma_db/ 目录）=====
chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
collection = chroma_client.get_or_create_collection(name="knowledge")  # 知识库，没有就创建，有就直接用

CHUNK_SIZE = 300  # 每个切片最多 300 字（字太短上下文不足，太长检索不准）
CHUNK_OVERLAP = 50  # 相邻切片重叠 50 字，防止句子被截断


def extract_text(content: bytes, ext: str) -> str:
    """从文件内容（字节）提取纯文字，不再依赖磁盘文件"""
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(content))  # BytesIO：把字节包装成内存中的"假文件"给 pypdf 读
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return content.decode("utf-8", errors="ignore")  # errors="ignore"：遇到编码怪异的字节直接跳过，不崩


def split_text(text: str) -> list:
    """长文切片：每片最多 CHUNK_SIZE 字，相邻两片重叠 CHUNK_OVERLAP 字"""
    text = text.replace("\n", " ")  # 换行压成空格，避免列表、标题被拆散后语义断裂
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():  # 纯空白片段跳过（如文件末尾）
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP  # 往回退 overlap 长度，保证重叠区内容不丢（见 8.2 ③）
    return chunks


class ChatRequest(BaseModel):
    messages: list  # 接收前端发来的完整对话历史


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """流式对话接口（SSE：sources / token / done 三种事件）"""
    messages = req.messages

    # ===== RAG 核心 1：检索（include 新增 metadatas，把文件名带回来）=====
    if collection.count() > 0:
        last_question = messages[-1]["content"]
        results = collection.query(
            query_texts=[last_question],
            n_results=3,
            include=["documents", "distances", "metadatas"]   # ← 新增 metadatas
        )
        # 阈值过滤照旧，只是连 metadata 一起打包成 (文档, 元数据) 元组
        hits = [
            (doc, meta) for doc, dist, meta in zip(
                results["documents"][0], results["distances"][0], results["metadatas"][0])
            if doc and doc.strip() and dist < 1.2
        ]
    else:
        hits = []

    # ===== 新增：构建编号引用源（按“文件名+内容”去重，重复切片只算一条来源）=====
    sources = []          # 发给前端的卡片列表：[{id, filename, snippet}]
    context_parts = []    # 拼进提示词的编号资料：[1] (来自: xx) 内容...
    seen = set()
    for doc, meta in hits:
        key = (meta.get("filename"), doc)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"id": len(sources) + 1, "filename": meta.get("filename", "未知来源"), "snippet": doc})
        context_parts.append(f"[{len(sources)}] (来自: {meta.get('filename', '未知来源')})\n{doc}")

    # ===== RAG 核心 2：拼提示词（升级为“编号版”，并要求模型标注引用）=====
    if sources:
        context = "\n\n".join(context_parts)
        system_prompt = (
            "你是企业知识库助手。以下是参考资料，每条以编号开头：\n\n"
            f"{context}\n\n"
            "请基于以上参考资料回答用户问题。要求：\n"
            "1. 在引用了资料的句子末尾标注资料编号，如 [1]，编号只能来自上述资料；\n"
            "2. 如果资料中没有相关内容，请如实说‘知识库中未提及’，不要编造。"
        )
        messages = [{"role": "system", "content": system_prompt}] + messages

    def generate():
        # ① 先发 sources 事件：结构化数据单独一条消息，先于正文到达，前端可提前渲染卡片
        if sources:
            yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            stream=True
        )
        # ② 逐字流式：每个 token 包成一条 token 事件（JSON 转义保证换行不撕裂消息帧）
        for chunk in response:
            if chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                yield f"event: token\ndata: {json.dumps({'content': piece}, ensure_ascii=False)}\n\n"

        # ③ 结束事件：明确告诉前端流结束了（不依赖连接断开来判断）
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")   # ← text/plain 换掉了


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """文档上传接口"""
    # 1. 校验文件类型
    ext = os.path.splitext(file.filename)[1].lower()  # 取后缀，如 ".txt"
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # 2. 读取文件内容，计算内容指纹（MD5：内容相同指纹必相同）
    content = await file.read()
    content_hash = hashlib.md5(content).hexdigest()

    # 3. 先提取文字、切片：提不出内容就直接拒收，文件不落盘（新位置）
    text = extract_text(content, ext)  # 注意：现在传的是 content 字节，不再是路径
    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="未能提取到文字，可能是图片型/扫描件文件，暂不支持")

    # 4. 校验通过，才写盘（原逻辑不变）
    save_name = f"{content_hash}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    with open(save_path, "wb") as f:
        f.write(content)

    # 5. 入库知识库：
    ids = [f"{content_hash}-{i}" for i in range(len(chunks))]
    # 先查是不是已经入过库（只用于给前端返回提示，不影响去重逻辑）
    duplicated = len(collection.get(ids=ids)["ids"]) > 0
    # upsert = 存在则更新、不存在则插入；同一份文件传多少次结果都一样（幂等）
    collection.upsert(
        documents=chunks,
        ids=ids,
        metadatas=[{"filename": file.filename}] * len(chunks),
    )
    print(f"知识库切片总数: {collection.count()}")  # 验证观察点，打印在后端终端

    return {"filename": file.filename, "saved_as": save_name, "size": len(content),
            "chunks": len(chunks), "duplicated": duplicated}
