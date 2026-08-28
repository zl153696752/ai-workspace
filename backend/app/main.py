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


def extract_text(save_path: str, ext: str) -> str:
    """从上传文件里提取纯文字（.pdf 用 pypdf 解析，txt/md 直接读）"""
    if ext == ".pdf":
        reader = PdfReader(save_path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    with open(save_path, "r", encoding="utf-8") as f:
        return f.read()


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
    """流式对话接口（支持多轮上下文 + 知识库检索）"""
    messages = req.messages

    # ===== RAG 核心 1：检索 =====
    # 用用户最新一句话去知识库里找最相关的 3 个切片（问题自动转向量、自动比距离）
    if collection.count() > 0:  # 知识库里有文档才检索，空的直接跳过，避免报错
        last_question = messages[-1]["content"]
        # include 参数：除了文档原文，把“距离”也返回回来（距离越小越相关）
        results = collection.query(
            query_texts=[last_question],
            n_results=3,
            include=["documents", "distances"]
        )
        # 相关性过滤：只保留距离 < 1.2 的切片，无关问题（如闲聊）会被全部过滤掉（阈值是经验值，可自行调）
        retrieved = [
            doc for doc, dist in zip(results["documents"][0], results["distances"][0])
            if doc and doc.strip() and dist < 1.2
        ]
    else:
        retrieved = []

    # ===== RAG 核心 2：拼提示词 =====
    # 检索到资料，就以 system 消息的身份放在对话历史最前面（模型会把它当“背景知识”）
    if retrieved:
        context = "\n\n".join(retrieved)
        system_prompt = (
            "你是企业知识库助手。以下是参考资料：\n\n"
            f"{context}\n\n"
            "请基于以上参考资料回答用户问题。"
            "如果资料中没有相关内容，请如实说‘知识库中未提及’，不要编造。"
        )
        messages = [{"role": "system", "content": system_prompt}] + messages

    def generate():
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,  # 注意：用的是拼好资料的 messages，不是 req.messages
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")


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

    # 3. 用指纹作为磁盘文件名：同内容 → 同文件名 → 重复上传只覆盖同一个文件
    save_name = f"{content_hash}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)

    # 4. 写入磁盘（内容上面已读好，这里只管写）
    with open(save_path, "wb") as f:
        f.write(content)

    # 5. 入库知识库：提取文字 → 切片 → 向量化存入 Chroma（新增）
    text = extract_text(save_path, ext)
    chunks = split_text(text)
    if chunks:
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
    else:
        duplicated = False

    return {"filename": file.filename, "saved_as": save_name, "size": len(content), "chunks": len(chunks),
            "duplicated": duplicated}
