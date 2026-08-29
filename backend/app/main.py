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
MAX_FILE_SIZE = 5 * 1024 * 1024  # 文件大小上限 5MB：入库是同步的，超大文件会卡死上传接口，必须拦
MAX_TEXT_LENGTH = 300000  # 提取后文字量上限 30 万字（约 1000 切片）：文字密集的文件才是入库成本的真正决定因素

# ===== 向量数据库 Chroma（数据持久化到 backend/chroma_db/ 目录）=====
chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
collection = chroma_client.get_or_create_collection(name="knowledge")  # 知识库，没有就创建，有就直接用

CHUNK_SIZE = 300  # 每个切片最多 300 字（字太短上下文不足，太长检索不准）
CHUNK_OVERLAP = 50  # 相邻切片重叠 50 字，防止句子被截断

# ===== 助手人格（身份层：每次对话无条件生效，与知识库检索结果无关）=====
PERSONA = (
    "你是牛来，亮哥（赵亮）的专属 AI 助手，性格沉稳可靠，像一位经验丰富的老秘书。\n"
    "说话规则：\n"
    "1. 始终称呼用户为'亮哥'；\n"
    "2. 结论先行：第一句话直接给答案，再补充细节和出处，不铺垫不客套；\n"
    "3. 语气专业简洁，可偶尔用 emoji 点缀气氛（每条回复最多 1-2 个），不过度；\n"
    "4. 不知道的事如实说明，绝不编造。"
)


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
            if doc and doc.strip() and dist < 1.1  # 阈值 1.1：实测真命中≤1.069、假命中≥1.131，卡在分界线上
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

    # ===== RAG 核心 2：拼提示词（人格永远在，知识库资料按检索结果叠加）=====
    system_prompt = PERSONA  # 身份层：无论是否命中知识库，人格都生效（没命中就是纯闲聊模式）
    if sources:
        context = "\n\n".join(context_parts)
        system_prompt += (
            "\n\n以下是参考资料，每条以编号开头：\n\n"
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

        try:
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
        except Exception:
            # 模型服务故障（限流/断网/密钥失效）：发 error 事件，前端据此提示用户，不明不白卡死是最差的体验；
            # 不透传原始报错细节，避免泄露内部信息（密钥、堆栈等）
            yield f"event: error\ndata: {json.dumps({'message': '模型服务暂时不可用，请稍后再试'}, ensure_ascii=False)}\n\n"

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

    # 2. 大小闸门：超限直接拒收，文件内容不进内存（file.size 缺失时用读完后的兜底检查）
    if file.size is not None and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件太大（超过 5MB），请压缩或拆分后再上传")

    # 3. 读取文件内容，计算内容指纹（MD5：内容相同指纹必相同）
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件太大（超过 5MB），请压缩或拆分后再上传")
    content_hash = hashlib.md5(content).hexdigest()

    # 4. 先提取文字、切片：提不出内容就直接拒收，文件不落盘（新位置）
    try:
        text = extract_text(content, ext)  # 注意：现在传的是 content 字节，不再是路径
    except Exception:
        # 文件损坏/加密等导致解析抛异常：给用户明确提示，而不是裸 500
        raise HTTPException(status_code=400, detail="文件解析失败：文件可能已损坏、加密或格式异常，请检查后重新上传")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="文件文字内容过多（超过 30 万字），请拆分成多份上传")
    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="未能提取到文字，可能是图片型/扫描件文件，暂不支持")

    # 5. 校验通过，才写盘（原逻辑不变）
    save_name = f"{content_hash}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    with open(save_path, "wb") as f:
        f.write(content)

    # 6. 入库知识库：
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


@app.get("/api/files")
async def list_files():
    """知识库文件清单：从 Chroma 元数据聚合而来（后端才是真相之源，不依赖前端本地记录）"""
    if collection.count() == 0:
        return {"files": []}
    data = collection.get(include=["metadatas"])
    counter = {}  # {文件名: 切片数}
    for meta in data["metadatas"]:
        name = meta.get("filename", "未知来源")
        counter[name] = counter.get(name, 0) + 1
    return {"files": [{"filename": n, "chunks": c} for n, c in counter.items()]}


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """从知识库删除文档：按元数据把该文档的所有切片批量删掉（入库时存的 filename 在此兑现）"""
    before = collection.count()
    collection.delete(where={"filename": filename})  # where = 按元数据条件过滤删除
    deleted = before - collection.count()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="知识库中没有这个文件")
    print(f"知识库切片总数: {collection.count()}")
    return {"filename": filename, "deleted_chunks": deleted}
