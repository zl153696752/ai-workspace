from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import UploadFile, File, HTTPException
import uuid


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

class ChatRequest(BaseModel):
    messages: list  # 接收前端发来的完整对话历史

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """流式对话接口（支持多轮上下文）"""
    def generate():
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=req.messages,   # ← 核心改动：把整个历史传给模型（原来只传一句话）
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

    # 2. 生成随机文件名，防止同名文件互相覆盖（原名返回给前端展示用）
    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)

    # 3. 写入磁盘（"wb" = 以二进制写入，文件上传必须用二进制）
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    return {"filename": file.filename, "saved_as": save_name, "size": len(content)}
