from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel


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
