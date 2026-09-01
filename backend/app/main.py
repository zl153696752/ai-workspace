from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
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
import sys

# langchain
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool as langchain_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  # langchain 1.x 把老 Agent API 挪进了 classic 兼容包（见手册 10.9 坑 4）
# langgraph（第 8 步：预构建 ReAct Agent，流式输出）
# langgraph 1.x 把 create_react_agent 标记弃用，新家：langchain.agents.create_agent（参数名见下方注释）
from langchain.agents import create_agent as create_react_agent
from langchain_core.messages import AIMessageChunk  # 流式过滤用：只把模型的正文块转给前端（11.6 的过滤器要它）

from langchain_mcp_adapters.client import MultiServerMCPClient  # 第 9 步：把 MCP 工具转成 LangChain 工具

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
    "你叫牛来（'牛来'就是你的名字），是亮哥（赵亮）的专属 AI 助手，性格沉稳可靠，像一位经验丰富的老秘书。\n"
    "说话规则：\n"
    "1. 始终称呼用户为'亮哥'；\n"
    "2. 结论先行：第一句话直接给答案，再补充细节和出处，不铺垫不客套；\n"
    "3. 语气专业简洁，可偶尔用 emoji 点缀气氛（每条回复最多 1-2 个），不过度；\n"
    "4. 不知道的事如实说明，绝不编造；\n"
    "5. 被问及'你叫什么名字'等身份问题时，直接回答自己叫牛来，不要否认或另起名字。"
)

# ===== 第 7 步：Tool Calling（让模型自己决定要不要查知识库）=====
# 工具描述 = 给模型看的“岗位说明书”：什么时候用、传什么参数，全写在这里。
# description 的措辞直接决定模型的调用准确率，这是工具设计的核心（见 10.9 坑 1）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "检索企业知识库（用户上传的文档）。"
                "当用户的问题涉及公司制度、内部规定、工作事务或用户个人档案时，"
                "必须先调用本工具获取文档原文，不要凭自己的知识回答；"
                "与上述内容无关的常识问题和闲聊不要调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索语句：把用户的问题改写成贴近文档措辞的查询"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def search_knowledge_base(query: str) -> list:
    """知识库检索工具的真正执行体（复用第 5 步检索逻辑：模型只负责决定调用，执行权始终在我们的代码手里）"""
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "distances", "metadatas"]
    )
    # 阈值 1.1 保留为第二道防线：即使模型决定调工具，不相关的切片依然会被过滤（双保险）
    return [
        (doc, meta) for doc, dist, meta in zip(
            results["documents"][0], results["distances"][0], results["metadatas"][0])
        if doc and doc.strip() and dist < 1.1
    ]


# ===== 第 7 步：LangChain 版 Agent（对照实现）=====
USE_LANGCHAIN = False  # 开关：默认 False 改成 True 后 /api/chat 由 LangChain Agent 接管，用来对比两版的行为差异
USE_LANGGRAPH = True   # 第 8 步：新主力。优先级最高：True 时 /api/chat 由 LangGraph 接管（流式+卡片都有）
USE_MCP = True           # 第 9 步：接入 MCP 工具（官方 fetch 联网抓取）。True 时牛来可以抓取网页；加载失败自动降级回普通图（12.9 坑 2）


_mcp_client = None       # MCP 客户端（懒加载后常驻）
_mcp_tools_cache = None  # MCP 工具缓存：None=没加载过，[]=加载失败，有值=加载成功（三种状态别混，12.9 坑 2）


async def get_mcp_tools():
    """MCP 工具加载器（方向 A）：懒加载+缓存，stdio 子进程只拉起一次（12.9 坑 5）；
    失败给空列表降级，绝不阻断主流程（12.9 坑 2）"""
    global _mcp_client, _mcp_tools_cache
    if not USE_MCP:
        return []
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache
    try:
        _mcp_client = MultiServerMCPClient({
            "fetch": {
                "command": sys.executable,       # 用当前 venv 的 python，别写死路径（12.9 坑 1）
                "args": ["-m", "mcp_server_fetch"],
                "transport": "stdio",
            }
        })
        async with _mcp_client.session("fetch"):
            _mcp_tools_cache = await _mcp_client.get_tools()
        print(f"[MCP] 已加载工具: {[t.name for t in _mcp_tools_cache]}")
    except Exception as e:
        print(f"[MCP] 加载失败，降级为普通模式: {e}")
        _mcp_tools_cache = []
    return _mcp_tools_cache


# @tool 装饰器：把普通函数变成工具，工具描述直接从函数签名和 docstring 自动生成（手写版要写几十行 JSON）
@langchain_tool
def search_knowledge_base_lc(query: str) -> str:
    """检索企业知识库。当问题涉及公司制度、内部规定或用户个人档案时必须先调用本工具。"""
    hits = search_knowledge_base(query)  # 复用同一个执行体：两版只是“编排方式”不同，“能力”是同一个
    if not hits:
        return "知识库中没有检索到相关内容"
    parts = [f"[{i + 1}] (来自: {meta.get('filename', '未知来源')})\n{doc}" for i, (doc, meta) in enumerate(hits)]
    return "\n\n".join(parts)


lc_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",  # DeepSeek 兼容 OpenAI 格式，改个地址直接连上（不需要专门的包）
)
# 提示词模板：人格 + 历史占位符 + agent_scratchpad（Agent 中间思考过程的固定占位，固定写法背下来即可）
lc_prompt = ChatPromptTemplate.from_messages([
    ("system", PERSONA),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
lc_agent = create_tool_calling_agent(lc_llm, [search_knowledge_base_lc], lc_prompt)
lc_executor = AgentExecutor(agent=lc_agent, tools=[search_knowledge_base_lc],
                            verbose=True)  # verbose=True：思考过程打印在后端终端，对比手写版时重点看这个
# ===== 第 8 步：LangGraph 版 Agent（新主力：流式输出 + 护栏齐全）=====
# 设计取舍（面试必讲）：检索由我们的代码先做（卡片先出场、未命中自然回应可控），
# 图只负责“基于资料可靠地生成回答”——确定性的活给代码，生成性的活给模型，还省一次决策调用。
# 工具照常注册：资料没查到时，模型可以主动复核查一下（图上循环自动支持，手写版做不到）
GRAPH_SYSTEM = (
    PERSONA + "\n\n回答规则：\n"
    "1. 用户提供【编号资料】时，只基于资料回答；引用了资料的句子末尾标注编号如[1][2]；资料没覆盖的就如实说明。\n"
    "2. 【编号资料】为空或和问题无关时，如实告知知识库中没有相关资料（不要调用工具，不要编造）。\n"
    "3. 用户的问题需要知识库里的事实、而【编号资料】显然没覆盖时，可以调用 search_knowledge_base 复核一次。\n"
    "4. 与知识库无关的常识和闲聊，直接回答，不要调用工具。\n"
    "5. 用户需要实时网页内容（某个网页的信息、最新内容）时，调用 fetch 工具抓取后回答；知识库问题和闲聊不要调用它。"
)
# create_react_agent（实为新 API create_agent 的别名）= 预构建的 ReAct 图（思考⇄工具自动循环）；
# 新版参数名是 system_prompt（旧版叫 prompt）；防死循环护栏 recursion_limit 改在调用时传（见 11.6 分支的 config）
lg_graph = create_react_agent(lc_llm, [search_knowledge_base_lc], system_prompt=GRAPH_SYSTEM)


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
    """流式对话接口（第 7 步：Tool Calling 版；SSE 事件：tool / sources / token / error / done）"""
    messages = req.messages

    # ===== LangGraph 分支（USE_LANGGRAPH = True 时接管，第 8 步新主力）=====
    if USE_LANGGRAPH:
        # 检索前先改写（修复“那餐补呢”查不到的问题）：“那餐补呢”距离 1.478 过不了 1.1 阈值，
        # 改写成文档措辞后距离降到 0.398——第 7 步模型自主改写的能力在“代码先检索”架构里丢了，这里补回；
        # 带最近几轮历史解决多轮指代（“那它呢”），失败就用原话兜底，绝不阻断主流程（一次短调用的成本）
        try:
            rewrite = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "system", "content": "结合对话历史，把用户最新问题改写成一句独立完整的检索语句（贴近知识库文档措辞）。只输出检索语句本身，不要解释。"}] + messages[-4:],
            )
            rewritten = (rewrite.choices[0].message.content or "").strip()
        except Exception:
            rewritten = ""
        query = rewritten or messages[-1]["content"]
        print(f"[LangGraph 检索] 原话: {messages[-1]['content']} → 改写: {query}")  # 观察点：后端终端看改写效果

        mcp_tools = await get_mcp_tools()
        # MCP 工具和本地工具在同一个列表里进图——模型眼里它们没有区别（都是说明书三件套）
        active_graph = create_react_agent(lc_llm, [search_knowledge_base_lc] + mcp_tools,
                                          system_prompt=GRAPH_SYSTEM) if mcp_tools else lg_graph

        hits = search_knowledge_base(query)  # 代码先检索：卡片先出场、未命中自然回应可控（设计取舍见 11.3）
        sources = []
        context_parts = []
        seen = set()
        for doc, meta in hits:  # 去重逻辑与手写版完全一致（第 6 步的活儿一行不丢）
            key = (meta.get("filename"), doc)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"id": len(sources) + 1, "filename": meta.get("filename", "未知来源"), "snippet": doc})
            context_parts.append(f"[{len(sources)}] (来自: {meta.get('filename', '未知来源')})\n{doc}")
        material = "\n\n".join(context_parts) if context_parts else "（无）"

        async def generate():
            # 注意必须是 async def：里面有 async for（消费 LangGraph 的异步流），普通 def 会报 SyntaxError；
            # StreamingResponse 对异步生成器原生支持（手写版的同步生成器照常工作，互不影响）
            # 卡片先发：顺序和第 6 步保持一致（先卡片后正文），前端零改动的前提
            yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"
            # 组装本次对话：人格+规则 + 历史 + 编号资料 + 当前问题（检索结果以资料形式进提示词）
            lg_messages = [{"role": "system", "content": GRAPH_SYSTEM}] + messages[:-1] + [
                {"role": "user", "content": f"【编号资料】\n{material}\n\n【用户问题】\n{query}"}
            ]
            try:
                # astream(stream_mode="messages")：模型每吐一个 token 就产出一条消息事件——全场景打字机的来源；
                # 对照第 7 步：手写版只有“调工具路径”才有打字机，这里闲聊也有（11.1 遗憾 1 清算）；
                # config 里的 recursion_limit=10 是防死循环护栏：最多转 10 圈强制熔断（新版 API 护栏在调用时传，不再在构图时传）
                async for _chunk, metadata in active_graph.astream(
                        {"messages": lg_messages},
                            config = {"recursion_limit": 10}, stream_mode = "messages"):
                    if isinstance(_chunk, AIMessageChunk) and _chunk.content:
                        yield f"event: token\ndata: {json.dumps({'content': _chunk.content}, ensure_ascii=False)}\n\n"
            except Exception:
                yield f"event: error\ndata: {json.dumps({'message': '模型服务暂时不可用，请稍后再试'}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ===== LangChain 分支（USE_LANGCHAIN = True 时接管：框架自动完成“决策-调用-回填”循环）=====
    elif USE_LANGCHAIN:
        def generate():
            try:
                result = lc_executor.invoke({
                    "input": messages[-1]["content"],  # 最新一句话作为当前输入
                    "chat_history": messages[:-1],  # 其余作为历史（框架自动转成消息对象）
                })
                # 框架把多轮循环全部封装在 invoke 内部，一把返回最终文本（对照手写版的两次显式调用）
                yield f"event: token\ndata: {json.dumps({'content': result['output']}, ensure_ascii=False)}\n\n"
            except Exception:
                yield f"event: error\ndata: {json.dumps({'message': '模型服务暂时不可用，请稍后再试'}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ===== ① 决策调用（注意：不开流式！）=====
    # 必须先拿到完整返回，才能判断模型是“想调工具”还是“直接回答”，流式边收边判做不到也没必要（这次调用很短）
    base_messages = [{"role": "system", "content": PERSONA}] + messages
    try:
        decision = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=base_messages,
            tools=TOOLS,  # 把工具说明书交给模型，用不用由它决定（第 5 步的“无条件检索”在这里被彻底替换）
        )
    except Exception:
        # 决策阶段就失败：用标准 HTTP 错误返回（第 6.5 步的前端错误兜底会自动接住并提示用户）
        raise HTTPException(status_code=500, detail="模型服务暂时不可用，请稍后再试")

    choice = decision.choices[0].message
    tool_calls = getattr(choice, "tool_calls", None)  # 模型的调用意图（没有就是 None = 它决定直接答）
    print(f"[工具决策] {'调用 ' + tool_calls[0].function.name if tool_calls else '直接回答'}")  # 观察点：打在后端终端

    sources = []  # 引用卡片（第 6 步逻辑原样保留）
    final_messages = base_messages
    if tool_calls:
        # ===== ② 模型决定查知识库：解析参数 → 执行工具 → 结果回填对话 =====
        call = tool_calls[0]  # 我们只有一个工具，取第一个即可（多工具场景要循环处理）
        query = json.loads(call.function.arguments)["query"]  # 参数是 JSON 字符串，要反序列化（解析失败说明模型乱传，见 10.9 坑 3）
        hits = search_knowledge_base(query)

        # 构建编号资料和引用卡片（与第 6 步完全一致）
        context_parts = []
        seen = set()
        for doc, meta in hits:
            key = (meta.get("filename"), doc)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"id": len(sources) + 1, "filename": meta.get("filename", "未知来源"), "snippet": doc})
            context_parts.append(f"[{len(sources)}] (来自: {meta.get('filename', '未知来源')})\n{doc}")

        if sources:
            tool_result = (
                    "以下是检索到的知识库内容，按编号排列：\n\n"
                    + "\n\n".join(context_parts)
                    + "\n\n请基于以上内容回答。在引用了资料的句子末尾标注编号如 [1]；资料中没有的就如实说明。"
            )
        else:
            tool_result = "知识库中没有检索到和问题相关的内容。请如实告知用户知识库中没有相关资料。"

        # 按协议追加两条消息：assistant 的“调用意图” + tool 角色的“执行结果”（顺序不能乱，id 必须配对）
        final_messages = base_messages + [
            choice,  # OpenAI SDK 支持把原始消息对象（含 tool_calls 字段）直接传回去，不用手工拼 dict
            {"role": "tool", "tool_call_id": call.id, "content": tool_result}
        ]

    def generate():
        # ③ 决策透明事件：前端不认识这个事件名会自动忽略（只认 sources/token/error），纯粹用于观察和以后扩展
        yield f"event: tool\ndata: {json.dumps({'called': bool(tool_calls)}, ensure_ascii=False)}\n\n"
        if sources:
            yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

        try:
            if tool_calls:
                # ④ 第二次调用（流式）：基于工具结果生成正式回答——打字机效果在这一步才有
                response = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=final_messages,
                    stream=True
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        piece = chunk.choices[0].delta.content
                        yield f"event: token\ndata: {json.dumps({'content': piece}, ensure_ascii=False)}\n\n"
            else:
                # 没调工具：决策调用已经拿到完整回答，整段一次吐出（无打字机效果，设计权衡见 10.9 坑 5）
                yield f"event: token\ndata: {json.dumps({'content': choice.content or ''}, ensure_ascii=False)}\n\n"
        except Exception:
            yield f"event: error\ndata: {json.dumps({'message': '模型服务暂时不可用，请稍后再试'}, ensure_ascii=False)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
        metadatas=[{"filename": file.filename, "saved_as": save_name}] * len(chunks),
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


@app.get("/api/files/{filename}/download")
async def download_file(filename: str):
    """下载知识库原文件：原文件名 → 查 Chroma 元数据拿哈希名 → 返回磁盘实体文件"""
    data = collection.get(where={"filename": filename}, include=["metadatas"], limit=1)
    if not data["ids"]:
        raise HTTPException(status_code=404, detail="知识库中没有这个文件")
    saved_as = data["metadatas"][0].get("saved_as")
    if not saved_as:
        # 旧数据升级前入库，元数据里没有 saved_as——如实告知，不蒙混（第 6.5 步的错误提示原则）
        raise HTTPException(status_code=404, detail="该文件为历史入库数据，未记录原文件，请重新上传")
    file_path = os.path.join(UPLOAD_DIR, saved_as)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="服务器上的原文件已丢失，请重新上传")
    return FileResponse(file_path, filename=filename)  # filename 参数让浏览器按原文件名下载（中文名自动编码）
