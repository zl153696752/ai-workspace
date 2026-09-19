# ===== FastAPI 应用组装 + 路由层（整个后端的入口）=====
# 只做两类事：① 应用组装（FastAPI 实例、跨域、请求体模型）；② 七个 HTTP 接口（/api/chat + 登录 + 上传/清单/删除/下载）。
# 业务逻辑不在这里（配置在 config.py、鉴权在 auth.py、检索切片在 rag.py、Agent 编排在 agents.py），本文件只负责接收请求→调用→按 HTTP 返回。
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
# HTTPException：抛出带状态码的错误，FastAPI 自动转成 {"detail": "错误文案"} 的 JSON 响应
from fastapi.middleware.cors import CORSMiddleware  # 跨域中间件（解决前端域名/端口与后端不同时浏览器的拦截）
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles  # 托管前端静态导出产物（见文件末尾）
# StreamingResponse：流式响应，打字机效果靠它；FileResponse：直接把磁盘文件作为响应体返回，供下载
from pydantic import BaseModel  # 请求体校验：定义好字段和类型，FastAPI 自动校验并生成接口文档
# MCP 工具加载成功时要现场重建带 MCP 工具的图，所以这里也要拿到构图函数
from langchain.agents import create_agent as create_react_agent
# LangGraph 流式会吐多种消息块，只想要“模型正文”那种，靠这个类做类型判断过滤
from langchain_core.messages import AIMessageChunk
import hashlib  # 计算文件内容的 MD5 指纹（用于内容查重和文件命名）
import asyncio  # C2：后台暖机 MCP 工具用 create_task，不阻塞启动
from contextlib import asynccontextmanager  # C2：把"启动暖机"包成 FastAPI lifespan 异步上下文
import json  # 序列化 SSE 推送的数据、解析模型返回的工具参数
import os  # 路径拼接、判断文件是否存在

# 下面几行是本项目自己的模块，依赖单向：config（资源）← auth（鉴权）/ rag（检索）← agents（编排）← main（组装+路由），不会循环引用。
from .config import (collection, UPLOAD_DIR, ALLOWED_EXT, MAX_FILE_SIZE, MAX_TEXT_LENGTH)
from .rag import extract_text, split_text
from .auth import verify_password, issue_token, get_identity, require_liang  # 步骤4：验口令+签发票+解析身份+亮哥守卫
from .graph import agent_graph  # 步骤5：多 Agent 编排图（Supervisor + 3 worker 的 StateGraph），唯一生产路径
from .agents import get_mcp_tools  # C2：启动暖机调它（依赖方向 main→agents 正确、不成环）
from .llm_gateway import start_trace, get_trace_spans
from .db import init_db, save_trace, get_memories, delete_memory  # 步骤11：记忆治理要读(get)/删(delete)
from .memory import extract_and_store  # 步骤11：记忆官，回答后异步抽取亮哥的偏好/事实

# ===== C2：启动暖机 MCP 工具 =====
# 为什么：Supervisor 是同步节点、每请求都调 build_tools_manifest() 读 _mcp_tools_cache；
# 而 MCP 工具是异步懒加载的，冷启动首请求进来时缓存还是 None → 工具清单只剩技能。
# 这里用启动钩子【后台】暖机（create_task 不 await）：既不阻塞启动、也不会被 MCP 子进程卡死；
# 暖好后 build_tools_manifest 的"冷启动不缓存"逻辑会自动重建出完整清单（B 步骤埋的自愈）。
_mcp_warm_task = None  # 存引用，防止 fire-and-forget 任务被垃圾回收中途掐断


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_warm_task
    init_db()   # ← 新增：启动即建表（幂等，第二次启动不会清数据）
    _mcp_warm_task = asyncio.create_task(get_mcp_tools())  # 不 await：后台跑；失败自动降级为 []（get_mcp_tools 内部已兜）
    print("[启动] 已后台发起 MCP 工具暖机（不阻塞启动）")
    yield  # ← 应用在此运行；yield 之后是关闭清理（本项目无需，子进程随主进程退出）


app = FastAPI(title="AI Workspace", lifespan=lifespan)  # title 显示在 /docs；lifespan 挂上启动暖机钩子

# 跨域配置：端口不同即跨域，浏览器会拦截前端(3000)请求后端(8000)，后端须显式声明允许的来源。
# 白名单从环境变量 CORS_ORIGINS 读（逗号分隔），本地默认放行 localhost:3000。
# 线上同源方案下前端与 API 同 origin 不触发跨域，这段主要为本地开发保留。
_cors_env = os.getenv("CORS_ORIGINS", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_env.split(",") if o.strip()],
    allow_methods=["*"],  # 允许所有 HTTP 方法（GET / POST / DELETE ...）
    allow_headers=["*"],  # 允许所有请求头
)


@app.get("/health")
async def health():
    """健康检查接口：部署平台靠它判断容器活没活。

    必须【轻】：不碰模型、不碰网络，只报告自身和数据库状态（拿 /api/chat 探活会烧 token、慢，还会因模型超时误判容器挂了）。
    返回的 chroma_chunks 顺便当“知识库空没空”的自检：部署完看这个数字，为 0 说明种子文档没灌进去。
    """
    return {
        "status": "ok",
        "chroma_chunks": collection.count(),  # 知识库里的切片总数，0 = 空库
        "uploads_dir_ok": os.path.isdir(UPLOAD_DIR),
    }


class LoginRequest(BaseModel):
    """登录请求体：只要一个口令。身份固定是亮哥（本项目只有这一个特权身份，无注册）。"""
    password: str


@app.post("/api/login")
async def login(req: LoginRequest):
    """亮哥登录：验口令 → 签发 30 天有效的 JWT 门票 → 返回给前端存起来、后续请求带上。

    口令错 / 未配置哈希 → 401。统一文案"口令不正确"，绝不透露是"口令错"还是"没配置"，
    避免给暴力破解者反馈（这是安全接口的常规做法）。
    """
    if not verify_password(req.password):
        raise HTTPException(status_code=401, detail="口令不正确")
    return {"token": issue_token(), "token_type": "bearer"}


class ChatRequest(BaseModel):
    """对话接口的请求体结构。

    后端无状态、不保存会话，前端每次发消息都把完整历史一起发来（上下文由前端维护，后端重启不丢历史）。
    messages 是 OpenAI 约定的消息数组：[{"role": "user"/"assistant", "content": "..."}, ...]
    """
    messages: list  # 完整对话历史，最后一条就是用户刚发的这句话


# ===== 步骤11：记忆抽取的后台任务管理 =====
# 🔴 fire-and-forget 陷阱：asyncio.create_task 的返回值若不被任何变量引用，任务可能在跑完前就被垃圾回收掐断。
#    用一个模块级 set 持有活跃任务的引用、跑完自动 discard——这是 asyncio 后台任务的标准防丢写法（和上面 _mcp_warm_task 存引用同理）。
_memory_tasks = set()


def _spawn_memory_extract(query: str):
    """把【同步阻塞】的记忆抽取丢进线程池后台跑（asyncio.to_thread），既不阻塞事件循环、也不拖累 SSE 收尾。
    extract_and_store 内部已全程 try/except，这里只负责发起、不 await 结果：抽取失败无非这轮不记，绝不影响已经推完的回答。"""
    task = asyncio.create_task(asyncio.to_thread(extract_and_store, query))
    _memory_tasks.add(task)
    task.add_done_callback(_memory_tasks.discard)


@app.post("/api/chat")
async def chat(req: ChatRequest, identity: dict = Depends(get_identity)):
    """流式对话接口（整个项目的核心接口）。

    返回方式是 SSE（Server-Sent Events，服务器推送事件）：
    一次 HTTP 请求不断开，服务器持续往前端推一条条“事件”，前端收到就实时渲染，这就是打字机效果的实现方式。
    （对比你熟悉的前端概念：SSE 类似只能服务器→客户端单向推送的 WebSocket，基于普通 HTTP，不用额外协议）
    每条事件的文本格式固定为两行 + 一个空行：
        event: 事件名
        data: JSON 字符串
    本项目共 6 种事件：
        sources —— 引用卡片数据（文件名 + 原文片段），前端渲染成可展开的来源卡片
        token   —— 模型正文的一小段文字，前端把它们拼接起来显示
        meta    —— 本次问答的轻量指标（延迟/token/成本/调用次数），全员可见，渲染在回答角落
        trace   —— 完整全链路 trace（各步 span 明细），【仅亮哥】可见，渲染"🔍 推理链路"抽屉
        error   —— 出错了，前端弹出提示文案
        done    —— 本次回答结束，前端收尾（停止 loading 动画）
    """
    messages = req.messages

    # ============================================================
    # 步骤5b：完整流程已搬进多 Agent 图（graph.py）——Supervisor 路由 → 检索质检 worker（改写+检索+质检）
    #         → 合成 worker（流式生成）。这里只做两件事：组装初始 State、消费图的双模式流转成 SSE。
    # ============================================================
    # 组装初始 State（黑板）：把这句话 + 身份喂进图，其余字段由各节点填充
    init_state = {
        "messages": messages,  # 完整对话历史
        "query": messages[-1]["content"],  # 用户当前这句原话（合成时用它，不用改写句）
        "is_liang": identity["is_liang"],  # 身份：决定检索过滤 + 人格风格
        "trace": [],
    }

    async def generate():
        # 🔴 双模式流式 stream_mode=["messages", "updates"] —— 这是 5b 保住 SSE 的关键：
        #   updates  ：每个节点跑完吐一次它的 State 更新；用来【在 synthesize 出完答案后发准确溯源卡片】。
        #   messages ：模型每吐一个 token 一条；用来发正文（打字机），但要【按节点名过滤】只放行 synthesize 的。
        # 多模式下 astream 每次产出 (mode, chunk) 二元组：mode 是 "updates"/"messages"，chunk 随 mode 不同。
        start_trace()
        node_spans = []  # ← 新增：累积节点级 span（State.trace 的增量）
        try:
            async for mode, chunk in agent_graph.astream(
                    init_state, config={"recursion_limit": 10}, stream_mode=["messages", "updates"]):
                if mode == "updates":
                    # 🔴 卡片准确化：不再 eager 发 retrieve 的全部 top-3，改成【等 synthesize 出完答案】发它过滤后的
                    #    cited_sources（只含答案真正引用到的片段）。所以卡片在正文之后到达（②甲）——准确溯源优先于秒出。
                    # synthesize 的 updates 在它那批 token 全流完之后才到，天然满足"答案后发卡"。
                    # ← 新增：每个节点跑完吐一次更新，把它返回的 trace 增量收进来
                    for node_out in chunk.values():
                        if isinstance(node_out, dict):
                            node_spans.extend(node_out.get("trace") or [])
                    upd = chunk.get("synthesize")
                    if upd is not None:
                        cited = upd.get("cited_sources") or []
                        if cited:  # 只在确有被引用的卡片时才发；空（没标/闲聊/天气）不发，前端自然无卡片
                            yield f"event: sources\ndata: {json.dumps(cited, ensure_ascii=False)}\n\n"
                else:  # mode == "messages"
                    # chunk 是 (消息块, 元数据)；三重过滤：synthesize 节点 + AIMessageChunk + content 非空。
                    # 🔴 按 langgraph_node 过滤是关键：不然 retrieve 里的改写、tool 里的中间消息会漏进正文显示给用户。
                    msg, meta = chunk
                    if meta.get("langgraph_node") == "synthesize" and isinstance(msg, AIMessageChunk) and msg.content:
                        yield f"event: token\ndata: {json.dumps({'content': msg.content}, ensure_ascii=False)}\n\n"
        except Exception:
            # 生成中出错（超时/限流/图异常/超 recursion_limit）：HTTP 头已发出改不了状态码，只能用 SSE 事件告诉前端
            yield f"event: error\ndata: {json.dumps({'message': '模型服务暂时不可用，请稍后再试'}, ensure_ascii=False)}\n\n"
        # ← 新增：收尾汇总本次问答的所有模型调用 span（放 done 之前，成败都汇总）
        spans = get_trace_spans()
        summary = {
            "calls": len(spans),
            "total_latency": round(sum(s.get("latency") or 0 for s in spans), 3),
            "prompt_tokens": sum(s.get("prompt_tokens") or 0 for s in spans),
            "completion_tokens": sum(s.get("completion_tokens") or 0 for s in spans),
            "total_cost": round(sum(s.get("cost") or 0 for s in spans), 6),
            "degraded": any(s.get("degraded") for s in spans),
        }
        print(f"[本次问答汇总] {summary}")  # B-1 先 print 验证归集对不对；B-2 再落 SQLite
        # ← 新增：从节点 span 拎出意图/作用域，合并两类降级，落一条全链路 trace
        sup = next((s for s in node_spans if s.get("node") == "supervisor" and not s.get("degraded")), {})
        try:
            save_trace(
                query=messages[-1]["content"],
                identity="liang" if identity["is_liang"] else "guest",
                intents=",".join(sup.get("intents") or []),
                scope=sup.get("scope") or "",
                degraded=summary["degraded"] or any(s.get("degraded") for s in node_spans),
                summary=summary, llm_spans=spans, node_spans=node_spans,
            )
        except Exception as e:
            print(f"[trace落库] 失败（不阻断问答）：{e}")  # 降级留痕：落库炸了也绝不能拖垮用户这次问答

        # ===== 阶段C：可观测性数据分层透出前端 =====
        # ① meta —— 轻量指标，【全员】可见（阶段D 渲染在回答角落："本次 3.7s · 4250 tokens · ¥0.011"）
        meta = {
            "latency": summary["total_latency"],
            "tokens": summary["prompt_tokens"] + summary["completion_tokens"],
            "cost": summary["total_cost"],
            "calls": summary["calls"],
        }
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # ② trace —— 完整全链路明细，【仅亮哥】可见（阶段D 渲染"🔍 推理链路"抽屉）
        #    游客收不到这条事件（后端不发，非前端隐藏）→ 真·身份隔离
        if identity["is_liang"]:
            trace_payload = {"summary": summary, "llm_spans": spans, "node_spans": node_spans}
            yield f"event: trace\ndata: {json.dumps(trace_payload, ensure_ascii=False)}\n\n"

        yield "event: done\ndata: {}\n\n"

        # ===== 步骤11：回答已完整推给用户，此刻才后台异步抽记忆（只对亮哥，游客不记）=====
        # 🔴 为什么放在 done 之后：打字机正文、来源卡片、meta、trace 全都发完了才抽，记忆官再慢也影响不到这次体验。
        if identity["is_liang"]:
            _spawn_memory_extract(messages[-1]["content"])

    # media_type="text/event-stream" 是 SSE 的标准 MIME 类型，前端靠它识别流式响应
    return StreamingResponse(generate(), media_type="text/event-stream")


# ===== 步骤11：长期记忆治理端点（亮哥专属：看系统记了啥 + 一键删错记）=====
@app.get("/api/memories")
async def list_memories(identity: dict = Depends(require_liang)):
    """查亮哥的全部长期记忆（治理入口）。
    🔴 和注入(s4)不同：治理要能看【全部】（含低置信的），故 min_confidence=0、limit 放大——
       注入只喂 confidence>=0.6 的以防污染；治理要让亮哥看到“系统到底记了我啥”，一条都不能藏。"""
    return {"memories": get_memories("liang", min_confidence=0.0, limit=200)}


@app.delete("/api/memories/{memory_id}")
async def delete_one_memory(memory_id: int, identity: dict = Depends(require_liang)):
    """删一条记忆（亮哥发现错记/过时，一键删）。memory_id 走路径参数。
    delete_memory 内部带 user 校验：只能删自己的；删不到(不存在/非本人)→ 404。"""
    if not delete_memory(memory_id, "liang"):
        raise HTTPException(status_code=404, detail="记忆不存在或无权删除")
    return {"deleted": memory_id}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), private: bool = Form(False), identity: dict = Depends(get_identity)):
    """文档上传入库接口：校验 → 提取文字 → 切片 → 查重 → 存盘 → 写入向量库。

    参数 file 由 FastAPI 从 multipart/form-data 自动解析（File(...) 表示必填）。
    返回：{"filename": 原文件名, "saved_as": 磁盘上的实际文件名, "size": 字节数,
           "chunks": 切了多少片, "overwritten": 是否覆盖了同名旧文件}
    前端靠 overwritten 分流提示文案（“上传成功” vs “已覆盖旧版本”）。
    """
    # ----- 0. 私人归属判定 -----
    # 仅亮哥可标私人；游客传的一律强制公共（匿名无归属），即使表单带了 private=true 也忽略——权限矩阵的硬规则。
    is_private = bool(private) and identity["is_liang"]
    # ----- 1. 校验文件类型（白名单）-----
    # splitext 拆出后缀并转小写，".TXT" 和 ".txt" 都能通过。
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        # 400 = 客户端错误。detail 会被前端直接弹给用户，要写清具体原因。
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # ----- 2. 大小闸门（第一道，读内容之前）-----
    # 此时内容还没进内存，超限直接拒收，避免大文件吃满内存。
    if file.size is not None and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件太大（超过 5MB），请压缩或拆分后再上传")

    # ----- 3. 读取内容 + 计算内容指纹 -----
    content = await file.read()  # 异步读出全部字节（await 期间不阻塞其他请求）
    # 第二道大小检查：某些客户端不传 size（file.size 为 None），读完后用 len(content) 兜底。
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件太大（超过 5MB），请压缩或拆分后再上传")
    # MD5 内容指纹：用途有二 —— ① 内容查重（第 5 步）；② 作为磁盘文件名（第 7 步）。
    content_hash = hashlib.md5(content).hexdigest()

    # ----- 4. 提取文字 + 切片（先解析，后落盘）-----
    # 顺序关键：先确认能提取出文字再写盘入库，避免解析失败的文件留下垃圾。
    try:
        text = extract_text(content, ext)  # 传内存字节，非磁盘路径（文件还没存盘）
    except Exception:
        # PDF 加密/损坏/格式伪装都会让解析抛异常，转成 400 + 明确文案，而非裸奔成 500。
        raise HTTPException(status_code=400, detail="文件解析失败：文件可能已损坏、加密或格式异常，请检查后重新上传")
    # 文字量闸门：决定入库成本的是文字量而非文件体积
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="文件文字内容过多（超过 30 万字），请拆分成多份上传")
    chunk_pairs = split_text(text)  # [(正文, {section, part}), ...]
    chunks = [txt for txt, _m in chunk_pairs]  # 只取正文：后续 len(chunks)/documents=chunks 保持不变
    if not chunks:
        # 解析成功却无文字 = 扫描版/图片型 PDF（无文字层），需 OCR，本项目不做。
        raise HTTPException(status_code=400, detail="未能提取到文字，可能是图片型/扫描件文件，暂不支持")

    # ----- 5. 内容查重：完全相同的文件禁止重复上传 -----
    # 切片入库 id = 内容指纹 + 序号（内容寻址命名：id 由内容决定，而非文件名或时间）。
    ids = [f"{content_hash}-{i}" for i in range(len(chunks))]
    # 拿这批 id 查库：查到即说明此内容以前传过（哪怕用的是别的文件名）。
    existing = collection.get(ids=ids, include=["metadatas"])
    if existing["ids"]:
        # 409 = 资源冲突。告诉用户库里已有的文件叫什么（可能是用别的名传的同一份内容）。
        old_name = existing["metadatas"][0].get("filename", file.filename)
        raise HTTPException(status_code=409,
                            detail=f"该文件已上传，禁止重复上传（库中已有内容完全相同的文件：{old_name}）")
    # 拦在写盘前，被拒的上传不留任何副作用（磁盘/库里都不多东西）

    # ----- 6. 同名覆盖：文件名相同但内容不同 = 新版本 -----
    # 不先清旧版，新旧切片会同时留在库里，检索时被一起捞出塞进提示词，模型看到矛盾规则。
    overwritten = False  # 标记“新上传”还是“覆盖旧版”，返回给前端分流提示文案
    old = collection.get(where={"filename": file.filename}, include=["metadatas"])  # 按元数据里的 filename 查出旧版的所有切片
    if old["ids"]:
        # 收集旧版磁盘文件名集合（set 去重：一个文件的切片共享同一 saved_as）
        old_saved = {m.get("saved_as") for m in old["metadatas"]}
        collection.delete(ids=old["ids"])  # 删向量库里的旧切片
        # 再清磁盘旧文件。顺序不能反：须等切片删完再查引用，cleanup 内部靠此判断防误删共享文件。
        cleanup_saved_files(old_saved)
        overwritten = True
        print(f"[覆盖] {file.filename} 旧版 {len(old['ids'])} 个切片已清理")

    # ----- 7. 所有校验都过了，才把原文件写到磁盘 -----
    # 存盘名用“内容指纹 + 原后缀”而非用户文件名：避免撞名、绕开中文编码问题；
    # 原文件名存进元数据 filename 字段，下载时还原。
    save_name = f"{content_hash}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    with open(save_path, "wb") as f:  # "wb" 二进制写；with 保证自动关闭
        f.write(content)

    # ----- 8. 写入向量库 -----
    # upsert：id 存在则更新、否则插入（比 insert 安全，不会因 id 重复报错）。
    collection.upsert(
        documents=chunks,  # 切片正文：Chroma 自动转成向量存起来（embedding）
        ids=ids,  # 每片的主键，和 documents 一一对应
        # 元数据：记录每片来自哪个文件、磁盘名是什么。[字典] * N 复制成每片一份。
        # 不参与向量计算，但用于过滤查询，也是前端引用卡片显示文件名的来源。
        # 每片元数据 = 片级(section, part) + 文件级(filename, saved_as)
        metadatas=[{**m, "filename": file.filename, "saved_as": save_name, "private": is_private} for _txt, m in
                   chunk_pairs],
    )
    print(f"知识库切片总数: {collection.count()}")  # 观察点：终端可看到入库后切片总数

    return {"filename": file.filename, "saved_as": save_name, "size": len(content),
            "chunks": len(chunks), "overwritten": overwritten, "private": is_private}


@app.get("/api/files")
async def list_files(identity: dict = Depends(get_identity)):
    """知识库文件清单接口：返回库里有哪些文件、各切了多少片（前端侧边栏的文件列表靠它）。

    返回：{"files": [{"filename": "公司制度.txt", "chunks": 3}, ...]}
    数据来源是 Chroma 元数据 —— 向量库是“知识库里有什么”的唯一真相之源，
    故换浏览器/清缓存/重启服务清单都不会错。
    """
    if collection.count() == 0:
        return {"files": []}  # 空库直接返回空列表
    # collection.get 不带条件 = 取所有切片元数据（只取 metadatas 不取正文，少传数据）
    data = collection.get(include=["metadatas"])
    # 库里存“切片”，前端要“文件”维度，故按 filename 聚合：统计每个文件名出现次数 = 切片数。
    counter = {}  # {文件名: 切片数}
    priv_flag = {}  # {文件名: 是否私人}——同一文件各切片 private 一致，记录供前端画公/私徽章
    for meta in data["metadatas"]:
        is_priv = bool(meta.get("private", False))  # 老数据无 private 字段 → 默认公共
        if is_priv and not identity["is_liang"]:
            continue  # 游客：私人切片直接跳过——不计入清单、不暴露存在（私人对游客彻底隐身）
        name = meta.get("filename", "未知来源")  # .get 带默认值：早期数据可能没这个字段，不至于报错
        counter[name] = counter.get(name, 0) + 1  # 字典计数：没这个键当 0 再 +1
        priv_flag[name] = is_priv
    # 转成前端好遍历的数组；带 private 字段供前端画「公/私」徽章
    return {"files": [{"filename": n, "chunks": c, "private": priv_flag.get(n, False)} for n, c in counter.items()]}


def cleanup_saved_files(saved_names):
    """清理 uploads/ 目录里已经没人引用的物理文件（覆盖上传、删除文件两条路都用它）。

    参数 saved_names：一批磁盘文件名（就是入库时的 save_name，形如 "md5哈希.txt"）
    无返回值：直接删磁盘文件，删不掉也只打印日志，不抛异常

    规则：删完切片后，再查库里“还有没有切片的 saved_as 等于此名”，0 条才能安全删物理文件。
    因为存盘名由内容指纹决定，同一内容用两个文件名上传时磁盘上是同一个共享文件，
    只删其中一个文件名的切片时，另一个仍在引用，直接删会导致下载不到原文。
    """
    for name in saved_names:
        if not name:
            continue  # 跳过 None/空串：早期数据可能没 saved_as，os.path.join 遇 None 会崩
        # 按元数据 saved_as 查库里还有多少切片引用这个磁盘文件，0 条才能删
        if len(collection.get(where={"saved_as": name})["ids"]) == 0:
            path = os.path.join(UPLOAD_DIR, name)
            if os.path.exists(path):  # 文件可能早就被手动删了，exists 判断避免 os.remove 抛 FileNotFoundError
                try:
                    os.remove(path)
                except OSError as e:
                    # Windows 下文件被占用（如正在下载）会删失败。清理只是“顺手家务”，
                    # 失败不该让整个请求变 500，只记日志放过——最坏就是 uploads/ 多留一个无引用文件。
                    print(f"[清理] 物理文件删除失败（不影响主流程）: {name} {e}")


# 禁删名单：知识库演示样本，只允许“上传同名覆盖更新”，不允许删除（误删会导致 RAG 演示当场失效）。
# 拦在后端而非前端：后端才是真相之源，绕过页面直接调 DELETE 接口时前端限制形同虚设。
PROTECTED_FILES = {"公司制度.txt"}  # 用集合：in 判断更快，语义表示“一堆不重复的名字”


@app.delete("/api/files/{filename}")
async def delete_file(filename: str, _identity: dict = Depends(require_liang)):
    """从知识库删除一个文档：向量库里的切片 + uploads/ 里的物理文件一起清掉。

    路径参数 filename 是原文件名（非磁盘哈希名），FastAPI 从 URL 自动取出。
    返回：{"filename": ..., "deleted_chunks": 实际删掉的切片数}
    两件事必须一起做：只删切片会在磁盘留下无人引用的孤儿文件。
    """
    # 禁删名单检查放最前，命中直接 403（权限不足，区别于 404“资源不存在”：文件在，只是不许删）
    if filename in PROTECTED_FILES:
        # detail 同时给出替代方案（上传同名即可覆盖），而非冷冰冰的“不允许”
        raise HTTPException(status_code=403,
                            detail=f"「{filename}」是知识库的演示样本文件，不支持删除；需要更新内容时，上传同名文件即可自动覆盖旧版")
    before = collection.count()  # 记下删除前切片总数，删完相减得删除数（用于 404 判断和返回）
    # 删前须先捞出 saved_as：切片一删元数据就没了，再查不到该清哪个物理文件
    data = collection.get(where={"filename": filename}, include=["metadatas"])
    saved_names = {m.get("saved_as") for m in data["metadatas"]}  # 集合推导式：顺带完成去重
    collection.delete(where={"filename": filename})  # where = 按元数据条件批量删除（一次删掉该文件的所有切片）
    deleted = before - collection.count()
    if deleted == 0:
        # 一个切片都没删 = 库里没这个文件。返回 404 而非“删除成功 0 条”，前端才能准确提示。
        raise HTTPException(status_code=404, detail="知识库中没有这个文件")
    cleanup_saved_files(saved_names)  # 切片已删完，这时查“还有谁引用这个物理文件”才是准的
    print(f"知识库切片总数: {collection.count()}")
    return {"filename": filename, "deleted_chunks": deleted}


@app.get("/api/files/{filename}/download")
async def download_file(filename: str, identity: dict = Depends(get_identity)):
    """下载知识库里的原文件。

    链路：原文件名 → 查 Chroma 元数据拿到磁盘哈希名 → 返回该磁盘文件。
    绕这一道是因为磁盘存的是 "md5哈希.txt"，原文件名只在元数据里，须先查库翻译。
    """
    # limit=1：切片共享同一 saved_as，取一条即可
    data = collection.get(where={"filename": filename}, include=["metadatas"], limit=1)
    if not data["ids"]:
        raise HTTPException(status_code=404, detail="知识库中没有这个文件")
    # 私人文档权限：游客请求私人文件一律 403（亮哥放行）。放在 404 之后——先确认文件存在，再判权限。
    if bool(data["metadatas"][0].get("private", False)) and not identity["is_liang"]:
        raise HTTPException(status_code=403, detail="该文档为亮哥私人内容，游客无权下载")
    saved_as = data["metadatas"][0].get("saved_as")
    if not saved_as:
        # 早期数据无 saved_as 字段（那时还没做“保存原文件”）。如实告知“没存原文件、请重新上传”，
        # 而非返回空文件或报 500 —— 统一原则：宁可说清做不到，也不蒙混。
        raise HTTPException(status_code=404, detail="该文件为历史入库数据，未记录原文件，请重新上传")
    file_path = os.path.join(UPLOAD_DIR, saved_as)
    if not os.path.exists(file_path):
        # 元数据说文件在、磁盘却找不到（被手动删/目录被清）—— 数据不一致，也如实报
        raise HTTPException(status_code=404, detail="服务器上的原文件已丢失，请重新上传")
    # FileResponse 直接把磁盘文件作为响应体返回。filename 指定浏览器保存名 = 原文件名
    # （中文名自动 URL 编码），不传则浏览器用 URL 末段或哈希名保存，用户看不懂。
    return FileResponse(file_path, filename=filename)


# ===== 托管前端静态文件（必须放在本文件最末尾）=====
# 同源方案：前端 next build 静态导出到 frontend/out/，由本 FastAPI 进程一起托管，
# 一个容器一个端口，CORS 彻底消失。
#
# 必须在所有 @app.xxx 路由之后：Starlette 按注册顺序匹配，mount("/") 是“兜住所有剩余请求”的
# 通配挂载，放前面会把 /api/chat 之类全吞掉。
#
# html=True：访问目录自动返回其 index.html（访问 / 返回 /index.html）。
#
# 套 if os.path.isdir：本地开发前端跑在 3000、没有 frontend_out 目录，
# 不套判断本地启动会报错；目录不存在时静默跳过，同一份代码本地和线上都能跑。
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend_out")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    print(f"[前端托管] 已挂载静态文件目录: {os.path.abspath(FRONTEND_DIR)}")
else:
    print("[前端托管] 未找到 frontend_out 目录，跳过挂载（本地开发属正常情况）")
