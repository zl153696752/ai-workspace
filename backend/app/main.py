# ===== FastAPI 应用组装 + 路由层（整个后端的入口）=====
# 本文件只做两类事：
#   ① 应用组装：创建 FastAPI 实例、配置跨域、定义请求体模型；
#   ② 六个 HTTP 接口：/api/chat（对话，含三套实现分支）+ 上传 / 清单 / 删除 / 下载。
# 业务逻辑不在这里：配置和资源在 config.py，检索与切片在 rag.py，Agent 编排在 agents.py，
# 本文件只负责“接收请求 → 调用它们 → 把结果按 HTTP 格式返回”。
from fastapi import FastAPI, UploadFile, File, HTTPException
# FastAPI：Web 框架；UploadFile / File：处理文件上传；
# HTTPException：抛出带状态码的错误，FastAPI 会自动把它转成 {"detail": "错误文案"} 的 JSON 响应
from fastapi.middleware.cors import CORSMiddleware   # 跨域中间件（解决前端域名/端口与后端不同时浏览器的拦截）
from fastapi.responses import StreamingResponse, FileResponse
# StreamingResponse：流式响应，打字机效果靠它；FileResponse：直接把磁盘文件作为响应体返回，供下载
from pydantic import BaseModel   # 请求体校验：定义好字段和类型，FastAPI 自动校验并生成接口文档
# 只有 MCP 工具加载成功时，才需要现场重建一张带 MCP 工具的图，所以这里也要能拿到构图函数
from langchain.agents import create_agent as create_react_agent
# LangGraph 流式输出会吐出多种类型的消息块，我们只想要“模型正文”那一种，靠这个类做类型判断过滤
from langchain_core.messages import AIMessageChunk
import hashlib  # 计算文件内容的 MD5 指纹（用于内容查重和文件命名）
import json      # 序列化 SSE 推送的数据、解析模型返回的工具参数
import os        # 路径拼接、判断文件是否存在

# 下面三行是本项目自己的模块（. 开头表示同一个包内），依赖方向是单向的：
# config（资源）← rag（检索能力）← agents（模型编排）← main（组装 + 路由），不会出现循环引用。
from .config import (client, collection, UPLOAD_DIR, ALLOWED_EXT, MAX_FILE_SIZE, MAX_TEXT_LENGTH,
                     USE_LANGCHAIN, USE_LANGGRAPH)
from .rag import search_knowledge_base, extract_text, split_text
from .agents import (PERSONA, TOOLS, GRAPH_SYSTEM, get_mcp_tools, search_knowledge_base_lc,
                     lc_llm, lc_executor, lg_graph, load_skill)

app = FastAPI(title="AI Workspace")   # title 会显示在自动生成的接口文档页（启动后访问 /docs）上

# 跨域配置：浏览器默认禁止 localhost:3000（前端）的页面请求 localhost:8000（后端），
# 因为端口不同就算跨域，请求会被浏览器拦下。后端必须显式声明“允许这个来源访问”，前端才能拿到响应。
# 上线后这里要换成真实域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 白名单：只允许前端开发服务器的地址
    allow_methods=["*"],   # 允许所有 HTTP 方法（GET / POST / DELETE ...）
    allow_headers=["*"],   # 允许所有请求头
)


class ChatRequest(BaseModel):
    """对话接口的请求体结构。

    前端每次发消息都会把完整对话历史一起发过来，因为后端不保存会话（是“无状态”的）：
    多轮对话的上下文由前端维护，后端重启也不会丢历史。
    messages 的格式是 OpenAI 约定的消息数组：[{"role": "user"/"assistant", "content": "..."}, ...]
    """
    messages: list   # 完整对话历史，最后一条就是用户刚发的这句话


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """流式对话接口（整个项目的核心接口）。

    返回方式是 SSE（Server-Sent Events，服务器推送事件）：
    一次 HTTP 请求不断开，服务器持续往前端推一条条“事件”，前端收到就实时渲染，这就是打字机效果的实现方式。
    （对比你熟悉的前端概念：SSE 类似只能服务器→客户端单向推送的 WebSocket，基于普通 HTTP，不用额外协议）
    每条事件的文本格式固定为两行 + 一个空行：
        event: 事件名
        data: JSON 字符串
    本项目共 5 种事件：
        tool    —— 告知前端“这次有没有调工具”（只有手写版会发，前端目前不消费，留着观察和以后扩展）
        sources —— 引用卡片数据（文件名 + 原文片段），前端渲染成可展开的来源卡片
        token   —— 模型正文的一小段文字，前端把它们拼接起来显示
        error   —— 出错了，前端弹出提示文案
        done    —— 本次回答结束，前端收尾（停止 loading 动画）

    内部按 config.py 的开关走三套实现之一：
        USE_LANGGRAPH = True → LangGraph 版（主力：代码先检索 + 流式生成）
        USE_LANGCHAIN = True → LangChain 版（框架自主决定调工具，一次性返回）
        两个都 False        → 手写版（纯 OpenAI SDK，自己写两次调用的循环）
    """
    messages = req.messages

    # ============================================================
    # 分支一：LangGraph 版（USE_LANGGRAPH = True，当前主力实现）
    # 完整流程：改写问题 → 代码先检索 → 先发引用卡片 → 图内流式生成回答
    # ============================================================
    if USE_LANGGRAPH:
        # ----- 第 1 步：把用户的口语化问题改写成适合检索的语句 -----
        # 为什么要改写：向量检索对“措辞”很敏感。用户接着上文问“那餐补呢”，
        # 这四个字里既没主语也没完整语义，转成向量后和文档里“加班餐补补贴标准”距离很远，
        # 过不了相关度阈值（实测距离 1.478 > 阈值 1.1），就会误判成“库里没资料”。
        # 改写成“餐补发放标准是什么？”这样一句独立完整、贴近文档措辞的话，距离降到 0.398，检索就能命中。
        # 做法：带上最近几轮历史单独调一次模型，让它补全指代、还原完整语义。
        # 只取最近 4 条（messages[-4:]）是为了控制这次调用的成本和延迟——解指代通常只需要上一两轮。
        try:
            rewrite = client.chat.completions.create(
                model="deepseek-v4-flash",
                # 这段 system 提示词是“改写员”的岗位说明，两个要点：
                #   ① 只输出检索语句本身，不要解释、不要回答问题（否则它的输出会污染检索词）；
                #   ② 遇到闲聊、创作类请求（“给我讲个故事吧”）必须原样输出、不许改写。
                # 第 ② 条必须显式写：这类请求对改写员来说是“超纲输入”，
                # 提示词只能影响模型的输出候选概率、不能给它上锁，不给它一个明确的放行出口，它可能自由发挥——
                # 曾经出现过用户说“讲个故事”，改写员直接编了一个完整故事出来的情况。
                messages=[{"role": "system", "content": "结合对话历史，把用户最新问题改写成一句独立完整的检索语句（贴近知识库文档措辞）。只输出检索语句本身，不要解释。若最新问题不是知识库查询类问题（闲聊、创作、讲故事等），原样输出该问题，不要改写、不要回答它。"}] + messages[-4:],
            )
            # choices[0].message.content 是模型返回的文本；or "" 兜住返回 None 的情况，再 strip 掉首尾空白
            rewritten = (rewrite.choices[0].message.content or "").strip()
        except Exception:
            rewritten = ""   # 改写失败（网络抖动、限流等）不影响主流程，下面用原话兜底
        # 改写结果非空就用它检索，否则退回用户原话——改写只是“锦上添花”，不能让它成为单点故障
        query = rewritten or messages[-1]["content"]
        print(f"[LangGraph 检索] 原话: {messages[-1]['content']} → 改写: {query}")  # 观察点：后端终端能直接看到改写效果，排查检索问题先看这里

        # ----- 第 2 步：加载 MCP 外部工具，决定本次用哪张图 -----
        mcp_tools = await get_mcp_tools()   # await：异步等待加载结果（首次会拉子进程，之后直接读缓存）
        # MCP 工具（网页抓取、天气）和本地工具（知识库检索、技能包加载）放进同一个列表交给图——
        # 在模型眼里它们没有任何区别，都是“一份工具说明书”，模型只看说明决定用哪个。
        # 加载成功就现场重建一张带全部工具的图；加载失败（返回 []）就复用 agents.py 里预建的普通图。
        # load_skill 在这里必须一并带上：预建的 lg_graph 里有它，重建时漏掉的话，
        # MCP 加载成功（也就是正常情况）反而会让技能功能失效，
        # 而且不报错——只是模型拿不到这个工具，产品问题又变回“知识库里没有相关资料”。
        active_graph = create_react_agent(lc_llm, [search_knowledge_base_lc, load_skill] + mcp_tools,
                                          system_prompt=GRAPH_SYSTEM) if mcp_tools else lg_graph

        # ----- 第 3 步：代码先检索，把结果整理成编号资料（给模型）+ 引用卡片（给前端）-----
        hits = search_knowledge_base(query)   # 不交给模型决定，这里直接查（设计取舍的理由见 agents.py 的 LangGraph 段注释）
        sources = []         # 给前端的引用卡片数据
        context_parts = []   # 给模型的编号资料文本
        seen = set()         # 去重用的集合，记录已处理过的（文件名, 正文）组合
        for doc, meta in hits:
            # 为什么要去重：切片之间有 50 字重叠，同一份内容可能同时命中两个相邻切片；
            # 不去重的话，前端会出现两张一模一样的引用卡片，模型也会看到重复资料。
            key = (meta.get("filename"), doc)
            if key in seen:
                continue     # 已经收过，跳过
            seen.add(key)
            # id 从 1 开始递增：这个数字就是回答里 [1][2] 的编号，也是前端卡片的序号，两边靠它对应
            sources.append({"id": len(sources) + 1, "filename": meta.get("filename", "未知来源"), "snippet": doc})
            # 同一份内容拼成给模型看的格式，编号必须和 sources 里的完全一致
            context_parts.append(f"[{len(sources)}] (来自: {meta.get('filename', '未知来源')})\n{doc}")
        # 一条都没命中时给个明确的“（无）”，而不是空字符串：
        # 提示词里规则 2 要求“资料为空时如实告知没有相关资料”，模型得看得见“为空”这个事实才会照做
        material = "\n\n".join(context_parts) if context_parts else "（无）"

        # ----- 第 4 步：定义流式生成器，边生成边推给前端 -----
        async def generate():
            # 先搞清 yield：带 yield 的函数叫“生成器”，它不一次性返回结果，
            # 而是每调一次吐一条数据、然后停在原地等下一次——刚好契合“模型说一句、前端显示一句”的需求。
            # StreamingResponse 会在背后不断拉取这个生成器，每拉到一条就立即发送给浏览器。
            #
            # 必须声明成 async def：函数体里有 async for（消费 LangGraph 的异步流），
            # 普通 def 里写 async for 会直接报语法错误（SyntaxError）。
            # StreamingResponse 对异步生成器原生支持，不用额外配置（下面手写版用同步生成器也照常工作）。

            # 先发引用卡片，后发正文——顺序很关键：
            # 前端收到 sources 事件时正文还是空的，可以先把卡片渲染出来，用户不用等回答写完才看到出处。
            # ensure_ascii=False：让中文按原样输出，否则会被转成 \uXXXX 转义序列（前端能解析，但抓包/终端里没法看）
            yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

            # 组装本次要发给图的消息列表：系统提示词 + 完整历史 + 当前问题（带着检索到的资料）。
            # 【用户问题】这里必须用用户原话 messages[-1]["content"]，不能用上面改写好的 query，原因：
            #   改写句是另一次轻量模型调用的产物，它的培养目标是“贴近文档措辞”——
            #   这对检索是优点，对回答是缺陷：改写过程会丢掉语气、丢掉细节，甚至可能改变原意。
            #   把改写句当用户问题回灌，等于偷偷替换了用户的提问。
            #   真实事故：用户说“给我讲个故事吧”，改写模型直接编了一个故事当检索词，
            #   这个故事被当成用户问题送给牛来，结果牛来回复“亮哥这故事讲得真好”——它以为故事是用户写的。
            #   至于指代问题（“那餐补呢”）不用担心：完整历史就在 messages[:-1] 里，回答模型自己解得开。
            lg_messages = [{"role": "system", "content": GRAPH_SYSTEM}] + messages[:-1] + [
                {"role": "user", "content": f"【编号资料】\n{material}\n\n【用户问题】\n{messages[-1]['content']}"}
            ]
            # 拼接顺序说明：messages[:-1] = 除最后一条外的全部历史（保持多轮上下文）；
            # 最后一条单独重写成“资料 + 问题”的格式，检索结果就是以这种方式进提示词的
            try:
                # astream = 异步流式执行图。stream_mode="messages" 表示：
                # 图运行过程中每产生一条消息就吐出来（模型每生成一个 token 就是一条），
                # 这就是打字机效果的数据来源——不管是查资料还是纯闲聊，都有逐字效果。
                # config 里的 recursion_limit=10 是防死循环护栏：
                # 图里“模型→工具→模型”是个循环，万一模型反复要求调工具，最多转 10 圈就强制中断，
                # 避免无限循环烧掉 token 又让请求永远挂着。新版 API 规定护栏在调用时传，不在构图时传。
                async for _chunk, metadata in active_graph.astream(
                        {"messages": lg_messages},
                            config = {"recursion_limit": 10}, stream_mode = "messages"):
                    # 流里会混着多种消息（工具调用消息、工具返回消息等），
                    # 只有 AIMessageChunk 且 content 非空的才是“模型正在说的正文”，其余全部丢弃。
                    # 不过滤的话，工具调用的中间过程（一大段 JSON）会被当成正文显示给用户。
                    if isinstance(_chunk, AIMessageChunk) and _chunk.content:
                        yield f"event: token\ndata: {json.dumps({'content': _chunk.content}, ensure_ascii=False)}\n\n"
            except Exception:
                # 生成过程中出错（模型服务超时、限流、图执行异常等）：
                # 此时 HTTP 响应头已经发出去了，不能再改状态码，只能通过 SSE 事件告诉前端。
                yield f"event: error\ndata: {json.dumps({'message': '模型服务暂时不可用，请稍后再试'}, ensure_ascii=False)}\n\n"
            # 无论成功失败都要发 done：前端靠它结束 loading 状态，漏发会让界面一直转圈
            yield "event: done\ndata: {}\n\n"

        # media_type="text/event-stream" 是 SSE 的标准 MIME 类型，前端靠它识别这是个流式响应
        return StreamingResponse(generate(), media_type="text/event-stream")

    # ============================================================
    # 分支二：LangChain 版（USE_LANGCHAIN = True）
    # 特点：“决定调不调工具 → 执行工具 → 结果回填 → 再生成回答”这整个循环全在框架内部完成，
    #       我们只调一次 invoke 就拿到最终答案（对比下面手写版的两次显式调用）。
    #       代价是没法流式：必须等循环全部跑完才能返回，所以这个分支没有打字机效果，
    #       整段回答一次性推给前端。
    # ============================================================
    elif USE_LANGCHAIN:
        def generate():   # 这个分支里没有 async for，用普通 def 即可
            try:
                result = lc_executor.invoke({
                    # invoke 的入参是一个字典，键名要和 agents.py 里提示词模板的占位符一一对应：
                    "input": messages[-1]["content"],  # 对应模板里的 {input}：用户当前这句话
                    "chat_history": messages[:-1],     # 对应 MessagesPlaceholder("chat_history")：其余作为历史
                                                       # 直接传字典数组即可，框架会自动转成它内部的消息对象
                })
                # result["output"] 是 Agent 循环跑完后的最终回答文本
                #（中间的思考过程由 verbose=True 打印到后端终端，不会进回答）
                yield f"event: token\ndata: {json.dumps({'content': result['output']}, ensure_ascii=False)}\n\n"
            except Exception:
                yield f"event: error\ndata: {json.dumps({'message': '模型服务暂时不可用，请稍后再试'}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ============================================================
    # 分支三：手写版（两个开关都关掉时走这里）
    # 不依赖任何 Agent 框架，纯用 OpenAI SDK 自己实现 Tool Calling 的完整循环。
    # 一共两次模型调用：
    #   调用① 决策：把工具说明书发给模型，问它“这问题你要不要用工具？”
    #   调用② 生成：如果用了工具，把工具执行结果回填进对话，再让模型基于结果写正式回答
    # 这套流程就是所有 Agent 框架底层在做的事，手写一遍才知道框架帮我们省了什么。
    # ============================================================

    # ----- 调用①：决策调用（注意：这里故意不开流式）-----
    # 原因：必须先拿到模型的完整返回，才能判断它是“想调工具”（返回 tool_calls 字段）
    # 还是“直接回答”（返回 content 字段）。流式是边收边判，判不了；也没必要——这次调用返回的内容很短。
    base_messages = [{"role": "system", "content": PERSONA}] + messages
    # system 消息放人格设定，后面拼上完整对话历史（历史里的 role 已经是 user/assistant，直接可用）
    try:
        decision = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=base_messages,
            tools=TOOLS,  # 关键参数：把工具说明书交给模型，用不用由它自己决定。
                          # 不传这个参数，模型只会凭自己的知识回答（企业文档它不可能知道，只能编）
        )
    except Exception:
        # 第一次调用就失败（网络/密钥/限流问题）：此时还没开始流式响应，可以正常抛 HTTP 错误，
        # FastAPI 会转成 {"detail": "..."} 的 JSON 响应，前端的 !res.ok 分支会自动接住并把文案弹给用户。
        raise HTTPException(status_code=500, detail="模型服务暂时不可用，请稍后再试")

    # choices[0].message = 模型返回的那条消息对象（可能含 content，也可能含 tool_calls）
    choice = decision.choices[0].message
    # getattr(对象, "属性名", 默认值)：安全地取属性，属性不存在时返回 None 而不是抛异常。
    # 用它是为了兼容不同 SDK 版本——有的版本没有 tool_calls 字段，直接 choice.tool_calls 会报错。
    # tool_calls 为 None 或空 = 模型决定不调工具、直接回答。
    tool_calls = getattr(choice, "tool_calls", None)
    print(f"[工具决策] {'调用 ' + tool_calls[0].function.name if tool_calls else '直接回答'}")  # 观察点：后端终端直接看到模型的决策结果

    sources = []                 # 引用卡片数据（只有调了工具且检索到内容时才会有值）
    final_messages = base_messages   # 第二次调用要用的消息列表；没调工具时它就是最终列表（不用再加东西）
    if tool_calls:
        # ----- 模型决定要查知识库：解析参数 → 执行工具 → 结果回填对话 -----
        call = tool_calls[0]   # 本项目只注册了一个工具，取第一个即可；多工具场景要循环处理每一项
        # call.function.arguments 是模型生成的参数，但它是一个 JSON **字符串**（不是字典），
        # 必须用 json.loads 反序列化才能取值。模型偶尔会生成格式错误的 JSON，
        # 那种情况会在下面抛异常，被最外层的错误兜底接住。
        query = json.loads(call.function.arguments)["query"]
        # 执行权在我们的代码手里：模型只说“我要查，查这个词”，真正去查的是下面这行
        hits = search_knowledge_base(query)

        # 整理检索结果：编号资料（给模型）+ 引用卡片（给前端），去重逻辑与上面 LangGraph 分支完全一致
        context_parts = []
        seen = set()
        for doc, meta in hits:
            key = (meta.get("filename"), doc)
            if key in seen:
                continue     # 重复内容跳过（切片重叠区可能导致同一内容命中两次）
            seen.add(key)
            sources.append({"id": len(sources) + 1, "filename": meta.get("filename", "未知来源"), "snippet": doc})
            context_parts.append(f"[{len(sources)}] (来自: {meta.get('filename', '未知来源')})\n{doc}")

        # 工具执行结果要写成一段“给模型看的说明文字”，两种情况措辞不同：
        if sources:
            # 查到了：把资料按编号列出，并明确要求模型“基于以上内容回答 + 标注编号 + 没有的如实说明”。
            # 这三句要求缺一不可：不写“基于以上内容”模型会掺自己的知识，
            # 不写“标注编号”前端卡片对不上号，不写“如实说明”模型会硬编一个答案。
            tool_result = (
                    "以下是检索到的知识库内容，按编号排列：\n\n"
                    + "\n\n".join(context_parts)
                    + "\n\n请基于以上内容回答。在引用了资料的句子末尾标注编号如 [1]；资料中没有的就如实说明。"
            )
        else:
            # 没查到：直接命令模型如实告知。这句提示词是“不编造”的关键防线——
            # 如果只回一个空结果，模型很可能凭训练记忆编出一套公司制度来。
            tool_result = "知识库中没有检索到和问题相关的内容。请如实告知用户知识库中没有相关资料。"

        # 按 OpenAI 的 Tool Calling 协议，回填时必须追加两条消息，且顺序不能颠倒：
        final_messages = base_messages + [
            # 第一条：把模型自己那条“调用意图”消息原样传回去。
            # OpenAI SDK 支持直接传原始消息对象（它含 tool_calls 字段），不用手工拼字典。
            # 少了这一条，后面的 tool 消息会被判为非法（协议要求 tool 消息必须紧跟在带 tool_calls 的 assistant 消息之后）。
            choice,
            # 第二条：工具执行结果。tool_call_id 必须和上面那条意图里的 call.id 完全一致，
            # 模型靠这个 id 知道“这是我刚才那次调用的返回值”（一次调多个工具时靠它区分）。
            {"role": "tool", "tool_call_id": call.id, "content": tool_result}
        ]

    # ----- 调用②：生成回答，并流式推给前端 -----
    def generate():
        # 先发一个“决策透明”事件：告诉前端这次有没有调工具。
        # 前端目前只认 sources/token/error/done，收到不认识的事件名会自动忽略，
        # 所以这个事件只用于调试观察和以后扩展（比如界面上显示“已查询知识库”）。
        # bool(tool_calls)：tool_calls 可能是 None 或列表，统一转成 True/False 再序列化
        yield f"event: tool\ndata: {json.dumps({'called': bool(tool_calls)}, ensure_ascii=False)}\n\n"
        if sources:
            yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"   # 有引用卡片就先发，前端可以立刻渲染出处

        try:
            if tool_calls:
                # 走了工具路径：需要第二次调用模型，让它基于工具结果写正式回答。
                # 这次开流式（stream=True）：回答内容较长，逐字推送体验好得多。打字机效果只在这条路径上出现。
                response = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=final_messages,   # 含工具结果的完整消息列表
                    stream=True                # 开启流式：返回一个可迭代对象，模型每生成一小段就产出一个 chunk
                )
                for chunk in response:   # 同步 for 循环逐块消费（这个分支没有异步流，所以 generate 是普通 def）
                    # delta = 本次增量内容。第一个 chunk 通常只含 role 没有 content，所以要判空，
                    # 否则会把 None 序列化进去，前端拼出一堆 "null"。
                    if chunk.choices[0].delta.content:
                        piece = chunk.choices[0].delta.content
                        yield f"event: token\ndata: {json.dumps({'content': piece}, ensure_ascii=False)}\n\n"
            else:
                # 没调工具（闲聊、常识问题）：调用① 的返回里已经含完整回答了，不用再调一次模型，
                # 直接把 choice.content 整段推出去。代价是这条路径没有打字机效果（内容是一次性拿到的），
                # 这是“少一次调用换更快响应”的权衡。
                # or ''：content 可能是 None（比如模型只返回了工具意图），兜底成空串避免序列化出 null
                yield f"event: token\ndata: {json.dumps({'content': choice.content or ''}, ensure_ascii=False)}\n\n"
        except Exception:
            # 流已经开始，改不了 HTTP 状态码，只能用 SSE 事件通知前端
            yield f"event: error\ndata: {json.dumps({'message': '模型服务暂时不可用，请稍后再试'}, ensure_ascii=False)}\n\n"

        yield "event: done\ndata: {}\n\n"   # 收尾事件：前端靠它关闭 loading

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """文档上传入库接口：校验 → 提取文字 → 切片 → 查重 → 存盘 → 写入向量库。

    参数 file 由 FastAPI 自动从 multipart/form-data 请求里解析出来（File(...) 表示该字段必填），
    它就是前端 <input type="file"> 传过来的那个文件。
    返回：{"filename": 原文件名, "saved_as": 磁盘上的实际文件名, "size": 字节数,
           "chunks": 切了多少片, "overwritten": 是否覆盖了同名旧文件}
    前端靠 overwritten 分流提示文案（“上传成功” vs “已覆盖旧版本”）。
    """
    # ----- 1. 校验文件类型（白名单）-----
    # os.path.splitext 把文件名拆成（主名, 后缀），如 "制度.TXT" → ("制度", ".TXT")；
    # [1] 取后缀，.lower() 转小写，这样 ".TXT" 和 ".txt" 都能通过校验。
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        # 400 = 客户端错误（用户传了不该传的东西）。detail 里的文字会被前端直接弹给用户，
        # 所以要写人话、带上具体原因，不能只丢一个错误码。
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # ----- 2. 大小闸门（第一道，读内容之前）-----
    # file.size 是 FastAPI 从请求里拿到的字节数。此时文件内容还没读进内存，
    # 超限直接拒收，避免一个大文件把内存吃满。
    if file.size is not None and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件太大（超过 5MB），请压缩或拆分后再上传")

    # ----- 3. 读取内容 + 计算内容指纹 -----
    content = await file.read()   # 异步读出全部字节（await 期间不阻塞其他请求，这是 async 的意义）
    # 第二道大小检查：某些客户端/代理不传 size，此时 file.size 是 None，上面那道闸拦不住，
    # 只能等读完后用 len(content) 兜底
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件太大（超过 5MB），请压缩或拆分后再上传")
    # MD5 指纹：同样的文件内容算出来的哈希值必然相同，内容差一个字节哈希就完全不同。
    # 它有两个用途：① 内容查重（下面第 5 步）；② 作为磁盘文件名（下面第 7 步）。
    content_hash = hashlib.md5(content).hexdigest()

    # ----- 4. 提取文字 + 切片（先解析，后落盘）-----
    # 顺序很重要：先确认这文件真能提取出文字，再写磁盘和入库。
    # 反过来的话，一个解析不了的文件会在 uploads/ 里留下永远用不上的垃圾文件。
    try:
        text = extract_text(content, ext)   # 传的是内存里的字节，不是磁盘路径（文件此时还没存盘）
    except Exception:
        # PDF 加密、文件损坏、格式伪装（后缀是 .pdf 实际不是）都会让解析抛异常。
        # 这里转成 400 + 明确文案，而不是让异常裸奔成 500（500 前端只能显示“服务器错误”，用户不知道该怎么办）。
        raise HTTPException(status_code=400, detail="文件解析失败：文件可能已损坏、加密或格式异常，请检查后重新上传")
    # 文字量闸门：真正决定入库成本的是文字量而不是文件体积（理由见 config.py 的说明）
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="文件文字内容过多（超过 30 万字），请拆分成多份上传")
    chunks = split_text(text)   # 切成 300 字/片、重叠 50 字的小段
    if not chunks:
        # 解析成功但一个字都没有 = 扫描版/图片型 PDF（内容是图片，没有文字层）。
        # 支持它需要 OCR（图片文字识别），本项目不做，所以如实告知用户。
        raise HTTPException(status_code=400, detail="未能提取到文字，可能是图片型/扫描件文件，暂不支持")

    # ----- 5. 内容查重：完全相同的文件禁止重复上传 -----
    # 切片的入库 id 由“内容指纹 + 序号”组成：同一份内容的第 0 片 id 永远是 "哈希-0"。
    # 这叫内容寻址命名——id 由内容决定，而不是由文件名或上传时间决定。
    ids = [f"{content_hash}-{i}" for i in range(len(chunks))]
    # 拿这批 id 去库里查：查到了就说明这份内容以前上传过（哪怕当时用的是另一个文件名）。
    existing = collection.get(ids=ids, include=["metadatas"])
    if existing["ids"]:
        # 409 = 资源冲突。告诉用户库里已有的那个文件叫什么，避免他一头雾水
        #（“我明明没传过这个名字啊”——因为之前是用别的名字传的同一份内容）。
        old_name = existing["metadatas"][0].get("filename", file.filename)
        raise HTTPException(status_code=409,
                            detail=f"该文件已上传，禁止重复上传（库中已有内容完全相同的文件：{old_name}）")
    # 这一步拦在写盘之前，所以被拒的上传不会留下任何副作用（磁盘不多文件、库里不多切片）

    # ----- 6. 同名覆盖：文件名相同但内容不同 = 新版本 -----
    # 场景：用户改了制度文档重新上传。如果不先清掉旧版，新旧两版的切片会同时留在库里，
    # 检索时两版一起被捞出来塞进提示词，模型就会看到互相矛盾的规则（旧版年假 3 天、新版 5 天）。
    overwritten = False   # 标记本次是“新上传”还是“覆盖旧版”，返回给前端用于分流提示文案
    old = collection.get(where={"filename": file.filename}, include=["metadatas"])   # 按元数据里的 filename 查出旧版的所有切片
    if old["ids"]:
        # 先收集旧版的磁盘文件名集合（用 set 去重：一个文件的所有切片共享同一个 saved_as）
        old_saved = {m.get("saved_as") for m in old["metadatas"]}
        collection.delete(ids=old["ids"])   # 删向量库里的旧切片
        # 再清磁盘上的旧文件。顺序不能反：必须等切片删完再查引用，
        # 因为 cleanup_saved_files 内部会判断“库里还有没有切片引用这个文件”（防止误删共享文件）。
        cleanup_saved_files(old_saved)
        overwritten = True
        print(f"[覆盖] {file.filename} 旧版 {len(old['ids'])} 个切片已清理")

    # ----- 7. 所有校验都过了，才把原文件写到磁盘 -----
    # 存盘文件名用“内容指纹 + 原后缀”，而不是用户给的文件名。好处：
    #   ① 天然避免文件名冲突（两份不同内容永远不会撞名）；
    #   ② 避开中文/特殊字符文件名在各系统间的编码问题；
    #   ③ 用户看到的原文件名保存在向量库元数据的 filename 字段里，下载时再还原。
    save_name = f"{content_hash}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    with open(save_path, "wb") as f:   # "wb" = 以二进制写模式打开；with 保证写完自动关闭文件
        f.write(content)

    # ----- 8. 写入向量库 -----
    # upsert = update + insert：id 已存在就更新，不存在就插入（比 insert 更安全，不会因为 id 重复报错）。
    # 走到这一步的要么是全新文件，要么是已经清掉旧版的同名新内容，所以不会有重复数据。
    collection.upsert(
        documents=chunks,   # 切片正文：Chroma 会自动把每片文本转成向量存起来（这一步叫 embedding）
        ids=ids,            # 每片的主键，和 documents 一一对应
        # 元数据：给每个切片都记上“来自哪个文件、磁盘上存成了什么名字”。
        # [字典] * len(chunks) 是把同一个字典复制成 N 份（N = 切片数），每片一份。
        # 元数据不参与向量计算，但可以用来过滤查询（比如按 filename 查出某文件的所有切片），
        # 也是前端引用卡片显示文件名的数据来源。
        metadatas=[{"filename": file.filename, "saved_as": save_name}] * len(chunks),
    )
    print(f"知识库切片总数: {collection.count()}")   # 观察点：后端终端能看到入库后库里的切片总数

    return {"filename": file.filename, "saved_as": save_name, "size": len(content),
            "chunks": len(chunks), "overwritten": overwritten}


@app.get("/api/files")
async def list_files():
    """知识库文件清单接口：返回库里有哪些文件、各切了多少片（前端侧边栏的文件列表靠它）。

    返回：{"files": [{"filename": "公司制度.txt", "chunks": 3}, ...]}
    数据来源是 Chroma 的元数据，不是前端本地记录、也不是扫描 uploads/ 目录——
    向量库才是“知识库里到底有什么”的唯一真相之源。
    好处：换浏览器、清缓存、重启服务，清单都不会错。
    """
    if collection.count() == 0:
        return {"files": []}   # 空库直接返回空列表
    # collection.get 不带条件 = 取出所有切片的元数据
    #（只取 metadatas，不取 documents 正文，因为前端只要文件名和数量，少传很多数据）
    data = collection.get(include=["metadatas"])
    # 库里存的是“切片”，一个文件对应多个切片，而前端要的是“文件”维度的清单，
    # 所以要按 filename 聚合：统计每个文件名出现了多少次 = 它的切片数。
    counter = {}   # {文件名: 切片数}
    for meta in data["metadatas"]:
        name = meta.get("filename", "未知来源")   # .get 带默认值：早期数据可能没这个字段，不至于报错
        counter[name] = counter.get(name, 0) + 1  # 字典计数的标准写法：没这个键就当 0，然后 +1
    # 把字典转成前端好遍历的数组格式（字典在 JSON 里是对象，前端要 Object.keys 才能遍历，不如直接给数组）
    return {"files": [{"filename": n, "chunks": c} for n, c in counter.items()]}


def cleanup_saved_files(saved_names):
    """清理 uploads/ 目录里已经没人引用的物理文件（覆盖上传、删除文件两条路都用它）。

    参数 saved_names：一批磁盘文件名（就是入库时的 save_name，形如 "md5哈希.txt"）
    无返回值：直接删磁盘文件，删不掉也只打印日志，不抛异常

    为什么不能“删切片时顺手把文件删了”：
    存盘文件名是由内容指纹决定的，所以同一份内容即使用两个不同的文件名上传过，
    它们在磁盘上也是同一个文件（共享）。这时候只删掉其中一个文件名的切片，
    另一个文件名的切片还在引用这个磁盘文件，直接删就会导致它下载不到原文。
    所以规则是：删完切片之后，再去库里查“还有没有任何切片的 saved_as 等于这个名字”，
    查到 0 条才说明真没人用了，这时才能安全删物理文件。
    """
    for name in saved_names:
        if not name:
            continue   # 跳过 None / 空串：早期数据可能没有 saved_as 字段，os.path.join 遇到 None 会直接崩
        # 按元数据 saved_as 查库里还有多少切片引用这个磁盘文件，0 条才能删
        if len(collection.get(where={"saved_as": name})["ids"]) == 0:
            path = os.path.join(UPLOAD_DIR, name)
            if os.path.exists(path):   # 文件可能早就被手动删了，exists 判断避免 os.remove 抛 FileNotFoundError
                try:
                    os.remove(path)
                except OSError as e:
                    # Windows 下文件被其他进程占用（比如正在被下载）时删除会失败。
                    # 清理只是“顺手做的家务”，失败不该让整个上传/删除请求变成 500，
                    # 所以这里只记一条日志就放过——最坏结果是 uploads/ 里多留一个没人引用的文件。
                    print(f"[清理] 物理文件删除失败（不影响主流程）: {name} {e}")


# 禁删名单：知识库的演示样本文件，只允许“上传同名文件覆盖更新”，不允许删除。
# 为什么要有这个名单：这个文件是整个项目的演示数据，打开页面点一下推荐问题就能问出结果。
# 一旦被误删，知识库变空，所有 RAG 功能当场演示不出来。
# 为什么拦在后端而不是前端（前端把删除按钮藏起来）：后端才是真相之源，
# 绕过页面直接用 Postman/curl 调 DELETE 接口，前端的限制形同虚设，只有后端拦住才算真拦住。
PROTECTED_FILES = {"公司制度.txt"}   # 用集合而不是列表：in 判断更快，语义上也表示“一堆不重复的名字”


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """从知识库删除一个文档：向量库里的切片 + uploads/ 里的物理文件一起清掉。

    路径参数 filename 是用户看到的原文件名（不是磁盘上的哈希名），FastAPI 会从 URL 里自动取出来传进函数。
    返回：{"filename": ..., "deleted_chunks": 实际删掉的切片数}
    两件事必须一起做：只删切片的话，磁盘上会留下永远没人引用的孤儿文件，越攒越多。
    """
    # 禁删名单检查放在最前面，命中直接 403
    #（403 = 权限不足，和 404“资源不存在”语义不同：文件确实存在，只是不允许你删）
    if filename in PROTECTED_FILES:
        # detail 里同时告诉用户替代方案（上传同名文件即可覆盖），而不是一句冷冰冰的“不允许”
        raise HTTPException(status_code=403,
                            detail=f"「{filename}」是知识库的演示样本文件，不支持删除；需要更新内容时，上传同名文件即可自动覆盖旧版")
    before = collection.count()   # 记下删除前的切片总数，删完相减就知道删了多少（用于 404 判断和返回给前端）
    # 删除之前必须先把 saved_as 捞出来：切片一删，元数据就没了，再也查不到该清哪个物理文件
    data = collection.get(where={"filename": filename}, include=["metadatas"])
    saved_names = {m.get("saved_as") for m in data["metadatas"]}   # 集合推导式：顺带完成去重
    collection.delete(where={"filename": filename})   # where = 按元数据条件批量删除（一次删掉该文件的所有切片）
    deleted = before - collection.count()
    if deleted == 0:
        # 一个切片都没删掉 = 库里根本没有这个文件。返回 404 而不是“删除成功 0 条”，
        # 前端才能给出准确提示（用户可能是刷新前文件已被其他地方删掉了）
        raise HTTPException(status_code=404, detail="知识库中没有这个文件")
    cleanup_saved_files(saved_names)   # 切片已删完，这时查“还有谁引用这个物理文件”才是准的
    print(f"知识库切片总数: {collection.count()}")
    return {"filename": filename, "deleted_chunks": deleted}


@app.get("/api/files/{filename}/download")
async def download_file(filename: str):
    """下载知识库里的原文件。

    链路：用户给的原文件名 → 查 Chroma 元数据拿到磁盘上的哈希文件名 → 返回那个磁盘文件。
    为什么要绕这一道：磁盘上存的是 "md5哈希.txt"，用户不知道也不该知道这个名字，
    原文件名只记录在向量库元数据里，所以必须先查库翻译一次。
    """
    # limit=1：一个文件的所有切片共享同一个 saved_as，取一条就够了，不用把几十条元数据全捞出来
    data = collection.get(where={"filename": filename}, include=["metadatas"], limit=1)
    if not data["ids"]:
        raise HTTPException(status_code=404, detail="知识库中没有这个文件")
    saved_as = data["metadatas"][0].get("saved_as")
    if not saved_as:
        # 早期版本入库的数据没有 saved_as 字段（那时还没做“保存原文件”这个功能）。
        # 这种情况如实告诉用户“没存原文件、请重新上传”，而不是返回一个空文件或报 500——
        # 全站错误提示的统一原则：宁可说清楚做不到，也不蒙混过去。
        raise HTTPException(status_code=404, detail="该文件为历史入库数据，未记录原文件，请重新上传")
    file_path = os.path.join(UPLOAD_DIR, saved_as)
    if not os.path.exists(file_path):
        # 元数据说文件在，磁盘上却找不到（被手动删了、uploads 目录被清了）——数据不一致，也要如实报
        raise HTTPException(status_code=404, detail="服务器上的原文件已丢失，请重新上传")
    # FileResponse 直接把磁盘文件作为响应体返回，不用自己读字节再拼响应。
    # filename 参数指定浏览器保存时用的名字 = 用户最初上传的原文件名（中文名会自动做 URL 编码），
    # 不传的话浏览器会用 URL 末段或哈希名保存，用户看不懂。
    return FileResponse(file_path, filename=filename)
