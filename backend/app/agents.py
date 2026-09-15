# ===== Agent 定义（模型编排层）=====
# Agent = 能自己决定“要不要用工具、用哪个”的大模型应用，核心是“思考⇄行动”的 ReAct 循环：
# 模型输出“我要调某工具、参数xxx”的意图 → 代码执行 → 结果喂回 → 模型基于结果继续回答。
#
# 本文件放三套实现（学习对照，由 config.py 开关决定谁生效）：
#   TOOLS 手写版 / lc_agent+lc_executor LangChain 版 / lg_graph LangGraph 版（当前主力，支持流式）。
# 另有 MCP 外部工具加载器 get_mcp_tools、技能包工具 load_skill（原理见 skills.py）。
import os   # 读环境变量（API 密钥）
import sys  # sys.executable 拿当前 Python 解释器路径（启动 MCP 子进程用）

# ----- LangChain 相关 -----
from langchain_openai import ChatOpenAI                     # LangChain 封装的模型客户端，连 DeepSeek 改 base_url 即可
from langchain_core.tools import tool as langchain_tool     # @tool：把普通 Python 函数变成 Agent 可用的工具
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 提示词模板 + 历史消息占位符
# LangChain 1.x 把老的 Agent API 移进了 langchain-classic 兼容包，功能一样只是换包名
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
# ----- LangGraph 相关 -----
# 1.x 里 create_react_agent 的新家是 langchain.agents.create_agent，参数 prompt 改名 system_prompt；用 as 起回老名字
from langchain.agents import create_agent as create_react_agent
# MCP 让“外部工具服务”标准化接入大模型；下面的 adapter 把 MCP 工具自动转成 LangChain 工具对象，与本地工具无差别
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import USE_MCP                    # MCP 总开关
from .rag import search_knowledge_base         # 真正的检索能力（三套 Agent 共用）
from . import skills                           # 技能包加载器：list_skills（清单）+ load_skill（正文）

# ===== 助手人格设定 =====
# “身份层”提示词，每次对话无条件放进 system 消息。写成编号规则而非模糊形容词（“性格沉稳”模型执行很飘），
# 拆成可判定条目模型才守得住；最后一条必需，否则模型常否认自己有名字。（相邻字符串 Python 自动拼接）
PERSONA = (
    "你叫牛来（'牛来'就是你的名字），是亮哥（赵亮）的专属 AI 助手，性格沉稳可靠，像一位经验丰富的老秘书。\n"
    "说话规则：\n"
    "1. 始终称呼用户为'亮哥'；\n"
    "2. 结论先行：第一句话直接给答案，再补充细节和出处，不铺垫不客套；\n"
    "3. 语气专业简洁，可偶尔用 emoji 点缀气氛（每条回复最多 1-2 个），不过度；\n"
    "4. 不知道的事如实说明，绝不编造；\n"
    "5. 被问及'你叫什么名字'等身份问题时，直接回答自己叫牛来，不要否认或另起名字。"
)

# ===== 手写版工具说明书（Tool Calling 原始格式）=====
# 原理：把“有哪些工具”用 JSON 随请求发给模型；模型只返回“调用意图”（工具名+参数），执行权始终在我们代码手里。
# 三字段：name 工具名（代码靠它对号执行）；description 何时用/不用（直接决定调用准确率，负向约束同样重要）；
#         parameters 参数的 JSON Schema（query 为必填字符串）。
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

# ===== MCP 外部工具加载 =====
# 我们是 MCP 客户端，去连两个现成服务端（网页抓取 fetch、天气查询），把它们的工具接给模型用。
_mcp_client = None       # MCP 客户端对象（首次加载后常驻内存，后续复用）
# 🔴 工具缓存三状态必须严格区分（本模块最易踩的坑）：None=还没加载过 / []=加载失败（降级） / 有内容=成功。
# 若用 None 同时表示“没加载”和“失败”，失败后每次请求都会重拉子进程，卡住接口且永远好不了。
_mcp_tools_cache = None


async def get_mcp_tools():
    """加载 MCP 外部工具，返回可供 Agent 使用的工具对象列表。

    三个设计要点：懒加载（首次需要时才加载，避免拖慢启动或启动失败拖垮后端）、
    缓存（子进程只拉一次，后续复用）、降级（失败返回 []，不阻断主流程）。
    返回：工具列表（成功 5 个，失败或开关关闭时 []）。
    """
    global _mcp_client, _mcp_tools_cache  # 改模块级变量必须声明 global，否则只会创建同名局部变量
    if not USE_MCP:
        return []                        # 开关关闭：不加载
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache          # 已加载过（成功或失败都算）：返回缓存，不再拉子进程
    try:
        # 声明要连哪些 MCP 服务端：键是自取的名字，值说明怎么启动它
        _mcp_client = MultiServerMCPClient({
            "fetch": {
                # sys.executable = 当前运行后端的 Python 解释器完整路径；不能写死 "python"，否则可能找不到包
                "command": sys.executable,
                "args": ["-m", "mcp_server_fetch"],  # 等价于 python -m mcp_server_fetch
                "transport": "stdio",                # 通过子进程标准输入输出通信（本地工具标准做法）
            },
            "weather": {  # Open-Meteo 天气服务端：免费、无需 API key
                "command": sys.executable,
                "args": ["-m", "open_meteo_mcp"],
                "transport": "stdio",
            }
        })
        # 🔴 有几个服务端就开几个会话（嵌套 with），全打开后 get_tools() 才能拿到全部工具；
        # 少开一个不报错、只是静默少一批，很难发现。两个服务端合计 9 个工具。
        async with _mcp_client.session("fetch"):
            async with _mcp_client.session("weather"):
                all_tools = await _mcp_client.get_tools()
        # 工具精简（核心设计点）：天气服务端 8 个工具里时区/空气质量类与模型自身能力重叠，
        # 留着只增加决策噪音（工具越多越易选错），过滤后保留 5 个进模型清单：
        _mcp_tools_cache = [
            t for t in all_tools
            if t.name == "fetch" or t.name in {   # fetch 是网页抓取工具本体
                "get_current_weather",         # 当前天气
                "get_weather_byDateTimeRange", # 日期范围预报
                "get_weather_details",         # 详细天气（含预报）
                "get_current_datetime",        # 当前时间：模型算“明天”要靠它，故保留
            }
        ]
        print(f"[MCP] 已加载工具: {[t.name for t in _mcp_tools_cache]}")  # 启动观察点：打在后端终端
    except Exception as e:
        # 兜住所有异常：MCP 依赖外部子进程，缺包/超时都可能失败，记一笔并降级，服务照常跑
        print(f"[MCP] 加载失败，降级为普通模式: {e}")
        _mcp_tools_cache = []   # 缓存成 [] 而非 None：表示“试过了、失败了”，后续不再重试
    return _mcp_tools_cache


# ===== LangChain 版 Agent（对照实现）=====
# @tool 从函数签名和 docstring 自动生成工具说明书（对比手写版几十行 JSON）。
# 注意：docstring 不是普通注释，会被当提示词发给模型，措辞影响调用准确率。
@langchain_tool
def search_knowledge_base_lc(query: str) -> str:
    """检索企业知识库。当问题涉及公司制度、内部规定或用户个人档案时必须先调用本工具。"""
    # 真正干活的还是 rag.py 的 search_knowledge_base；三套 Agent 只是编排方式不同，底层检索同一个
    hits = search_knowledge_base(query)
    if not hits:
        # 没查到要返回“人话”而非空串：模型看到明确的“没检索到”才会如实告知，空串会被当成“工具坏了”或自己编
        return "知识库中没有检索到相关内容"
    # 拼成带编号文本 [1] (来自: 文件名)\n正文；编号是关键——提示词要求模型引用处标 [1]，前端才能对应引用卡片
    parts = [f"[{i + 1}] (来自: {meta.get('filename', '未知来源')})\n{doc}" for i, (doc, meta) in enumerate(hits)]
    return "\n\n".join(parts)   # enumerate 拿下标 i，i+1 让编号从 1 开始


# ===== 技能包工具（Skill 渐进式披露第二层）=====
# 分工：search_knowledge_base 提供“事实”（文档写了什么），load_skill 提供“章法”（某类任务怎么做）。
# 做成工具而非直接塞全文：手册上千 token，塞进系统提示词每次请求都要付这笔钱且分散注意力；
# 常驻的只有技能清单，模型判断需要时才调本工具把正文拉进上下文——即渐进式披露。
@langchain_tool
def load_skill(skill_name: str) -> str:
    """加载指定技能的完整操作手册。可用技能清单见系统提示词，skill_name 只能填清单里列出的名字。"""
    # 本函数只是壳：目录扫描、白名单校验（防路径穿越）、缓存都在 skills.py 里
    return skills.load_skill(skill_name)


# LangChain 版模型客户端：连的模型和 config.py 的 client 相同，只是 LangChain 要求用自己的封装类 ChatOpenAI
lc_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",  # DeepSeek 兼容 OpenAI 格式，ChatOpenAI 换地址即可连
)
# 提示词模板：定义每次请求发给模型的消息结构
lc_prompt = ChatPromptTemplate.from_messages([
    ("system", PERSONA),                      # 人格设定，固定不变
    MessagesPlaceholder("chat_history"),      # 占位符：调用时传入的历史消息插在这里
    ("human", "{input}"),                     # 用户当前这句话；{input} 被同名变量替换
    MessagesPlaceholder("agent_scratchpad"),  # Agent 草稿纸：中间思考/工具调用/结果记这里（固定写法，名字不能改）
])

# 组装 Agent：把模型、工具清单、提示词模板绑在一起，得到“会决定调工具的推理器”
lc_agent = create_tool_calling_agent(lc_llm, [search_knowledge_base_lc], lc_prompt)
# AgentExecutor 才是真正跑“决定→调工具→回填→再问”循环的执行器，lc_agent 只负责单步推理（推理与执行解耦）
lc_executor = AgentExecutor(agent=lc_agent, tools=[search_knowledge_base_lc],
                            verbose=True)  # verbose=True：把每步思考打印到后端终端，对比手写版时重点看这里

# ===== LangGraph 版 Agent（当前主力）=====
# LangGraph 把工作流建模成图：节点是步骤（调模型/执行工具），边是流转规则，“思考⇄行动”循环由图自动跑。
# 相比 LangChain 版的关键优势是支持流式：模型每吐一个字就能转发前端（打字机效果）。
#
# 核心取舍：知识库检索不交给模型决定，而由代码在进图前先做一次，把结果以“【编号资料】”写进提示词。理由：
#   ① 引用卡片能立刻发前端，不用等模型决策；② “没查到就如实告知”分支由代码控制，稳定可预期；③ 省一次决策调用。
#   一句话：确定性的活交给代码，生成性的活交给模型。但工具照常注册进图，模型可在代码检索没覆盖时自己再查复核。
#
# 系统提示词 = 人格 + 7 条回答规则，逐条对应一种场景（缺一条就出对应毛病）：有资料只用资料答+标编号引用、
# 资料空/无关时如实说没有、没覆盖时允许再查、闲聊直接答不调工具、实时网页用 fetch、天气走天气工具、
# 产品自身怎么用去加载技能手册。加规则 7 时必须同步给规则 2/4 加“产品问题按规则 7 处理”的例外，否则三条打架
# （产品问题在知识库必然查不到，会被规则 2/4 当闲聊挡回去）——提示词只能影响概率、不能上锁，规则间要留明确出口。

# ===== 技能清单（系统提示词里唯一的动态部分）=====
# 模块加载时扫一次技能目录，把每个技能的 name+description 拼成清单常驻系统提示词（渐进式披露第一层）。
# 好处：加新技能只要往 backend/skills/ 扔个文件夹，本文件不用改。代价：技能多了清单会撑大提示词，
# 到几十个时应改成“先按相关性筛选再注入”，不能无脑全塞。
_skill_list = skills.list_skills()
_skill_menu = "".join(f"- {s['name']}：{s['description']}\n" for s in _skill_list)

GRAPH_SYSTEM = (
    PERSONA + "\n\n回答规则：\n"
    "1. 用户提供【编号资料】时，只基于资料回答；引用了资料的句子末尾标注编号如[1][2]；资料没覆盖的就如实说明。\n"
    "2. 【编号资料】为空或和问题无关时，如实告知知识库中没有相关资料（不要反复调用 search_knowledge_base，不要编造）；"
    "但用户问的是本产品怎么用的情况按规则 7 处理。\n"
    "3. 用户的问题需要知识库里的事实、而【编号资料】显然没覆盖时，可以调用 search_knowledge_base 复核一次。\n"
    "4. 与知识库无关的常识和闲聊，直接回答，不要调用工具（询问本产品怎么用的除外，见规则 7）。\n"
    "5. 用户需要实时网页内容（某个网页的信息、最新内容）时，调用 fetch 工具抓取后回答；知识库问题和闲聊不要调用它。\n"
    "6. 用户询问某城市的天气时，调用天气工具（城市名用英文或拼音，如 Beijing）；需要判断“明天”等相对日期时先调用 get_current_datetime；天气问题不要用 fetch。\n"
    "7. 用户询问本产品自身怎么用时（如何上传/删除/下载文档、支持哪些格式和大小限制、为什么某个文件删不掉、"
    "回答里的[1][2]编号和来源卡片是什么、界面上的按钮在哪、架构开关怎么切），"
    "调用 load_skill(\"product-guide\") 取到产品手册，然后只按手册内容回答。"
    "这类问题【编号资料】为空是完全正常的，不要因此回答\"知识库中没有相关资料\"。\n"
    # 技能清单只在真扫到了技能时才拼进去，空清单不留一个空标题
    + ("\n可用技能清单（load_skill 的 skill_name 只能填下面列出的名字）：\n" + _skill_menu if _skill_menu else "")
)
# 构建 LangGraph 图对象（模块加载时构建一次，之后所有请求复用）。工具含本地知识库工具 + 技能包工具。
# MCP 工具异步加载（要 await），导入阶段拿不到，所以 main.py 在 MCP 加载成功后会现场重建带 MCP 工具的图，失败则用这张。
# 防死循环护栏 recursion_limit 不在这里传，而是每次调用时通过 config 传（新版 API 规定，见 main.py）。
# 已知局限：技能包工具只在 LangGraph 版生效（手写版循环写死只取第一个工具、LangChain 版模板用的是不含规则的 PERSONA）。
lg_graph = create_react_agent(lc_llm, [search_knowledge_base_lc, load_skill], system_prompt=GRAPH_SYSTEM)
