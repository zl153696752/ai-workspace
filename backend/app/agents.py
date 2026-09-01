# ===== Agent 定义（模型编排层）=====
# 先搞清一个词：Agent（智能体）= 能自己决定“要不要用工具、用哪个工具”的大模型应用。
#   普通模型调用是“问一句答一句”；
#   Agent 是模型看完问题后先输出一个“我要调 search_knowledge_base，参数是xxx”的意图，
#   我们的代码执行完工具、把结果喂回去，模型再基于结果继续回答——
#   这个“思考 ⇄ 行动”的循环（行业叫 ReAct：Reasoning + Acting）就是 Agent 的本质。
#
# 本文件放了三套 Agent 实现（学习对照用，同时只有一套生效，由 config.py 的开关决定）：
#   1. TOOLS                —— 手写版：自己写工具说明书 JSON，自己写调用循环（循环代码在 main.py）
#   2. lc_agent/lc_executor  —— LangChain 版：框架接管“决定→执行→回填”整个循环
#   3. lg_graph              —— LangGraph 版：图结构编排，支持流式逐字输出（当前主力）
# 另外还有一件事：MCP 外部工具加载器 get_mcp_tools，把外部进程提供的工具接进来给模型用。
import os   # 读环境变量（API 密钥）
import sys  # 用 sys.executable 拿当前 Python 解释器的路径（启动 MCP 子进程要用）

# ----- LangChain 相关 -----
from langchain_openai import ChatOpenAI                     # LangChain 封装的模型客户端，连 DeepSeek 也是用它（改 base_url 即可）
from langchain_core.tools import tool as langchain_tool     # @tool 装饰器：把普通 Python 函数一键变成 Agent 能用的工具
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 提示词模板 + “此处插入历史消息”的占位符
# LangChain 1.x 的版本变动：老的 Agent API（create_tool_calling_agent / AgentExecutor）被移出了主包，
# 移进了 langchain-classic 兼容包。所以这里从 langchain_classic 导入，功能完全一样，只是换了个包名。
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
# ----- LangGraph 相关 -----
# LangGraph 提供预构建的 ReAct Agent（上面说的“思考⇄行动”循环开箱即用，不用自己写 if 判断）。
# 版本坑：老写法 from langgraph.prebuilt import create_react_agent 在 1.x 已被弃用，
# 新家在 langchain.agents.create_agent，参数名也从 prompt 改成了 system_prompt。
# 这里用 as 起回老名字，下面代码保持熟悉的叫法。
from langchain.agents import create_agent as create_react_agent
# MCP：Model Context Protocol（模型上下文协议），让“外部工具服务”标准化接入大模型的协议。
# 好处：工具由独立进程/服务提供，我们的项目不用把“抓取网页”“查天气”的代码写在自己仓库里，
# 任何遵守该协议的工具都能即插即用。下面这个 adapter 的作用：把 MCP 工具自动转成 LangChain 工具对象，
# 转换后模型用起来和本地工具没有任何区别。
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import USE_MCP                    # MCP 总开关
from .rag import search_knowledge_base         # 真正的检索能力（三套 Agent 共用同一个）

# ===== 助手人格设定 =====
# 这是“身份层”提示词：每次对话都无条件放进 system 消息，与知识库检索结果无关。
# 为什么写成编号规则而不是一句模糊描述：“性格沉稳”这类形容词模型执行得很飘，
# 拆成可判定的具体条目（称呼谁、第一句说什么、emoji 用几个）模型才守得住。
# 最后一条是必需的：不写显式指令时，模型常会否认自己有名字（“我是 AI 助手，没有名字”）。
# 另外：这里用括号包着多个字符串相邻写的形式，Python 会自动把它们拼成一个长字符串（不用写 + 号）。
PERSONA = (
    "你叫牛来（'牛来'就是你的名字），是亮哥（赵亮）的专属 AI 助手，性格沉稳可靠，像一位经验丰富的老秘书。\n"
    "说话规则：\n"
    "1. 始终称呼用户为'亮哥'；\n"
    "2. 结论先行：第一句话直接给答案，再补充细节和出处，不铺垫不客套；\n"
    "3. 语气专业简洁，可偶尔用 emoji 点缀气氛（每条回复最多 1-2 个），不过度；\n"
    "4. 不知道的事如实说明，绝不编造；\n"
    "5. 被问及'你叫什么名字'等身份问题时，直接回答自己叫牛来，不要否认或另起名字。"
)

# ===== 手写版工具说明书（Tool Calling 的原始格式）=====
# Tool Calling 的原理：我们把“有哪些工具可用”用 JSON 描述好，随请求一起发给模型；
# 模型不执行任何代码，它只返回一个结构化的“调用意图”（工具名 + 参数），
# 真正执行工具的是我们自己的代码，执行完再把结果发回给模型。
# 一句话记住：模型负责“决定用不用”，代码负责“实际去做”——执行权始终在我们手里，这是安全边界。
#
# 下面这段 JSON 就是给模型看的“岗位说明书”，三个字段各有作用：
#   name        —— 工具名，模型返回调用意图时会带上它，我们的代码靠它对号执行
#   description —— 什么时候该用、什么时候不该用。这段措辞直接决定模型的调用准确率，是工具设计的核心：
#                  写得太宽（“有问题就调用”）会导致闲聊也去查库，白耗一次检索；
#                  写得太窄会漏查，模型凭自己的记忆编造公司制度。
#                  注意这里明确写了“无关的常识问题和闲聊不要调用”——负向约束和正向约束一样重要。
#   parameters  —— 参数的 JSON Schema（一种描述数据格式的规范）：query 是字符串、必填。
#                  模型会按这个格式生成参数，我们的代码再把它解析出来。
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
# 角色说明：MCP 分“客户端”和“服务端”两个角色。这里我们是客户端，
# 去连接两个现成的 MCP 服务端（网页抓取 fetch、天气查询），把它们提供的工具接给模型用。
_mcp_client = None       # MCP 客户端对象（第一次加载后常驻内存，后续请求复用）
# 工具缓存的三种状态必须严格区分（这是本模块最容易踩的坑）：
#   None    = 还没尝试加载过
#   []      = 尝试过了但失败了（降级状态）
#   有内容 = 加载成功
# 如果用 None 同时表示“没加载”和“加载失败”，那失败之后每次对话请求都会重新去拉一次子进程，
# 白白卡住接口好几秒，而且永远好不了。
_mcp_tools_cache = None


async def get_mcp_tools():
    """加载 MCP 外部工具，返回可供 Agent 使用的工具对象列表。

    三个设计要点：
    - 懒加载：不在模块导入时就加载，而是第一次真正需要时才加载。
      因为 MCP 工具通过 stdio 方式连接，会拉起独立的 Python 子进程，
      放在导入阶段会拖慢服务启动，而且一旦启动失败会导致整个后端起不来。
    - 缓存：子进程只拉起一次，后续请求直接复用内存里的缓存列表。
    - 降级：加载失败返回空列表 []，也不让 MCP 的问题阻断主流程——
      没有外部工具，牛来依然能正常回答知识库问题和闲聊。
    返回：工具对象列表（成功时 5 个，失败或开关关闭时 []）
    """
    global _mcp_client, _mcp_tools_cache  # 要修改模块级变量必须声明 global，否则下面的赋值只会创建一个同名局部变量
    if not USE_MCP:
        return []                        # 开关关闭：直接不加载
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache          # 已加载过（成功或失败都算）：直接返回缓存，不再拉子进程
    try:
        # 声明要连接哪些 MCP 服务端。每个键是自己取的名字，值里说明怎么启动它。
        _mcp_client = MultiServerMCPClient({
            "fetch": {
                # sys.executable = 当前正在运行后端的这个 Python 解释器的完整路径（venv 里的那个）。
                # 不能写死成 "python"：系统可能装了多个 Python，写死会导致找不到 mcp_server_fetch 这个包。
                "command": sys.executable,
                "args": ["-m", "mcp_server_fetch"],  # 等价于命令行执行 python -m mcp_server_fetch
                "transport": "stdio",                # 通信方式：通过子进程的标准输入输出传递消息（本地工具的标准做法）
            },
            "weather": {  # Open-Meteo 天气服务端：免费、不需要申请 API key
                "command": sys.executable,
                "args": ["-m", "open_meteo_mcp"],
                "transport": "stdio",
            }
        })
        # 有几个服务端就要开几个会话（嵌套 with），全部打开后 get_tools() 才能拿到所有工具。
        # 只开一个会话的话，另一个服务端的工具会取不到——而且不报错，只是静默地少一批，很难发现。
        # 两个服务端合计提供 9 个工具。
        async with _mcp_client.session("fetch"):
            async with _mcp_client.session("weather"):
                all_tools = await _mcp_client.get_tools()
        # 工具精简（本段的核心设计点）：天气服务端自带 8 个工具，其中时区转换、空气质量类
        # 与模型自身能力重叠，留着只会增加决策噪音（工具越多，模型选错的概率越高），
        # 所以只保留真正需要的，过滤后共 5 个工具进模型的工具清单：
        _mcp_tools_cache = [
            t for t in all_tools
            if t.name == "fetch" or t.name in {   # fetch 是网页抓取工具本体
                "get_current_weather",         # 当前天气：“北京现在天气怎么样”
                "get_weather_byDateTimeRange", # 日期范围预报：“明天天气怎么样”
                "get_weather_details",         # 详细天气（含预报）：“给我份详细天气报告”
                "get_current_datetime",        # 当前时间：模型算“明天”是哪天要靠它（所以没被精简掉）
            }
        ]
        print(f"[MCP] 已加载工具: {[t.name for t in _mcp_tools_cache]}")  # 启动观察点：打在后端终端
    except Exception as e:
        # 兜住所有异常：MCP 依赖外部子进程，环境缺包、启动超时都可能失败，
        # 这里记一笔并降级，让服务照常跑。
        print(f"[MCP] 加载失败，降级为普通模式: {e}")
        _mcp_tools_cache = []   # 注意：缓存成 [] 而不是 None，表示“试过了、失败了”，后续不再重试
    return _mcp_tools_cache


# ===== LangChain 版 Agent（对照实现）=====
# @tool 装饰器的作用：把一个普通 Python 函数变成 Agent 可用的工具对象。
# 工具说明书（名字、参数类型、用途描述）由装饰器自动从函数签名和 docstring 里生成——
# 对比上面的手写版：同样的信息，手写版要写几十行 JSON，这里只要写好类型注解和 docstring。
# 注意：docstring 不是普通注释，它会被当成提示词发给模型，措辞同样影响调用准确率。
@langchain_tool
def search_knowledge_base_lc(query: str) -> str:
    """检索企业知识库。当问题涉及公司制度、内部规定或用户个人档案时必须先调用本工具。"""
    # 真正干活的还是 rag.py 里那个 search_knowledge_base——
    # 三套 Agent 实现只是“编排方式”不同（谁来决定查、什么时候查），底层检索能力是同一个，不重复实现。
    hits = search_knowledge_base(query)
    if not hits:
        # 没查到时要返回一句“人话”给模型，而不是空字符串：
        # 模型看到明确的“没有检索到内容”，才会如实告诉用户库里没资料；
        # 返回空串模型容易理解成“工具坏了”，或者干脆开始自己编。
        return "知识库中没有检索到相关内容"
    # 把检索结果拼成带编号的文本，格式：[1] (来自: 文件名)\n正文
    # 编号是关键——提示词里要求模型“引用了资料的句子末尾标注 [1]”，
    # 前端才能把回答里的编号和引用卡片对应起来，用户点一下就能核对原文。
    parts = [f"[{i + 1}] (来自: {meta.get('filename', '未知来源')})\n{doc}" for i, (doc, meta) in enumerate(hits)]
    return "\n\n".join(parts)   # enumerate 同时拿到下标 i 和元素，i + 1 让编号从 1 开始（人看着习惯）


# LangChain 版的模型客户端：和 config.py 里的 client 连的是同一个模型，
# 只是 LangChain 要求用自己的封装类（ChatOpenAI）才能接入它的 Agent 体系。
lc_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",  # DeepSeek 兼容 OpenAI 格式，所以 ChatOpenAI 换个地址就能连上
)
# 提示词模板：定义每次请求发给模型的消息结构。
# ChatPromptTemplate.from_messages 接收一个列表，每项是一类消息：
lc_prompt = ChatPromptTemplate.from_messages([
    ("system", PERSONA),                      # 人格设定，固定不变
    MessagesPlaceholder("chat_history"),      # 占位符：调用时传进来的历史消息会插在这里
    ("human", "{input}"),                     # 用户当前这句话；{input} 会被调用时传的同名变量替换
    MessagesPlaceholder("agent_scratchpad"),  # Agent 的“草稿纸”：中间的思考过程、工具调用和返回结果都记在这里。
                                              # 这是 LangChain Agent 的固定写法，名字不能改，照着写即可
])

# 组装 Agent：把模型、工具清单、提示词模板三者绑在一起，得到一个“会决定调工具的推理器”。
lc_agent = create_tool_calling_agent(lc_llm, [search_knowledge_base_lc], lc_prompt)
# AgentExecutor = 执行器：真正跑“决定→调工具→回填→再问模型”循环的是它，lc_agent 只负责单步推理。
# 两者分开是 LangChain 的设计：推理（agent）和执行（executor）解耦。
lc_executor = AgentExecutor(agent=lc_agent, tools=[search_knowledge_base_lc],
                            verbose=True)  # verbose=True：把 Agent 每一步的思考过程打印在后端终端，
                                           # 能直观看到框架做了几次模型调用、每次传了什么，对比手写版时重点看这里

# ===== LangGraph 版 Agent（当前主力）=====
# LangGraph 是什么：把 Agent 的工作流建模成一张“图”——节点是步骤（调模型、执行工具），
# 边是流转规则（模型要调工具就去工具节点，工具执行完回到模型节点，模型说完了就结束）。
# “思考⇄行动”的循环由图自动跑，我们不用像手写版那样自己写 if 判断和两次调用。
# 相比 LangChain 版，它的关键优势是支持流式：模型每吐一个字就能立刻转发给前端（打字机效果）；
# LangChain 版的 invoke 必须等整个循环跑完才一次性返回。
#
# 核心设计取舍：知识库检索不放给模型自主决定，而是由我们的代码在进图之前先做一次，
# 把结果以“【编号资料】”的形式写进提示词。理由有三：
#   ① 引用卡片能立刻发给前端，不用等模型决定完才出现（体验更快）；
#   ② 没查到时“如实告知库里没资料”这个分支由代码控制，稳定可预期，不靠模型自觉；
#   ③ 省掉一次“要不要查库”的模型决策调用，响应更快。
# 一句话概括这个原则：确定性的活交给代码，生成性的活交给模型。
# 但工具照常注册进图：万一代码检索的结果没覆盖用户问题，模型还能自己再查一次复核
#（图的循环天然支持这种“回头补课”，手写版做不到）。
#
# 系统提示词 = 人格（PERSONA）+ 回答规则。规则逐条对应一种场景，缺一条就会出对应的毛病：
#   规则 1：有资料时只用资料答 + 标编号引用（不标编号，前端的引用卡片就对不上号）
#   规则 2：资料为空/无关时如实说没有（不写这条，模型会凭训练记忆编造公司制度）
#   规则 3：资料没覆盖时允许自己再查一次（兜底复核）
#   规则 4：闲聊直接答，别调工具（不写这条，“你好”也可能触发一次无意义的检索）
#   规则 5：需要实时网页内容时用 fetch 工具（并限定场景，避免知识库问题也去联网）
#   规则 6：天气问题走天气工具。两个细节必须写清：城市名用英文/拼音（天气服务的地理编码对中文支持不稳），
#          以及遇到“明天”这类相对日期要先查当前时间（模型不知道今天是几号，算不出明天的日期）
GRAPH_SYSTEM = (
    PERSONA + "\n\n回答规则：\n"
    "1. 用户提供【编号资料】时，只基于资料回答；引用了资料的句子末尾标注编号如[1][2]；资料没覆盖的就如实说明。\n"
    "2. 【编号资料】为空或和问题无关时，如实告知知识库中没有相关资料（不要调用工具，不要编造）。\n"
    "3. 用户的问题需要知识库里的事实、而【编号资料】显然没覆盖时，可以调用 search_knowledge_base 复核一次。\n"
    "4. 与知识库无关的常识和闲聊，直接回答，不要调用工具。\n"
    "5. 用户需要实时网页内容（某个网页的信息、最新内容）时，调用 fetch 工具抓取后回答；知识库问题和闲聊不要调用它。\n"
    "6. 用户询问某城市的天气时，调用天气工具（城市名用英文或拼音，如 Beijing）；需要判断“明天”等相对日期时先调用 get_current_datetime；天气问题不要用 fetch。"
)
# 构建 LangGraph 图对象（模块加载时构建一次，之后所有请求复用）。
# 参数：模型 + 工具列表 + 系统提示词（新版 API 的参数名是 system_prompt，旧版叫 prompt）。
# 注意：这里的工具列表只有本地知识库工具。MCP 工具是异步加载的（要 await），模块导入阶段拿不到，
# 所以 main.py 里在 MCP 加载成功后会现场再构建一张带 MCP 工具的图；加载失败就用下面这张。
# 防死循环护栏 recursion_limit 不在这里传，而是在每次调用时通过 config 传（新版 API 的规定，见 main.py）。
lg_graph = create_react_agent(lc_llm, [search_knowledge_base_lc], system_prompt=GRAPH_SYSTEM)
