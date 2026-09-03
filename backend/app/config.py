# ===== 全局配置与资源初始化 =====
# 本文件只做一件事：把整个后端要共用的“资源”集中创建一次——
# 环境变量、大模型客户端、向量数据库连接、业务常量、架构开关。
# 其他模块（rag.py / agents.py / main.py）直接 import 这里的东西用，谁都不再自己新建一份。
#
# 为什么必须集中在一处：数据库连接、模型客户端这类对象创建成本高（要连磁盘、要建网络会话），
# 如果每个文件各建一份，既浪费资源，又容易出现“配置不一致”——
# 比如 A 文件连的是这个向量库目录、B 文件连的是另一个，A 写入的数据 B 永远查不到，这种 bug 极难排查。
# 这种“集中创建一次、全项目复用”的写法叫资源单例。
import os
from dotenv import load_dotenv   # 读取 .env 配置文件的第三方库
from openai import OpenAI       # OpenAI 官方 SDK，用它连 DeepSeek（原因见下）
import chromadb                 # 向量数据库 Chroma 的客户端库
from .embeddings_bge import BgeZhEF   # 自己写的中文嵌入模型（为什么不用 Chroma 自带的，见 embeddings_bge.py 文件头）

# 读取 backend/.env 文件里的键值对（本项目里主要是 DEEPSEEK_API_KEY），写进程序的环境变量。
# 两个要点：
#   ① 必须在任何 os.getenv(...) 之前调用，否则读到的是 None；
#   ② 密钥放在 .env 而不是写死在代码里，代码可以放心提交到 git，.env 已被 gitignore 排除，密钥不会泄露。
load_dotenv()

# ===== 大模型客户端 =====
# 这里用的是 OpenAI 官方 SDK，但连的是 DeepSeek——能这么做是因为 DeepSeek 的接口格式与 OpenAI 完全兼容：
# 只要把 base_url 换成 DeepSeek 的地址、api_key 换成 DeepSeek 的密钥，
# 原本写给 OpenAI 的代码一行都不用改（国产大模型普遍兼容 OpenAI 格式，这已是行业事实标准）。
# 之后所有“直接调模型”的地方（改写问题、手写版 Agent）都复用这个 client 对象。
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),   # 从环境变量取密钥（上一行 load_dotenv 已把 .env 加载进来）
    base_url="https://api.deepseek.com"      # 换成 DeepSeek 的服务地址
)

# ===== 上传文件的存放目录与限制 =====
# os.path.dirname(__file__) 是本文件所在目录（backend/app），接 ".." 退回 backend，
# 最终拼出 backend/uploads/ ——用户上传的原文件实体就存在这个目录（存的是磁盘文件，不是向量）。
# 用 __file__ 推导路径而不是写死 "E:/ai-workspace/..."：项目挪到任何机器、任何盘符都能正常跑。
# 数据根目录：默认跟代码放在一起（本地开发），线上可用环境变量 DATA_DIR 指到别处。
# 为什么留这个活口：见 13.0.1 表 C 的「磁盘是否跨重启持久」未验证项。
DATA_DIR = os.getenv("DATA_DIR") or os.path.join(os.path.dirname(__file__), "..")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 目录不存在就创建；exist_ok=True 表示已存在也不报错（否则服务第二次启动就崩）
ALLOWED_EXT = [".txt", ".md", ".pdf"]   # 上传白名单：只收这三种后缀，其余格式（如 .docx）后端没有解析能力，在接口层直接拒掉
MAX_FILE_SIZE = 5 * 1024 * 1024         # 单文件上限 5MB（5 * 1024 * 1024 字节）。为什么必须拦：入库流程（提取文字→切片→算向量）
                                        # 是同步阻塞的，一个超大文件会把上传接口卡住几十秒甚至几分钟，所以要在读进内存之前先挡住
MAX_TEXT_LENGTH = 300000                # 提取后的文字量上限 30 万字（约 1000 个切片）。为什么和文件大小分开限制：
                                        # 真正决定入库成本的是文字量而不是文件体积——5MB 的扫描版 PDF 可能一个字都提不出来，
                                        # 而 1MB 的纯文本可能有几十万字。只卡体积会漏掉“小体积大文字量”的文件

# ===== 向量数据库 Chroma =====
# 什么是向量数据库（本项目最核心的概念）：
#   普通数据库按关键词精确匹配，搜“年假”只能找到含“年假”二字的记录；
#   向量数据库按“语义相似度”匹配——每段文字会被 embedding 模型转成一串数字（向量），
#   语义越接近的两段文字，它们的向量在空间里的距离越近。
#   检索时把问题也转成向量，找距离最近的几段文字，于是问“年假有几天”
#   也能命中写着“员工每年享有带薪休假 5 天”的文档，哪怕一个关键词都不重合。
# PersistentClient = 数据持久化到磁盘目录（backend/chroma_db/），服务重启数据还在。
# 与之相对的 EphemeralClient 只存在内存里，进程一退数据全没（仅适合写测试）。
chroma_client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))
# collection 相当于关系型数据库里的“一张表”：所有知识库切片都存在名为 knowledge 的这一张表里。
# get_or_create_collection：第一次运行时自动建表，之后运行直接拿到已有数据，不需要手动初始化。
collection = chroma_client.get_or_create_collection(name="knowledge", embedding_function=BgeZhEF())

# ===== 切片参数 =====
# 为什么要切片，而不是整篇文档存一条：
#   ① 模型有上下文长度限制，整本员工手册塞不进提示词；
#   ② 检索粒度问题——整篇文档当一个切片，问什么它都“沾一点边”，距离分数被无关内容稀释，反而查不准。
# 切成小片后每片只讲一件事，语义集中，命中才精确。
CHUNK_SIZE = 300     # 每个切片最多 300 字。太短则上下文不足（一句话被拆成两片，哪片都表达不完整），
                     # 太长则一片里混了好几个话题，检索精度下降。300 字是实测下来的平衡点
CHUNK_OVERLAP = 50   # 相邻切片重叠 50 字。纯按 300 字硬切会把句子拦腰截断（前半句在片 A、后半句在片 B），
                     # 让上一片末尾的 50 字在下一片开头再出现一次，就能保证任何一句话至少在某一片里是完整的

# ===== 架构开关（学习对照用）=====
# 同一个 /api/chat 接口，本项目保留了三套实现：
#   ① 手写版      —— 纯 OpenAI SDK，自己写“决定调工具 → 执行 → 回填 → 再生成”的两次调用
#   ② LangChain 版 —— 框架接管上面那个循环，我们只调一次 invoke
#   ③ LangGraph 版 —— 把流程建模成一张图，支持流式逐字输出（当前主力）
# 三套代码同时留在仓库里，靠下面两个开关决定这次请求走哪一套。
# 这么做的目的是学习对照：把开关拨一下，就能直观看到“框架到底替我做了什么”。
# 生产项目只会保留一套实现，不会这样共存。
# 优先级由 main.py 里的 if / elif / else 顺序决定：USE_LANGGRAPH > USE_LANGCHAIN > 手写版。
USE_LANGCHAIN = False  # True 时 /api/chat 交给 LangChain Agent：框架自动完成“决定调不调工具→执行→结果回填→生成回答”
USE_LANGGRAPH = True   # True 时交给 LangGraph（当前生效的实现）：流式打字机 + 引用卡片都齐全
USE_MCP = True         # MCP（Model Context Protocol，模型上下文协议）是一套让“外部工具服务”标准化接入大模型的协议。
                       # True 时加载两个外部 MCP 工具服务：网页抓取（fetch）、天气查询，牛来因此具备联网能力；
                       # 加载失败会自动降级成“只有知识库工具”的普通模式，不影响服务启动和其余功能
