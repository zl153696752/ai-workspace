# ===== 全局配置与资源初始化 =====
# 把后端共用的资源（环境变量、模型客户端、向量库连接、业务常量、架构开关）集中创建一次，
# 其他模块直接 import 复用（资源单例），避免各建一份导致配置不一致、数据互相查不到。
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from .embeddings_bge import BgeZhEF   # 自己写的中文嵌入模型

# 加载 backend/.env（主要是 DEEPSEEK_API_KEY）到环境变量。必须在任何 os.getenv 之前调用。
load_dotenv()

# ===== 大模型客户端 =====
# 用 OpenAI SDK 连 DeepSeek（接口格式完全兼容，只换 base_url 和 api_key）。全局复用此 client。
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ===== 上传文件的存放目录与限制 =====
# 数据根目录默认跟代码放一起，线上可用环境变量 DATA_DIR 指到别处；uploads/ 存用户上传的原文件实体。
DATA_DIR = os.getenv("DATA_DIR") or os.path.join(os.path.dirname(__file__), "..")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)  # exist_ok=True：目录已存在也不报错，避免二次启动崩
ALLOWED_EXT = [".txt", ".md", ".pdf"]   # 上传白名单：其余格式后端无解析能力，接口层直接拒
MAX_FILE_SIZE = 5 * 1024 * 1024         # 单文件上限 5MB：入库同步阻塞，超大文件会卡死上传接口，读进内存前先挡
MAX_TEXT_LENGTH = 300000                # 文字量上限 30 万字：决定入库成本的是文字量而非体积，与 MAX_FILE_SIZE 分开卡

# ===== 向量数据库 Chroma =====
# 向量库按“语义相似度”检索：文字转向量，语义越近向量距离越近，问“年假”能命中“带薪休假”。
# PersistentClient 把数据持久化到磁盘（chroma_db/），服务重启数据还在。
chroma_client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))
# collection 相当于一张表，所有知识库切片都存在名为 knowledge 的表里；get_or_create 首次自动建表，之后复用
collection = chroma_client.get_or_create_collection(name="knowledge", embedding_function=BgeZhEF())

# ===== 切片参数 =====
# 切片而非整篇存：受模型上下文长度限制，且整篇当一片语义太杂、检索会被无关内容稀释。
CHUNK_SIZE = 300     # 每片最多 300 字：太短上下文不足，太长一片混多个话题降低精度，300 是实测平衡点
CHUNK_OVERLAP = 50   # 相邻片重叠 50 字：避免硬切把句子拦腰截断，保证任一句至少在某一片里完整

# ===== 架构开关（学习对照用）=====
# 同一个 /api/chat 保留三套实现（手写版 / LangChain / LangGraph），靠开关决定走哪套，方便对照“框架做了什么”。
# 优先级由 main.py 的 if/elif/else 决定：USE_LANGGRAPH > USE_LANGCHAIN > 手写版。生产只留一套。
USE_LANGCHAIN = False  # True：/api/chat 交给 LangChain Agent，框架自动完成“决定调工具→执行→回填→生成”
USE_LANGGRAPH = True   # True：交给 LangGraph（当前生效），流式打字机 + 引用卡片齐全
USE_MCP = True         # True：加载外部 MCP 工具服务（网页抓取、天气），失败自动降级为“只有知识库工具”，不影响启动
