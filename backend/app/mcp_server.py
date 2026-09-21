"""知识库 MCP 服务端：把我们自己的知识库检索能力开放成标准 MCP 服务。

本项目 MCP 两边都做了：agents.py 里是客户端（连网页抓取、天气等外部服务）；
本文件里是服务端，把知识库能力开放出去，任何支持 MCP 的客户端（Claude Desktop / Cherry Studio 等）都能即插即用地查。
启动方式：由 MCP 客户端当子进程拉起（stdio 传输），不需自己启动；手动 python -m app.mcp_server 会“卡着”等客户端连接，是正常状态。
"""
import os
import sys

# 把 backend 根目录加进模块搜索路径：无论被 python -m app.mcp_server 启动还是被客户端用文件路径拉起，
# 下面的 from app.main import ... 都能成立（两种方式 sys.path 不同，客户端用文件路径拉起时 backend 不在搜索路径里）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP   # MCP 官方 Python SDK 的服务端框架，装饰器风格类似 FastAPI
# 检索能力复用主项目那一份，不重复实现。起别名 rag_search 是必须的：下面要用 @mcp.tool() 定义同名函数
# search_knowledge_base，不起别名后者会覆盖导入的名字，工具内部再调用就变成无限递归。
from app.rag import search_knowledge_base as rag_search

mcp = FastMCP("niulai-knowledge")  # 服务名：MCP 客户端的工具列表里会看到它


# @mcp.tool() 把普通函数注册成 MCP 工具（同 LangChain 的 @tool）：工具名取函数名、说明取 docstring、参数说明取类型注解
@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """检索企业知识库（用户上传的文档）。传入检索语句，返回最相关的文档片段及来源文件名。
    当问题涉及公司制度、内部规定、工作事务或员工个人档案时使用。"""
    hits = rag_search(query)   # 调用主项目检索函数，拿回 [(切片正文, 元数据), ...]
    if not hits:
        # 没查到要返回明确的“人话”，客户端的模型才知道该如实告知用户而非自己编
        return "知识库中没有检索到相关内容"
    # 拼成带编号文本返回：MCP 工具返回值只能是字符串，要把结构化结果拼成模型好读的格式
    parts = [f"[{i + 1}] (来自: {meta.get('filename', '未知来源')})\n{doc}" for i, (doc, meta) in enumerate(hits)]
    return "\n\n".join(parts)


if __name__ == "__main__":
    # 只有直接运行本文件时才执行（被 import 时不执行）。transport="stdio"：靠标准输入输出通信，由客户端当子进程拉起，不监听端口
    mcp.run(transport="stdio")
