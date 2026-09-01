"""知识库 MCP 服务端：把我们自己的知识库检索能力开放成标准 MCP 服务。

角色说明：MCP 分“客户端”和“服务端”两个角色，本项目两边都做了：
  - 在 agents.py 里我们是客户端，去连别人提供的服务（网页抓取、天气）；
  - 在本文件里我们是服务端，把自己的能力开放出去，
    任何支持 MCP 的客户端（Claude Desktop / Cherry Studio 等）都能即插即用地查我们的知识库。

启动方式：由 MCP 客户端当子进程拉起（stdio 传输），不需要我们自己启动。
手动执行 python -m app.mcp_server 会一直“卡着”不动，那是正常状态——
它在等客户端连接，不是死机（Ctrl+C 退出即可）。
"""
import os
import sys

# 把 backend 根目录加进模块搜索路径：这样无论本文件是被 `python -m app.mcp_server` 启动，
# 还是被 MCP 客户端用文件路径直接拉起，下面的 `from app.main import ...` 都能成立。
# 原因：两种启动方式下 Python 的 sys.path（模块搜索路径列表）不一样，
# 客户端用文件路径拉起时 backend 目录不在搜索路径里，不加这行就会 ModuleNotFoundError。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 拆开看：abspath(__file__) = 本文件的完整路径 → 第一层 dirname 得到 backend/app → 第二层 dirname 得到 backend；
# insert(0, ...) 是插到搜索路径最前面，优先从这里找模块

from mcp.server.fastmcp import FastMCP   # MCP 官方 Python SDK 提供的服务端框架，装饰器风格和 FastAPI 很像
# 检索能力直接复用主项目里那一份，不重复实现。
# 起别名 rag_search 是必须的：下面要用 @mcp.tool() 定义一个同名的工具函数 search_knowledge_base，
# 不起别名的话，后定义的函数会把导入进来的名字覆盖掉，工具内部再调用就变成无限递归。
from app.main import search_knowledge_base as rag_search

mcp = FastMCP("ai-workspace-knowledge")  # 服务名：MCP 客户端的工具列表里会看到这个名字


# @mcp.tool() 把一个普通函数注册成 MCP 工具，和 LangChain 的 @tool 一个道理：
# 工具名取函数名，工具说明取 docstring，参数说明取类型注解——所以 docstring 要写清楚什么时候用。
@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """检索企业知识库（用户上传的文档）。传入检索语句，返回最相关的文档片段及来源文件名。
    当问题涉及公司制度、内部规定、工作事务或员工个人档案时使用。"""
    hits = rag_search(query)   # 调用主项目的检索函数，拿回 [(切片正文, 元数据), ...]
    if not hits:
        # 没查到要返回一句明确的“人话”，客户端的模型才知道该如实告知用户，而不是自己编
        return "知识库中没有检索到相关内容"
    # 拼成带编号的文本返回：MCP 工具的返回值只能是字符串，所以要把结构化结果拼成模型好读的格式
    parts = [f"[{i + 1}] (来自: {meta.get('filename', '未知来源')})\n{doc}" for i, (doc, meta) in enumerate(hits)]
    return "\n\n".join(parts)


if __name__ == "__main__":
    # 只有直接运行本文件时才执行这里（被其他模块 import 时不执行）。
    # transport="stdio"：靠标准输入输出通信，由客户端当子进程拉起，不监听任何网络端口
    #（另一种方式是 HTTP，需要占端口，适合服务部署在远端的场景）。
    mcp.run(transport="stdio")
