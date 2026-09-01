"""知识库 MCP Server（第 9 步方向 B）：把企业知识库检索能力暴露为标准 MCP 服务，
任何支持 MCP 的客户端（Claude Desktop / Cherry Studio 等）都能即插即用。
启动方式：由客户端拉起（stdio 传输）；手动跑会一直'卡着'等连接，那是正常状态（12.9 坑 6）"""
import os
import sys

# 把 backend 根目录加入搜索路径：无论被 `python -m app.mcp_server` 还是被客户端用文件路径拉起，
# `from app.main import` 都成立（两种启动方式的 sys.path 不一样，12.9 坑 4）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from app.main import search_knowledge_base as rag_search  # 复用同一个执行体：检索能力只有一份；起别名避免和下面的工具函数同名（12.9 坑 7）

mcp = FastMCP("ai-workspace-knowledge")  # 服务名：客户端里会看到这个名字


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """检索企业知识库（用户上传的文档）。传入检索语句，返回最相关的文档片段及来源文件名。
    当问题涉及公司制度、内部规定、工作事务或员工个人档案时使用。"""
    hits = rag_search(query)
    if not hits:
        return "知识库中没有检索到相关内容"
    parts = [f"[{i + 1}] (来自: {meta.get('filename', '未知来源')})\n{doc}" for i, (doc, meta) in enumerate(hits)]
    return "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")  # stdio：靠标准输入输出通信，由客户端当子进程拉起；不监听任何端口（12.12 进阶换 HTTP）
