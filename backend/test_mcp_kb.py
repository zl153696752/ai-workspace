"""知识库 MCP Server 验证脚本（第 9 步）：把 app.mcp_server 当子进程拉起，拿工具并真实检索一次"""
import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient({
        "kb": {
            "command": sys.executable,
            "args": ["-m", "app.mcp_server"],
            "transport": "stdio",
        }
    })
    async with client.session("kb"):
        tools = await client.get_tools()
        print("暴露的工具:", [t.name for t in tools])
        out = await tools[0].ainvoke({"query": "年假"})
        print("真实检索结果（前 100 字）:", str(out)[:100])


asyncio.run(main())
