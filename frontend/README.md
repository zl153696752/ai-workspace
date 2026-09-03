# 前端（Next.js）

> 项目总说明在[仓库根目录的 README](../README.md)，这里只写本地开发相关的。

## 本地开发

```powershell
pnpm install
pnpm dev
# → http://localhost:3000（后端要同时在 8000 端口跑着）
```

## 构建静态产物

```powershell
pnpm build
# 产出在 out/ 目录（next.config.ts 里设了 output: "export"）
```

线上部署时 `out/` 的内容会被拷进 Docker 镜像、由后端 FastAPI 托管，所以生产环境**不需要 Node 进程**。

## 目录

| 路径 | 作用 |
| --- | --- |
| `src/app/page.tsx` | 页面：业务编排 |
| `src/components/` | Sidebar / Welcome / MessageList / ChatInput |
| `src/hooks/` | useConversations / useKnowledgeFiles |
| `src/lib/api.ts` | API 基地址（开发/生产自动切换） |
| `src/types.ts` | 共享类型 |
