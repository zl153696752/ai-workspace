import type { Metadata } from "next";
import "./globals.css";

// 不用 next/font/google：它会在 build 时联网去 Google 拉字体（国内构建机拉不到），且字体本就没生效。

// 下面两项会出现在浏览器标签页和页面源码里（原为脚手架默认值）。
export const metadata: Metadata = {
  title: "牛来 · AI 知识库助手",
  description:
    "基于 FastAPI + Next.js + Chroma + DeepSeek 的 RAG 知识库助手，支持流式对话、文档上传、引用溯源、Agent 工具调用与 MCP 接入。",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    // lang=zh-CN：页面主体是中文，对读屏和浏览器翻译更友好
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
