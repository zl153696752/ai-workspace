import type { Metadata } from "next";
import "./globals.css";

// 0-1：删掉了 next/font/google 的 import 和两个 Geist 常量。
// 原因：它会在 next build 时联网去 Google 拉字体，而部署平台的构建机在国内拉不到。
// 而且实测发现这两个字体本来就没生效（globals.css 把 body 硬编码成了 Arial），删了零视觉变化。

// 0-2：下面两项会出现在面试官的浏览器标签页和页面源码里。
// 原来是脚手架默认的「Create Next App」，初始化后从未改过。
export const metadata: Metadata = {
  title: "牛来 · AI 知识库助手",
  description:
    "基于 FastAPI + Next.js + Chroma + DeepSeek 的 RAG 知识库助手，支持流式对话、文档上传、引用溯源、Agent 工具调用与 MCP 接入。",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    // 0-2：lang 从 en 改成 zh-CN，页面主体是中文，对读屏和浏览器翻译都更友好
    // 0-1：className 里去掉了 geistSans.variable / geistMono.variable，因为那两个常量已删
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
