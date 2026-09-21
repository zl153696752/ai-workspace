"use client";
// ===== 欢迎页组件 =====
// 新对话时展示：三大能力卡片 + 6 条推荐问题，点一下即演示。
import type { LucideIcon } from "lucide-react";
import {
  Bot,
  BookOpen,
  Brain,
  GitBranch,
  Globe,
  Search,
  Sparkles,
  UtensilsCrossed,
} from "lucide-react";

// 推荐问题：点击直接发送；6 条各演示一项硬实力（复合拆解 / 区分度检索 / 长期记忆 / 引用溯源 / 联网抓取 / 人格自述）。
const SUGGESTIONS: { icon: LucideIcon; text: string }[] = [
  { icon: GitBranch, text: "上海明天天气怎么样？顺便讲下公司年假规定" },
  { icon: Search, text: "N7 Pro 和 N7s 到底有什么区别？" },
  { icon: Brain, text: "你还记得我是做什么工作的吗？" },
  { icon: UtensilsCrossed, text: "加班餐补怎么算？" },
  { icon: Globe, text: "帮我看看 example.com 页面的标题是什么" },
  { icon: Bot, text: "介绍下你自己" },
];

type WelcomeProps = {
  onSend: (text: string) => void; // 推荐问题点击后交给页面发送（开会话的逻辑在页面手里）
};

export default function Welcome({ onSend }: WelcomeProps) {
  return (
    <div className="flex-1 flex flex-col overflow-y-auto px-6">
      {/* 外层 flex-col + m-auto 实现垂直居中；内容超出时从顶部可滚动 */}
      <div className="m-auto max-w-[680px] flex flex-col items-center py-10 w-full">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#4d6bfe] to-[#7c93ff] flex items-center justify-center shadow-lg shadow-blue-100">
          <Sparkles className="w-7 h-7 text-white" />
        </div>
        <h1 className="mt-5 text-[22px] font-medium">嗨，我是牛来</h1>
        <p className="mt-2 text-sm text-gray-400 text-center leading-6 max-w-[520px]">
          一个能独当一面的企业级 AI Agent：复杂问题自动拆解、并行处理，
          基于企业文档精准回答并标注来源，有跨会话的长期记忆，还能联网查天气、抓网页
        </p>

        {/* 能力卡片：功能标题给普通用户、括号里的技术词（RAG/MCP/LangGraph）故意保留给面试官看；4 张 2×2，窄窗口降为单列 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-8 w-full">
          <div className="px-4 py-3.5 rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-800">
              <GitBranch className="w-4 h-4 text-[#4d6bfe]" />
              复杂问题拆解
            </div>
            <p className="mt-1.5 text-xs text-gray-400 leading-5">
              一次问好几件事，自动拆成子任务并行处理再汇总（LangGraph 多 Agent 编排）
            </p>
          </div>
          <div className="px-4 py-3.5 rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-800">
              <BookOpen className="w-4 h-4 text-[#4d6bfe]" />
              知识库问答
            </div>
            <p className="mt-1.5 text-xs text-gray-400 leading-5">
              企业文档混合检索 + Reranker 精排（RAG），回答带编号引用，来源卡片可核对、下载原文
            </p>
          </div>
          <div className="px-4 py-3.5 rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-800">
              <Brain className="w-4 h-4 text-[#4d6bfe]" />
              长期记忆
            </div>
            <p className="mt-1.5 text-xs text-gray-400 leading-5">
              跨会话记住确认过的重要信息，越用越懂你（SQLite 持久化）
            </p>
          </div>
          <div className="px-4 py-3.5 rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-800">
              <Globe className="w-4 h-4 text-[#4d6bfe]" />
              联网工具
            </div>
            <p className="mt-1.5 text-xs text-gray-400 leading-5">
              查任意城市实时天气、抓取指定网页内容（MCP 协议接入 Open-Meteo / fetch）
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 mt-5 w-full max-w-[560px]">
          {SUGGESTIONS.map(({ icon: Icon, text }) => (
            <button
              key={text}
              onClick={() => onSend(text)}
              className="flex items-center gap-2.5 px-4 py-3.5 rounded-xl border border-gray-200 text-sm text-gray-600 text-left hover:border-[#4d6bfe] hover:text-[#4d6bfe] hover:bg-[#f5f7ff] transition-colors"
            >
              <Icon className="w-4 h-4 shrink-0" />
              {text}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
