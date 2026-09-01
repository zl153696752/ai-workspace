"use client";
// ===== 欢迎页组件（12.15 模块化拆分：从 page.tsx 抽出，12.14 的功能广告位）=====
// 默认新对话时展示：三大能力卡片 + 6 条推荐问题，点一下就是演示
import type { LucideIcon } from "lucide-react";
import {
  Bot,
  BookOpen,
  CalendarDays,
  CloudSun,
  Globe,
  Hash,
  Sparkles,
  UtensilsCrossed,
} from "lucide-react";

// 欢迎页推荐问题：点击直接发送，面试官零门槛体验核心功能；
// 6 条覆盖三大能力：知识库问答 ×3、天气查询 ×1（MCP）、网页抓取 ×1（MCP）、人格 ×1
const SUGGESTIONS: { icon: LucideIcon; text: string }[] = [
  { icon: CalendarDays, text: "年假几天？" },
  { icon: Hash, text: "公司代号是什么？" },
  { icon: UtensilsCrossed, text: "加班餐补怎么算？" },
  { icon: CloudSun, text: "北京现在天气怎么样？" },
  { icon: Globe, text: "帮我看看 example.com 页面的标题是什么" },
  { icon: Bot, text: "介绍下你自己" },
];

type WelcomeProps = {
  onSend: (text: string) => void; // 推荐问题点击后交给页面发送（开会话的逻辑在页面手里）
};

export default function Welcome({ onSend }: WelcomeProps) {
  return (
    <div className="flex-1 flex flex-col overflow-y-auto px-6">
      {/* 外层是 flex-col，m-auto 才能真正垂直居中；内容超出时从顶部开始可滚动，两头都不难受 */}
      <div className="m-auto max-w-[680px] flex flex-col items-center py-10 w-full">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#4d6bfe] to-[#7c93ff] flex items-center justify-center shadow-lg shadow-blue-100">
          <Sparkles className="w-7 h-7 text-white" />
        </div>
        <h1 className="mt-5 text-[22px] font-medium">嗨，我是牛来</h1>
        <p className="mt-2 text-sm text-gray-400 text-center leading-6 max-w-[520px]">
          一个能查知识库、也能联网的 AI 工作区：基于企业文档回答并标注来源，
          实时抓取网页内容，还能查询任意城市的天气
        </p>

        {/* 能力卡片：RAG / MCP 这些术语故意保留，给面试官看的；窄窗口降为单列，不压扁卡片 */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-8 w-full">
          <div className="px-4 py-3.5 rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-800">
              <BookOpen className="w-4 h-4 text-[#4d6bfe]" />
              知识库问答
            </div>
            <p className="mt-1.5 text-xs text-gray-400 leading-5">
              上传企业文档检索增强（RAG），回答带编号引用，来源卡片可展开核对、下载原文
            </p>
          </div>
          <div className="px-4 py-3.5 rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-800">
              <CloudSun className="w-4 h-4 text-[#4d6bfe]" />
              天气查询
            </div>
            <p className="mt-1.5 text-xs text-gray-400 leading-5">
              问任意城市当前或未来天气，MCP 协议接入 Open-Meteo，真实数据不编造
            </p>
          </div>
          <div className="px-4 py-3.5 rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-800">
              <Globe className="w-4 h-4 text-[#4d6bfe]" />
              网页抓取
            </div>
            <p className="mt-1.5 text-xs text-gray-400 leading-5">
              给一个网址，实时抓取页面内容后回答，MCP 协议接入官方 fetch 服务
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
