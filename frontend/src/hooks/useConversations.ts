// ===== 会话持久化 hook（12.15 模块化拆分：localStorage 逻辑从 page.tsx 抽出）=====
// 专管"会话列表 + 当前选中"的数据生命周期：恢复、回写、改最后一条、删除都在这里，
// 页面组件只管调用，不再直接碰 localStorage
import { useEffect, useState } from "react";
import type { Conversation, Msg } from "@/types";

// localStorage 键名（第 3 期：会话持久化，刷新页面不丢）
const LS_CONVS = "ai-workspace:conversations";

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null); // null = 未开始的新对话（欢迎页）
  const [hydrated, setHydrated] = useState(false); // 从 localStorage 读取完成后才允许回写，防止空数据覆盖

  // ===== 第 3 期：从 localStorage 恢复会话历史（只在首次挂载执行一次）=====
  // 注意：只恢复会话列表，不恢复“上次选中的会话”——打开页面永远落在新对话欢迎页
  //（面试演示第一现场：谁来打开都先看到功能介绍和推荐问题；旧会话点左侧列表随时切回）
  useEffect(() => {
    try {
      const convs = localStorage.getItem(LS_CONVS);
      if (convs) setConversations(JSON.parse(convs));
    } catch {
      // 数据损坏就当没有，从空白开始（不阻断页面）
    }
    setHydrated(true);
  }, []);

  // ===== 第 3 期：状态变化即写回 localStorage（hydrated 前不回写）=====
  useEffect(() => {
    if (hydrated) localStorage.setItem(LS_CONVS, JSON.stringify(conversations));
  }, [conversations, hydrated]);

  // 当前激活会话的消息列表（派生值，不额外存状态，避免数据不一致）
  const messages = (conversations.find(c => c.id === activeId) ?? null)?.messages ?? [];

  // 更新指定会话的最后一条消息（流式渲染用：token 逐段追加到最后一条 AI 消息）
  const patchLastMsg = (convId: string, patch: (m: Msg) => Msg) => {
    setConversations(prev =>
      prev.map(c => {
        if (c.id !== convId || c.messages.length === 0) return c;
        const msgs = [...c.messages];
        msgs[msgs.length - 1] = patch(msgs[msgs.length - 1]);
        return { ...c, messages: msgs };
      })
    );
  };

  // 删除会话（二次确认防误触；阻止冒泡是 UI 关注点，留在 Sidebar 的点击处理里做）
  const deleteConversation = (id: string) => {
    if (!window.confirm("确定删除这条对话吗？删除后无法恢复。")) return;
    setConversations(prev => prev.filter(c => c.id !== id));
    if (activeId === id) setActiveId(null);
  };

  return {
    conversations,
    setConversations,
    activeId,
    setActiveId,
    messages,
    patchLastMsg,
    deleteConversation,
  };
}
