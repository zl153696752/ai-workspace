"use client";
import { useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { role: "user", content: input };  
    const newMessages = [...messages, userMsg];  // ← 改动1：先算出包含本句话的完整历史
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    const response = await fetch("http://localhost:8000/api/chat", {  
      method: "POST",  
      headers: { "Content-Type": "application/json" },  // ← 改动2：告诉后端我发的是 JSON（必须有）  
      body: JSON.stringify({ messages: newMessages }),   // ← 改动3：发完整历史，不再用 ?message= 参数  
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let aiContent = "";

    setMessages(prev => [...prev, { role: "assistant", content: "" }]);

    while (reader) {
      const { done, value } = await reader.read();
      if (done) break;
      aiContent += decoder.decode(value);
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1] = { role: "assistant", content: aiContent };
        return newMsgs;
      });
    }
    setLoading(false);
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">AI Workspace</h1>

      <div className="space-y-4 mb-4 min-h-[400px] border rounded p-4">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role === "user" ? "text-right" : "text-left"}>
            <div className={`inline-block p-3 rounded-lg ${
              msg.role === "user" ? "bg-blue-500 text-white" : "bg-gray-100"
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
        {loading && <div className="text-gray-400">AI 正在思考...</div>}
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="输入消息..."
          className="flex-1 border rounded px-4 py-2"
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
        >
          发送
        </button>
      </div>
    </div>
  );
}
