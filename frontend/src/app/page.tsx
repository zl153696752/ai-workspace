"use client";
import { useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");

  type Source = { id: number; filename: string; snippet: string };
  type Msg = { role: string; content: string; sources?: Source[] };

  // 组件内：
  const [messages, setMessages] = useState<Msg[]>([]);

  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState<string[]>([]);   // 已上传的文件名列表
  const [uploading, setUploading] = useState(false);

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
    let buffer = "";   // SSE 缓冲区：网络分包和消息边界不对齐，必须攒够一条完整消息再解析  
  
    setMessages(prev => [...prev, { role: "assistant", content: "" }]);  
  
    while (reader) {  
      const { done, value } = await reader.read();  
      if (done) break;  
      buffer += decoder.decode(value, { stream: true });   // stream:true 防中文被拦腰截成乱码（多字节字符）  
  
      // 一条完整的 SSE 消息以空行 "\n\n" 结束；最后一段可能不完整，留在 buffer 里等下一轮  
      const events = buffer.split("\n\n");  
      buffer = events.pop() ?? "";  
  
      for (const evt of events) {  
        if (!evt.trim()) continue;  
        let name = "";  
        let data = "";  
        for (const line of evt.split("\n")) {  
          if (line.startsWith("event:")) name = line.slice(6).trim();  
          else if (line.startsWith("data:")) data = line.slice(5).trim();  
        }  
        if (!data) continue;  
        const payload = JSON.parse(data);  
  
        if (name === "sources") {  
          // 来源事件：写进当前 AI 消息的 sources 字段（卡片会立刻渲染出来，不用等回答完成）  
          setMessages(prev => {  
            const next = [...prev];  
            next[next.length - 1] = { ...next[next.length - 1], sources: payload };  
            return next;  
          });  
        } else if (name === "token") {  
          aiContent += payload.content;  
          setMessages(prev => {  
            const next = [...prev];  
            next[next.length - 1] = { ...next[next.length - 1], content: aiContent };  
            return next;  
          });  
        }  
        // done 事件：无需处理，流结束后下面会重置 loading  
      }  
    }  
    setLoading(false);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);  // "file" 这个名字要和后端 UploadFile 的参数名一致

    const res = await fetch("http://localhost:8000/api/upload", {
      method: "POST",
      body: formData,  // ⚠️ 注意：发 FormData 千万不要手动设置 Content-Type！
    });

    if (!res.ok) {
      const err = await res.json();
      alert(err.detail);  // 后端返回的错误信息，比如“不支持的文件类型”
    } else {
      const data = await res.json();
      setFiles(prev => [...prev, data.filename]);
    }

    setUploading(false);
    e.target.value = "";  // 清空 input，允许再次选择同一个文件
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">AI Workspace</h1>

      <div className="border rounded p-4 mb-4">
        <div className="flex items-center gap-2">
          <input
            type="file"
            accept=".txt,.md,.pdf"
            onChange={handleUpload}
            disabled={uploading}
            className="text-sm"
          />
          {uploading && <span className="text-gray-400 text-sm">上传中...</span>}
        </div>
        {files.length > 0 && (
          <div className="mt-2 text-sm text-gray-600">已上传：{files.join("、")}</div>
        )}
      </div>

      <div className="space-y-4 mb-4 min-h-[400px] border rounded p-4">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role === "user" ? "text-right" : "text-left"}>
            <div className={`inline-block p-3 rounded-lg ${
              msg.role === "user" ? "bg-blue-500 text-white" : "bg-gray-100"
            }`}>
              {msg.content}
            </div>
            {msg.sources && msg.sources.length > 0 && (  
              <div className="mt-2 text-xs text-gray-500 space-y-1 max-w-md">  
                {msg.sources.map(s => (  
                  <details key={s.id} className="bg-gray-50 border rounded px-2 py-1">  
                    <summary className="cursor-pointer select-none">  
                      [{s.id}] {s.filename}  
                    </summary>  
                    <div className="mt-1 whitespace-pre-wrap leading-relaxed">{s.snippet}</div>  
                  </details>  
                ))}  
              </div>  
            )}
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
