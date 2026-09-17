// ===== 登录状态 hook（步骤4c）=====
// 把「当前是不是亮哥 + 登录 + 退出」收口成一个 hook，页面用它拿身份、往下传给需要的组件。
// isLiang 只是前端「有没有票」的判断（决定画什么）；票真不真、过没过期，后端每次请求重新裁（纵深防御）。
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";
import { getToken, setToken, clearToken } from "@/lib/auth";

export function useAuth() {
  // 首屏统一 null（游客）：Next.js 静态导出的 HTML 里没有登录态，若初始化就读 localStorage 会水合不一致（服务端游客、客户端亮哥）。
  // 沿用项目惯例（见 useConversations）：挂载后在 useEffect 里恢复真实登录态，此时只在客户端跑，不冲突。
  const [token, setTokenState] = useState<string | null>(null);
  useEffect(() => {
    setTokenState(getToken());   // 挂载后（仅客户端）从 localStorage 恢复
  }, []);

  const isLiang = !!token;   // 有票 = 亮哥（前端视角）

  // 登录：口令 POST 给后端换门票，成功则存票 + 更新状态，返回是否成功（页面据此决定要不要刷新清单）
  const login = async (password: string): Promise<boolean> => {
    if (!password) return false;
    try {
      const res = await fetch(`${API_BASE}/api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (!res.ok) {
        alert("口令不正确");   // 401 统一文案，不透露「是口令错还是没配置」
        return false;
      }
      const data = await res.json();
      setToken(data.token);        // 存进 localStorage（刷新不丢）
      setTokenState(data.token);   // 更新 React 状态 → 重渲染成亮哥视图
      return true;
    } catch {
      alert("登录失败：无法连接后端服务，请确认后端已启动");
      return false;
    }
  };

  // 退出：清票 + 清状态（无状态 JWT，前端删票即退出，无需通知后端）
  const logout = () => {
    clearToken();
    setTokenState(null);
  };

  return { isLiang, login, logout };
}