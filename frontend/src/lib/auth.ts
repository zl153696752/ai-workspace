// ===== 前端鉴权工具（步骤4c）=====
// 无状态 JWT 的前端半边：门票存 localStorage（刷新不丢），每次请求带上 Authorization 头。
// 🔴 前端只管「存票 + 带票 + 按有没有票决定画什么」；真正的身份裁决永远在后端（前端能被绕过，见方案模块3纵深防御）。
// 键名沿用项目命名空间习惯（对比 useConversations 的 "ai-workspace:conversations"）。
const TOKEN_KEY = "ai-workspace:token";

// 读门票：没有返回 null（游客）。typeof window 判断是因为 Next.js 静态导出/构建期没有 window，直接读 localStorage 会报错。
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

// 存门票（登录成功后调）
export function setToken(token: string): void {
  if (typeof window !== "undefined") localStorage.setItem(TOKEN_KEY, token);
}

// 清门票（退出登录）：无状态 JWT 的「退出」= 前端删票，之后请求不带票，后端自然当游客
export function clearToken(): void {
  if (typeof window !== "undefined") localStorage.removeItem(TOKEN_KEY);
}

// 造鉴权头：有票返回 { Authorization: "Bearer <票>" }，没票返回 {}。
// 用法：fetch(url, { headers: { "Content-Type": "application/json", ...authHeaders() } }) —— spread 进去，游客自动不带这个头。
export function authHeaders(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}