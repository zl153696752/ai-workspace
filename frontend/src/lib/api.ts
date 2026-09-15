// ===== 后端 API 的基地址 =====
// 集中管理基地址，避免各处 fetch 把 http://localhost:8000 写死（否则线上会去请求用户自己的 8000 端口而失败）。
// 规则：开发环境端口不同(3000/8000)必须给全地址；生产环境前端由后端同源托管，给空串走相对路径 /api/xxx。
// NODE_ENV 由 Next.js 构建时替换成字面量，产物里没有 process 变量，故浏览器不会报 "process is not defined"。
// NEXT_PUBLIC_API_BASE 是活口：想指向别的后端，建 .env.production 写上它即可覆盖，不用改代码。
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ??
  (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "");
