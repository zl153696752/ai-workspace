// ===== 后端 API 的基地址 =====
// 为什么要有这个文件：原先 5 处 fetch 全把 http://localhost:8000 写死在字符串里，
// 部署到线上后，面试官的浏览器会去请求「他自己电脑的 8000 端口」，必然失败。
//
// 两种环境的差别：
//   本地开发：前端 next dev 跑在 3000，后端 uvicorn 跑在 8000 → 端口不同 = 跳域，必须写全地址
//   线上部署：前端静态文件由后端 FastAPI 自己托管 → 同源，用相对路径 /api/xxx 即可，
//             浏览器会自动拼上当前页面的协议 + 域名 + 端口
//
// 所以规则是：开发环境给全地址，生产环境给空串（空串 + "/api/chat" = 相对路径）。
//
// process.env.NODE_ENV 由 Next.js 在【构建时】直接替换成字符串字面量，
// 打进产物的代码里根本不存在 process 这个变量，所以浏览器里不会报 "process is not defined"。
// NEXT_PUBLIC_API_BASE 是留的活口：哪天想指向别的后端，建个 .env.production 写上它就能覆盖，不用改代码。
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ??
  (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "");
