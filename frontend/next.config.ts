import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 静态导出：next build 不再产出需要 Node 服务端运行的 .next/，
  // 而是产出一堆纯静态文件到 out/ 目录（index.html + _next/ 下的 js/css）。
  // 这样后端 FastAPI 就能像托管普通网页一样托管它，一个进程一个端口搞定（理由见 13.3.1）。
  output: "export",
  // 静态导出没有 Node 服务端，next/image 的实时优化功能用不了。
  // 本项目一处 next/image 都没用（已实测），加这行是防止以后用了导致构建直接失败。
  images: { unoptimized: true },
};

export default nextConfig;
