// ===== 知识库文件管理 hook（12.15 模块化拆分：从 page.tsx 抽出）=====
// 专管"跟后端 /api/files* 打交道"的全部逻辑：清单加载、上传（含三态提示）、删除、下载
// 原则不变：后端是真相之源，清单从 Chroma 元数据聚合，前端不做本地记录
import { useEffect, useState } from "react";
import type { KbFile } from "@/types";
import { API_BASE } from "@/lib/api";

export function useKnowledgeFiles() {
  const [files, setFiles] = useState<KbFile[]>([]); // 知识库文件清单（来自后端，非本地记录）
  const [uploading, setUploading] = useState(false);

  const loadFiles = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/files`);
      if (res.ok) setFiles((await res.json()).files);
    } catch {
      // 后端没启动时清单保持为空，不阻断页面
    }
  };

  useEffect(() => {
    loadFiles(); // 首次挂载拉一次真实清单（替代以前的本地记录）
  }, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // 大小预检：超 5MB 当场提示，不等网络往返（后端还有同样规则的最终裁决）
    if (file.size > 5 * 1024 * 1024) {
      alert("文件太大（超过 5MB），请压缩或拆分后再上传");
      e.target.value = "";
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file); // "file" 这个名字要和后端 UploadFile 的参数名一致

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData, // ⚠️ 注意：发 FormData 千万不要手动设置 Content-Type！
      });

      if (!res.ok) {
        // 后端返回的错误信息：409 重复上传（“该文件已上传，禁止重复上传…”）、400 类型/大小/解析问题等；
        // 响应体异常时兜底默认文案，绝不静默
        let detail = "上传失败，请稍后再试";
        try {
          const err = await res.json();
          if (err?.detail) detail = err.detail;
        } catch {
          // 响应体不是 JSON（如裸 500 文本）就用默认提示
        }
        alert(detail);
      } else {
        // 成功也要有声音：后端用 overwritten 标志区分“全新上传”和“同名覆盖”（10.13）
        let okMsg = "上传成功";
        try {
          const data = await res.json();
          okMsg = data.overwritten
            ? `同名文件覆盖成功：${data.filename}（旧版已清理，新版共 ${data.chunks} 个切片）`
            : `上传成功：${data.filename}（共 ${data.chunks} 个切片）`;
        } catch {
          // 响应体解析失败不影响主流程，用兜底文案
        }
        await loadFiles(); // 上传成功后重新拉后端清单（含新文件的切片数）
        alert(okMsg);
      }
    } catch {
      alert("上传失败：无法连接后端服务，请确认后端已启动");
    } finally {
      setUploading(false);
      e.target.value = ""; // 清空 input，允许再次选择同一个文件（无论成败都要清，否则选同一文件无反应）
    }
  };

  const deleteFile = async (filename: string) => {
    if (!window.confirm(`确定把「${filename}」从知识库删除吗？相关切片会一并删除。`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/files/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail); // 403 禁删名单的三段式文案从这里直达用户（10.13）
        return;
      }
      await loadFiles(); // 删除后刷新清单
    } catch {
      alert("删除失败：无法连接后端服务");
    }
  };

  // 下载知识库原文件：不能用 <a> 直跳（后端报 404 时会把整个 SPA 页面导航走），
  // 改用 fetch + blob：成功才触发下载，失败把后端 detail 弹出来，页面纹丝不动（与第 6.5 步错误提示原则一致）
  const downloadFile = async (filename: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/files/${encodeURIComponent(filename)}/download`);
      if (!res.ok) {
        let detail = "下载失败";
        try {
          const err = await res.json();
          if (err?.detail) detail = err.detail;
        } catch {
          // 响应体不是 JSON 就用默认提示（同 handleUpload 的兜底写法）
        }
        alert(detail);
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob); // 把文件流变成临时地址，交给浏览器下载后立刻释放，不留内存尾巴
      const a = document.createElement("a");
      a.href = url;
      a.download = filename; // 指定下载后的文件名（不指定就是哈希名）
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      alert("下载失败：无法连接后端服务，请确认后端已启动");
    }
  };

  return { files, uploading, handleUpload, deleteFile, downloadFile };
}
