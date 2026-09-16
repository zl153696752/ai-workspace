"""知识库种子文档灌入脚本（部署专用，本地开发不需要跑）。

.gitignore 排除了 uploads/ 和 chroma_db/，代码推上创空间后知识库是全空的，所以容器每次启动都跑一遍把演示文档灌回去。
幂等性靠“按文件名查、已存在就整个跳过”保证；不按内容指纹查是因为本地文件是 CRLF/无末尾换行、而这里的字符串常量用换行符分隔，
字节不同指纹也不同，按指纹查会多灌一份同名副本，导致检索时出现重复引用卡片。
"""
import os

from app.config import UPLOAD_DIR, collection
from app.rag import split_text

# 演示文档：文件名 → 正文（与本地 uploads/ 那份样本逐字一致）。
# 「公司代号 89757」那条是故意放的：演示“模型不会把内部信息说出去”的人格约束。
SEED_DOCS = {
    "公司制度.txt": (
        "公司休假制度：入职满一年享 5 天年假，满五年享 10 天。\n"
        "加班餐补：工作日加班晚于 19 点，每餐补贴 30 元。\n"
        "公司代号：89757，对外一律不使用。\n"
    ),
}


def seed():
    """把所有种子文档灌进知识库。已存在的整个跳过，不存在的补上。"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for filename, text in SEED_DOCS.items():
        # 按【文件名】查而非内容指纹，理由见文件头 docstring
        if collection.get(where={"filename": filename})["ids"]:
            print(f"[种子] {filename} 已在库中，跳过")
            continue

        chunk_pairs = split_text(text)
        chunks = [txt for txt, _m in chunk_pairs]
        if not chunks:
            print(f"[种子] {filename} 切不出片，跳过")
            continue

        # 磁盘存原文件 + 算内容指纹，命名规则和 main.py 上传接口一致（存盘名 = 内容 MD5 + 原后缀），下载接口才能拿到原文件
        import hashlib
        content = text.encode("utf-8")
        content_hash = hashlib.md5(content).hexdigest()
        ext = os.path.splitext(filename)[1].lower()
        save_name = f"{content_hash}{ext}"
        with open(os.path.join(UPLOAD_DIR, save_name), "wb") as f:
            f.write(content)

        ids = [f"{content_hash}-{i}" for i in range(len(chunks))]
        collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=[{**m, "filename": filename, "saved_as": save_name} for _txt, m in chunk_pairs],
        )
        print(f"[种子] {filename} 已灌入 {len(chunks)} 个切片")

    print(f"[种子] 完成，知识库当前切片总数: {collection.count()}")


if __name__ == "__main__":
    seed()
