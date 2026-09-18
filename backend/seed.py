"""知识库种子文档灌入脚本（部署专用，本地开发不需要跑）。

.gitignore 排除了 uploads/ 和 chroma_db/，代码推上创空间后知识库是全空的，所以容器每次启动都跑一遍把演示文档灌回去。
幂等性靠“按文件名查、已存在就整个跳过”保证；不按内容指纹查是因为本地文件是 CRLF/无末尾换行、而这里的字符串常量用换行符分隔，
字节不同指纹也不同，按指纹查会多灌一份同名副本，导致检索时出现重复引用卡片。
"""
import os

from app.config import UPLOAD_DIR, collection
from app.rag import split_text

# 语料目录：public/ 下为公开文档（private=False），private/ 下为私有文档（private=True）。
# 正文不再写死，改为随仓库提交的 seed_docs/*.txt——改语料只改文件、不动代码。
SEED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_docs")


def _load_seed_docs():
    """遍历 seed_docs/public 与 seed_docs/private，产出 (filename, text, private)。"""
    docs = []
    for sub, private in (("public", False), ("private", True)):
        d = os.path.join(SEED_DIR, sub)
        if not os.path.isdir(d):
            print(f"[种子] 警告：语料目录不存在 {d}")
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(".txt"):
                with open(os.path.join(d, name), encoding="utf-8") as f:
                    docs.append((name, f.read(), private))
    return docs


def seed():
    """把所有种子文档灌进知识库。已存在的整个跳过，不存在的补上。"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for filename, text, private in _load_seed_docs():
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
            metadatas=[{**m, "filename": filename, "saved_as": save_name, "private": private} for _txt, m in chunk_pairs],
        )
        print(f"[种子] {filename}{'（私有）' if private else ''} 已灌入 {len(chunks)} 个切片")

    print(f"[种子] 完成，知识库当前切片总数: {collection.count()}")


if __name__ == "__main__":
    seed()
