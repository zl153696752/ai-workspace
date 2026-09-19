# ===== SQLite 持久化基础设施（步骤9 建，步骤10 长期记忆复用同一套连接管理）=====
# 🔴 铁律：只用标准库 sqlite3、禁 ORM——表结构简单、SQL 直观、零额外依赖、便于调试（项目既定规范）。
import sqlite3
import os
import json
from contextlib import contextmanager

# db 文件落在 backend/ 下（与 chroma_db 平级）。__file__=app/db.py，dirname 两次到 backend。
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "niulai.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # 结果能按列名取（row["query"]），比下标直观
    return conn


@contextmanager
def _cursor():
    """统一连接管理：正常退出 commit、异常 rollback、无论如何 close（防连接泄漏/忘提交）。
    所有读写都走它——步骤10 的记忆读写也复用这个 contextmanager。"""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """建表（幂等：IF NOT EXISTS，重复调不清数据、不报错）。应用启动时调一次。
    步骤9 先建 traces；步骤10 再往这里追加 memories 表的 CREATE，共用本模块。"""
    with _cursor() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                TEXT    DEFAULT (datetime('now','localtime')),
                query             TEXT,      -- 用户问了什么
                identity          TEXT,      -- guest / liang
                intents           TEXT,      -- 意图（kb/tool/chitchat）
                scope             TEXT,      -- 作用域 company/general/both
                degraded          INTEGER,   -- 本次问答是否触发过降级（0/1）
                calls             INTEGER,   -- 模型调用次数
                total_latency     REAL,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                total_cost        REAL,
                llm_spans         TEXT,      -- JSON：模型调用级 span 明细（逐次 token/延迟/成本）
                node_spans        TEXT       -- JSON：节点级 span 明细（意图/检索质检/降级/引用检查）
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user        TEXT NOT NULL,    -- 只记 'liang'；游客不记（隐私 + 降风险）
                type        TEXT,             -- 'preference'(偏好) / 'fact'(稳定事实)，只这两类
                content     TEXT,             -- 记忆正文，描述性措辞（如“亮哥偏好详细讲解”），不用指令口吻
                confidence  REAL DEFAULT 1.0, -- 抽取置信度 0~1，低于阈值不注入（防错记污染）
                source      TEXT,             -- 从哪句原话抽的，可回溯核查
                topic       TEXT,             -- 主题键：同 user+topic 只留最新一条（去重更新，防自相矛盾）
                created_at  TEXT DEFAULT (datetime('now','localtime')),
                updated_at  TEXT DEFAULT (datetime('now','localtime'))
            )
        """)


def save_trace(*, query, identity, intents, scope, degraded, summary, llm_spans, node_spans):
    """一次问答的全链路 trace 落一行：汇总指标拆成【列】(方便聚合查询)，两类 span 明细各存【JSON】。"""
    with _cursor() as conn:
        conn.execute("""
            INSERT INTO traces
                (query, identity, intents, scope, degraded, calls,
                 total_latency, prompt_tokens, completion_tokens, total_cost,
                 llm_spans, node_spans)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            query, identity, intents, scope, int(bool(degraded)), summary.get("calls", 0),
            summary.get("total_latency", 0.0), summary.get("prompt_tokens", 0),
            summary.get("completion_tokens", 0), summary.get("total_cost", 0.0),
            json.dumps(llm_spans, ensure_ascii=False),
            json.dumps(node_spans, ensure_ascii=False),
        ))


# ===== 步骤11：长期记忆 CRUD（只服务亮哥，游客不记）=====
def upsert_memory(*, user, type, content, confidence, source, topic):
    """写一条记忆，按 (user, topic) 去重更新：同主题已存在就覆盖（更新优于追加，防自相矛盾），否则新增。
    🔴 去重键用 topic 不用 content——“喜欢详细讲解”和“偏好讲细一点”content 不同、topic 都是“回答风格”，该合并成一条。"""
    with _cursor() as conn:
        row = conn.execute("SELECT id FROM memories WHERE user=? AND topic=?", (user, topic)).fetchone()
        if row:
            conn.execute("""
                UPDATE memories SET type=?, content=?, confidence=?, source=?,
                       updated_at=datetime('now','localtime') WHERE id=?
            """, (type, content, confidence, source, row["id"]))
            return row["id"]
        cur = conn.execute("""
            INSERT INTO memories (user, type, content, confidence, source, topic)
            VALUES (?,?,?,?,?,?)
        """, (user, type, content, confidence, source, topic))
        return cur.lastrowid


def get_memories(user, *, min_confidence=0.6, limit=20):
    """读某用户可注入的记忆：滤掉低置信、按更新时间倒序、限量防 prompt 膨胀。返回 [dict]。"""
    with _cursor() as conn:
        rows = conn.execute("""
            SELECT id, type, content, confidence, source, topic, updated_at
            FROM memories WHERE user=? AND confidence>=?
            ORDER BY updated_at DESC LIMIT ?
        """, (user, min_confidence, limit)).fetchall()
        return [dict(r) for r in rows]


def delete_memory(memory_id, user):
    """删一条记忆（治理入口用）。带 user 校验：只能删自己的，防串号误删他人记忆。返回是否删到。"""
    with _cursor() as conn:
        cur = conn.execute("DELETE FROM memories WHERE id=? AND user=?", (memory_id, user))
        return cur.rowcount > 0