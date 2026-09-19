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