# ===== 鉴权模块（步骤4：JWT 签发/校验 + 口令验证 + 身份依赖）=====
# 职责：把"你是谁"收口到一个模块——登录时验口令、发门票(JWT)；每次请求时验门票、解析身份。
# 设计：无状态。服务端不存任何会话表，身份全写在门票里、靠签名防伪（原理见方案模块3）。
#
# 两种身份：
#   游客 guest —— 没带门票 / 门票无效或过期，只能用公共内容；
#   亮哥 liang —— 带有效门票，解锁私人内容、可上传/删除。
import datetime as dt

import bcrypt          # 口令哈希：checkpw 把"输入口令"与"存的哈希指纹"比对（依赖已在 requirements）
import jwt             # PyJWT：签发/校验 JWT 门票（依赖已在 requirements）
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import JWT_SECRET, ADMIN_PASSWORD_HASH, TOKEN_EXPIRE_DAYS

_ALGORITHM = "HS256"   # JWT 签名算法：HMAC-SHA256，用同一密钥签和验（对称），单后端标准选择
LIANG = "liang"        # 门票里代表亮哥的身份标识（payload 的 sub 字段），本项目唯一特权身份


# ===== 口令验证 =====
def verify_password(plain: str) -> bool:
    """把用户输入的明文口令与 .env 里存的 bcrypt 哈希比对，一致返回 True。

    🔴 绝不存/比明文：.env 里是哈希指纹，这里把输入现场哈希再比。
    未配置哈希（空串）时直接 False —— 宁可谁都登不进，也不"空口令放行"。
    bcrypt 只吃 bytes，故两边都 encode 成 utf-8。
    """
    if not ADMIN_PASSWORD_HASH:
        print("[鉴权] 未配置 ADMIN_PASSWORD_HASH，拒绝所有登录（生成方式见本文件末尾小工具）")
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), ADMIN_PASSWORD_HASH.encode("utf-8"))
    except Exception as e:
        print(f"[鉴权] 口令校验异常，判为失败：{e}")   # 哈希串损坏也不抛给用户，记一笔判失败
        return False


# ===== 门票签发 / 校验 =====
def issue_token(identity: str = LIANG) -> str:
    """签发一张 JWT 门票：payload 写清"是谁(sub) + 何时签发(iat) + 何时过期(exp)"，用 JWT_SECRET 签名。

    门票三段式 头部.内容.签名：内容(base64)谁都能看，但【签名】只有持有 JWT_SECRET 的后端算得出，
    别人改了内容(比如把游客改成亮哥)签名就对不上，一验即穿。
    """
    now = dt.datetime.now(dt.timezone.utc)   # 统一用带时区的 UTC，避免本地时区导致 exp 计算错乱
    payload = {
        "sub": identity,                                     # 身份标识
        "iat": now,                                          # 签发时间
        "exp": now + dt.timedelta(days=TOKEN_EXPIRE_DAYS),  # 过期时间（30 天）
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=_ALGORITHM)


def decode_token(token: str) -> dict | None:
    """校验门票签名与有效期，通过则返回 payload（含 sub），否则返回 None（无效/过期/被篡改一律 None）。"""
    try:
        # decode 会自动验签名 + 验 exp（过期抛 ExpiredSignatureError），任一不过都进 except
        return jwt.decode(token, JWT_SECRET, algorithms=[_ALGORITHM])
    except jwt.PyJWTError as e:
        print(f"[鉴权] 门票校验未通过：{e}")
        return None


# ===== FastAPI 身份依赖（4b 挂到接口上用）=====
# HTTPBearer：从请求头 Authorization: Bearer <token> 抽出 token。
# auto_error=False：没带头时不自动抛 403、而是给 None —— 因为【游客是合法身份】，不能一没 token 就报错。
_bearer = HTTPBearer(auto_error=False)


def get_identity(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    """解析本次请求身份，返回 {"is_liang": bool}。这是"可选鉴权"：
    带有效门票 → 亮哥；没带 / 无效 / 过期 → 游客（不报错）。chat、清单这类接口用它。
    """
    if creds is None:
        return {"is_liang": False}
    payload = decode_token(creds.credentials)
    return {"is_liang": bool(payload and payload.get("sub") == LIANG)}


def require_liang(identity: dict = Depends(get_identity)) -> dict:
    """强制亮哥身份的守卫：游客一律 403。上传、删除这类"写"操作挂它。
    依赖可叠加：先跑 get_identity 拿身份，再判断——不是亮哥直接 403，接口函数根本不执行。
    """
    if not identity["is_liang"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="该操作仅限亮哥登录后使用",
        )
    return identity


# ===== 一次性小工具：生成口令哈希 + 开发用密钥（直接运行本文件即可，不依赖服务）=====
# 用法：backend 目录跑  python -m app.auth  → 按提示输入口令 → 把打印的两行抄进 backend/.env
if __name__ == "__main__":
    import getpass
    import secrets
    pwd = getpass.getpass("请输入亮哥登录口令（输入时不显示）：")
    if not pwd:
        print("口令为空，已取消。")
    else:
        hashed = bcrypt.hashpw(pwd.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        print("\n把下面两行抄进 backend/.env（ADMIN_PASSWORD_HASH 是一整行，别断开）：\n")
        print(f"ADMIN_PASSWORD_HASH={hashed}")
        print(f"JWT_SECRET={secrets.token_urlsafe(32)}")
        print("\n（JWT_SECRET 若线上已在环境变量设了就不用这行；本地开发抄进去即可）")