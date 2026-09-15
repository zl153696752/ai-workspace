# ===== 技能包（Skill）加载器 =====
# Skill（Agent Skills 开放标准）= 把“某类任务按什么章法做”写成一个文件夹，核心是 SKILL.md（YAML 头 + Markdown 正文），
# 就是躺在磁盘上的文件，宿主扫到就能用。对比 MCP：MCP 管“能连哪些外部工具”（要拉子进程），Skill 管“拿到工具后怎么把事做对”（纯知识、零运行时）。
# 本模块实现 Skill 的核心思想——渐进式披露：第一层启动时只扫 name+description 拼成清单常驻系统提示词；
# 第二层模型判断需要时调 load_skill 才把完整正文读进上下文。不全塞正文是因为每次请求都要付 token、且提示词越长注意力越涣散。
import os    # 目录扫描与路径拼接
import yaml  # 解析 SKILL.md 的 YAML 头（PyYAML，随 langchain 装进来，非新增依赖）

# 技能包根目录 backend/skills/；用 __file__ 推导而非写死绝对路径，保证换机器/换盘符都能跑
SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")

# 两份缓存对应渐进式披露两层：_skills_cache 技能清单、_bodies_cache 技能正文。
# 技能文件运行期不变而每次请求都可能用到，缓存避免每个请求都重新遍历目录/读文件/解析 YAML。
_skills_cache = None
_bodies_cache = {}


def _split_frontmatter(raw: str):
    """把 SKILL.md 原始内容拆成「元数据字典 + 正文文本」。

    参数 raw：整个文件文本；返回：(meta, body)，meta 是 YAML 头解析的字典（无头则空字典），body 是去头后的正文。
    格式约定：以一行 --- 开头，接若干行 YAML，再一行 --- 结束，之后全是正文。
    """
    text = raw.lstrip("\ufeff")        # 去掉 Windows 记事本可能写入的 BOM 头，否则 startswith 判断会失败
    if not text.startswith("---"):
        return {}, text                # 没有元数据头：整个文件当正文（容错，不报错）
    # split("---", 2) 限制最多切两刀 → ['', 'YAML头', '正文']；不限制的话正文里的 Markdown 分隔线 --- 会把正文切碎
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text                # 只有开头 --- 没结尾，格式不完整，退回当纯正文
    # 用 safe_load 而非 load：load 能执行 YAML 里的 Python 对象标签，等于给外部文件开代码执行入口；技能文件夹随 git 分发，必须最保守解析
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, parts[2].strip()    # YAML 写错（缩进/冒号）：正文仍可用，元数据当空
    if not isinstance(meta, dict):
        return {}, parts[2].strip()    # 头解析出来不是字典（如只写了一行字符串），同样当空
    return meta, parts[2].strip()


def list_skills() -> list:
    """扫描技能目录，返回技能清单（只含 name 和 description，不含正文）。

    返回：[{"name", "description", "path"}, ...]；agents.py 用它拼系统提示词的技能清单，load_skill 用它做白名单校验。
    三条原则（同 get_mcp_tools 的降级思路）：懒加载+缓存、单个技能出错只跳过不抛异常、目录不存在返回空清单不报错。
    """
    global _skills_cache   # 改模块级变量必须声明 global，否则赋值只会创建同名局部变量
    if _skills_cache is not None:
        return _skills_cache

    found = []
    if not os.path.isdir(SKILLS_DIR):
        print(f"[Skill] 技能目录不存在，本次不启用任何技能: {SKILLS_DIR}")
        _skills_cache = found
        return found

    # sorted 保证扫描顺序稳定（按目录名字母序），日志和提示词里的技能顺序不会每次启动都变
    for entry in sorted(os.listdir(SKILLS_DIR)):
        skill_md = os.path.join(SKILLS_DIR, entry, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue       # 不是技能目录（可能是 README 或空文件夹），跳过
        try:
            with open(skill_md, encoding="utf-8") as f:
                meta, _body = _split_frontmatter(f.read())
        except OSError as e:
            print(f"[Skill] 读取 {entry}/SKILL.md 失败，跳过: {e}")
            continue
        # name/description 是规范的两个必填字段：name 是模型调 load_skill 传的唯一标识，
        # description 决定模型要不要加载这个技能，措辞质量直接影响触发准确率
        name = str(meta.get("name") or "").strip()
        description = str(meta.get("description") or "").strip()
        if not name or not description:
            print(f"[Skill] {entry}/SKILL.md 缺少 name 或 description，跳过")
            continue
        if name != entry:
            # 规范要求 name 与父目录名一致（看目录即知有哪些技能）；不一致只提醒不拒绝，按 name 注册（模型看到的是 name）
            print(f"[Skill] 提醒：{entry}/SKILL.md 的 name「{name}」与目录名不一致，建议改成一致")
        found.append({"name": name, "description": description, "path": skill_md})

    _skills_cache = found
    print(f"[Skill] 已加载技能清单: {[s['name'] for s in found]}")   # 启动观察点：打在后端终端
    return found


def load_skill(skill_name: str) -> str:
    """按名字取出一个技能的完整正文，作为工具返回值喂回给模型。

    参数 skill_name：模型在工具调用里传的技能名；返回：SKILL.md 正文（不含 YAML 头）。
    出错时返回一句说明性“人话”、绝不返回空串（同 search_knowledge_base_lc）：模型看到明确失败说明才会如实告知或换办法，空串会被当成“工具坏了”甚至瞎编。
    """
    available = list_skills()

    # ===== 白名单校验：本函数最重要的安全边界，不能省 =====
    # skill_name 是模型生成的不可信输入，可能被对话诱导（提示注入）。直接拼路径的话 load_skill("../../.env") 就能读到密钥文件。
    # 防法不是过滤 ../（黑名单总能被绕过），而是只认清单里已有的名字、路径一律取自清单记录的 path 字段，用户传的字符串全程不参与路径拼接。
    target = next((s for s in available if s["name"] == skill_name), None)
    if target is None:
        names = ", ".join(s["name"] for s in available) or "（当前没有可用技能）"
        return f"没有名为「{skill_name}」的技能，可用技能：{names}"

    if skill_name in _bodies_cache:
        return _bodies_cache[skill_name]     # 已读过：直接用内存里的正文，不再碰磁盘

    try:
        with open(target["path"], encoding="utf-8") as f:
            _meta, body = _split_frontmatter(f.read())
    except OSError as e:
        return f"技能「{skill_name}」的文件读取失败：{e}"

    if not body:
        # 文件在但正文空（只写了 YAML 头）：如实说明，让模型别硬编内容
        return f"技能「{skill_name}」的正文为空，请检查 SKILL.md 是否只写了元数据头"

    _bodies_cache[skill_name] = body
    print(f"[Skill] 已加载技能正文: {skill_name}（{len(body)} 字）")   # 观察点：能看到模型这次调了哪个技能
    return body
