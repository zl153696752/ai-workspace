# ===== 技能包（Skill）加载器 =====
# 先搞清 Skill 是什么（Agent Skills 开放标准，规范站点 agentskills.io）：
#   把“某类任务该按什么章法做”的经验写成一个文件夹，核心是一个 SKILL.md
#   （YAML 元数据头 + Markdown 正文），可以附带脚本、模板、参考资料。
#   它不是服务：不用启动进程、不用握手连接、不用注册端口——就是躺在磁盘上的文件，宿主扫到就能用。
#   对比一下已经接过的 MCP：MCP 管“能连上哪些外部工具”（要拉子进程、走协议），
#   Skill 管“拿到工具后按什么章法把事做对”（纯知识，零运行时）。
#
# 本模块实现的是 Skill 机制的核心思想——渐进式披露（progressive disclosure），分两层：
#   第一层：启动时只扫每个技能的 name + description，拼成一份清单常驻系统提示词（约几十到两百 token）
#   第二层：模型判断这次任务需要某个技能时，调 load_skill 工具，才把完整正文读进上下文（上千 token）
# 为什么不干脆把正文全塞进系统提示词：
#   ① 每次请求都要为它付 token 钱，哪怕用户只是说“你好”；
#   ② 提示词越长模型注意力越涣散，关键规则会被淹没在无关内容里。
# 这套机制在 Claude Code、Codex 等产品里由宿主实现，本项目是自己手写一遍——
# 手写之后会发现它并不神秘：本质就是“按需把一段 Markdown 注入上下文”，而“按需”由模型通过工具调用来决定。
import os    # 目录扫描与路径拼接
import yaml  # 解析 SKILL.md 的 YAML 元数据头（PyYAML，随 langchain 一起装进来的，不是新增依赖）

# 技能包根目录：backend/skills/
# 用 __file__ 推导而不是写死绝对路径，理由和 config.py 的 UPLOAD_DIR 一样：项目挪到任何机器、任何盘符都能跑。
SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")

# 两份缓存，对应渐进式披露的两层：
#   _skills_cache —— 技能清单 [{"name", "description", "path"}]，启动后第一次用到时扫描一次
#   _bodies_cache —— 技能正文 {name: 正文文本}，某个技能第一次被加载时才读盘
# 为什么要缓存：技能文件在服务运行期间不会变，而每次对话请求都可能用到；
# 不缓存的话每个请求都要重新遍历目录、读文件、解析 YAML，白耗磁盘 IO。
_skills_cache = None
_bodies_cache = {}


def _split_frontmatter(raw: str):
    """把 SKILL.md 的原始内容拆成「元数据字典 + 正文文本」两部分。

    参数 raw：整个文件的文本内容
    返回：(meta, body) 元组。meta 是 YAML 头解析出的字典（没有头就是空字典），body 是去掉头之后的正文

    文件格式约定：文件以一行 --- 开头，接若干行 YAML，再一行 --- 结束，之后全是正文。
    """
    text = raw.lstrip("\ufeff")        # 去掉 Windows 记事本等工具可能写入的 BOM 头，否则 startswith 判断会失败
    if not text.startswith("---"):
        return {}, text                # 没有元数据头：整个文件当正文（不报错，容错处理）
    # split("---", 2) 的 2 是“最多切两刀”，切成三段：['', 'YAML头', '正文']。
    # 必须限制刀数：正文里可能出现 Markdown 分隔线 ---，不限制的话正文会被切碎。
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text                # 只有开头的 --- 没有结尾的，格式不完整，退回当纯正文
    # 用 safe_load 而不是 load：load 能执行 YAML 里写的 Python 对象标签，等于给外部文件开了代码执行入口。
    # 技能文件夹是会跟着 git 分发的，别人也能往里塞内容，所以必须用最保守的解析方式。
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, parts[2].strip()    # YAML 写错了（缩进、冒号问题）：正文还是能用，元数据当空处理
    if not isinstance(meta, dict):
        return {}, parts[2].strip()    # 头部解析出来不是字典（比如只写了一行字符串），同样当空处理
    return meta, parts[2].strip()


def list_skills() -> list:
    """扫描技能目录，返回技能清单（只含 name 和 description，不含正文）。

    返回：[{"name": 技能名, "description": 适用场景说明, "path": SKILL.md 完整路径}, ...]
    调用方：agents.py 用它拼系统提示词里的“可用技能清单”，load_skill 用它做白名单校验。

    三条设计原则（和 get_mcp_tools 的降级思路一致）：
    - 懒加载 + 缓存：第一次调用才扫盘，之后直接返回内存里的清单
    - 单个技能出问题不影响其他技能：解析失败就跳过它、打一条日志，不抛异常
    - 目录整个不存在也不算错误：返回空清单，牛来照常工作，只是没有技能可用
    """
    global _skills_cache   # 要改模块级变量必须声明 global，否则赋值只会创建一个同名局部变量
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
        # name 和 description 是规范里的两个必填字段：
        #   name        —— 技能唯一标识，模型调 load_skill 时传的就是它
        #   description —— 什么时候该用这个技能。模型完全靠这段话决定要不要加载，
        #                  所以它的措辞质量直接决定触发准确率（和工具说明书的 description 是同一个道理）
        name = str(meta.get("name") or "").strip()
        description = str(meta.get("description") or "").strip()
        if not name or not description:
            print(f"[Skill] {entry}/SKILL.md 缺少 name 或 description，跳过")
            continue
        if name != entry:
            # 规范要求 name 必须与父目录名一致（这样看目录就知道有哪些技能）。
            # 不一致时不拒绝加载，只提醒一句——按 name 注册，因为模型看到的是 name。
            print(f"[Skill] 提醒：{entry}/SKILL.md 的 name「{name}」与目录名不一致，建议改成一致")
        found.append({"name": name, "description": description, "path": skill_md})

    _skills_cache = found
    print(f"[Skill] 已加载技能清单: {[s['name'] for s in found]}")   # 启动观察点：打在后端终端
    return found


def load_skill(skill_name: str) -> str:
    """按名字取出一个技能的完整正文，作为工具返回值喂回给模型。

    参数 skill_name：模型在工具调用里传进来的技能名
    返回：SKILL.md 的正文文本（不含 YAML 头）；出错时返回一句说明性的“人话”，绝不返回空字符串

    为什么出错也要返回人话而不是抛异常或返回空串（和 search_knowledge_base_lc 同一个道理）：
    模型看到明确的失败说明，才会如实告诉用户或换个办法；
    返回空串它容易理解成“工具坏了”，甚至干脆凭自己的想象编一套答案。
    """
    available = list_skills()

    # ===== 白名单校验：这是本函数最重要的安全边界，不能省 =====
    # skill_name 是模型生成的参数，属于不可信输入，而模型的决定可能被用户对话诱导（提示注入）。
    # 如果拿它直接拼路径，load_skill("../../.env") 就能读到存放 API 密钥的文件，
    # load_skill("../../app/config") 就能读到源码。
    # 防法不是去过滤 ../ 这种字符（黑名单总能被绕过），而是反过来：
    # 只认扫描出来的清单里已有的名字，路径一律取自清单里那条记录的 path 字段，
    # 用户传进来的字符串全程不参与任何路径拼接。不在清单里就直接拒绝。
    target = next((s for s in available if s["name"] == skill_name), None)
    if target is None:
        names = ", ".join(s["name"] for s in available) or "（当前没有可用技能）"
        return f"没有名为「{skill_name}」的技能，可用技能：{names}"

    if skill_name in _bodies_cache:
        return _bodies_cache[skill_name]     # 已经读过一次：直接用内存里的正文，不再碰磁盘

    try:
        with open(target["path"], encoding="utf-8") as f:
            _meta, body = _split_frontmatter(f.read())
    except OSError as e:
        return f"技能「{skill_name}」的文件读取失败：{e}"

    if not body:
        # 文件在但正文是空的（只写了 YAML 头）：如实说明，让模型别硬编内容
        return f"技能「{skill_name}」的正文为空，请检查 SKILL.md 是否只写了元数据头"

    _bodies_cache[skill_name] = body
    print(f"[Skill] 已加载技能正文: {skill_name}（{len(body)} 字）")   # 观察点：能看到模型这次到底调了哪个技能
    return body
