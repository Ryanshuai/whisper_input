"""Build an ASR biasing context from "what the user is working on right now".

Both backends accept a free-text biasing hint, so this module produces ONE string
that main.py hands to `asr.transcribe(..., context=...)`:

- Qwen3-ASR  → goes in the chat-template **system** slot (its native contextual
  biasing; the HF helper only exposes `language` there, but the template
  concatenates arbitrary system text, so we build the conversation ourselves).
- faster-whisper → appended to `hotwords`.

Measured on real captures from the dictation log, with a hint built from the
current Claude Code session, all three of these flipped to correct:
    兀兀兀松 → 雾凇     side car → sidecar     三个有田 → 三个有填

Sources (each individually switchable in config under `asr_context`):
  claude_session — the most recently active ~/.claude/projects/*.jsonl. Highest
                   hit rate: dictation is mostly *spoken into* Claude Code, so the
                   words about to be said are already in that session.
  clipboard      — the current clipboard (secret-shaped content is dropped).
  window         — the active X11 window title + recent GTK documents.

Output shape is "terms + prose": a deduped term list gives proper nouns a
concrete anchor, and a few verbatim recent lines give the model register and
sentence shape. Everything is capped by `max_chars`.

Built on a worker thread at dictation START (recording lasts seconds; the cost is
hidden) and consumed at STOP, so the hotkey path never blocks on disk or X11.
"""

import os
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter

HOME = os.path.expanduser('~')

# --- term extraction tuning -------------------------------------------------

# Latin tokens that are never worth biasing toward: they carry no identity, and
# spending term slots on them pushes real jargon out of the list.
_LATIN_STOP = frozenset('''
the a an and or but if then else for while of to in on at by with from into over
under this that these those is are was were be been being do does did done have
has had will would can could should may might must not no yes it its as so than
you your we our they their he she him her his i me my one two three four five
new old get set add remove use used using make made run ran see saw know known
what when where which who why how all any some more most much many very just
about after before again also into out up down off other same such only own
http https www com org net html json yaml file files line lines code text data
'''.split())

# Frequent Chinese characters. A CJK n-gram made *entirely* of these is ordinary
# speech ("的时候", "这个", "就是说") and gets dropped; one uncommon character is
# enough to mark a term worth biasing (雾"凇", "沇"...). Deliberately not a full
# frequency table — just dense enough to kill the filler.
_CJK_COMMON = set(
    '的一是不了在人有我他这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感'
    '见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重三机工物气每并别真打太新比才便夫再书部水像眼等体却加电主界门利海受听表德少克代员许稜先口由死安写性马光白或住难望教命花结乐色更拉东神记处让母父应直字场平报友关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原吃妈变通师立象数四失满战远格士音轻目条呢病始达深完今提求清王化空业思切怎非找片罗钱吗语元喜曾离飞科言干流欢约各即指合反题必该论交终林请医晚制球决传画保读运及则房早院量苦火布品近坐产答星精视五连司巴奇管类未朋且婚台夜青北队久乎越观落尽形影红爸百令周吧识步希亚术留市半热送兴造谈容极随演收首根讲整式取照办强石古华諣拿计您装似足双妻尼转诉米称丽客南领节衣站黑刻统断福城故历惊脸选包紧争另建维绝树系伤示愿持千史谁准联妇纪基买志静阿诗独复痛消社算义竟确酒需单治卡幸兰念举仅钟怕共毛句息功官待究跟穿室易游程号居考突皮哪费倒价图具刚脑永歌响商礼细专黄块脚味灵改据般破引食仍存众注笔甚某沉血备习校默务土微娘须试怀料调广蜖苏显赛查密议底列富梦错座参八除跑亮假印设线温虽掉京初养香停际致阳纸李纳验助激够严证帝饭忘趣支春集丈木研班普导顿睡展跳获艺六波察群皇段急庭创区奥器谢弟店否害草排背止组州朝封睛板角况曲馆育忙质河续哥呼若推境遇雨标姐充围案伦护冷警贝著雪索剧啊船险烟依斗值帮汉慢佛肯闻唱沙局伯族低玩资屋击速顾泪洲团圣旁堂兵七露园牛哭旅街劳型烈姑陈莫鱼异抱宝权鲁简态级票怪寻杀律胜份汽右洋范床舞秘午登楼贵吸责例追较职属渐左录丝牙品塔庄未莱阵允宗英雄'
    # Second tier. The first tier alone left ordinary words looking "rare", so
    # grammatical debris ("是以逆", "减掉自己") kept passing the term filter.
    '词库测翻减宽批散槽逆负载值域项类型码端源版块段落层级组件模式状态设置属性参数返回调用执行结果输出输入'
    '编写读取删除增加修改查找替换排序过滤统计计算比较判断循环条件跳转标记注释文档说明例子实现方法接口抽象'
    '继承封装重构优化性能效率速度延迟带宽容量规模数量比例百分点均值方差偏差误差精度准确召回覆盖'
    '训练推理评测验证测试集样本标注特征向量矩阵张量维度通道尺寸分辨率图像视频音频文本语音识别合成翻译'
    '模型权重梯度损失学习速率批次轮次收敛过拟合正则化归一激活函数卷积池化注意力编码解码'
    '服务客户端请求响应接收发送连接断开超时重试失败成功异常错误日志监控告警恢复备份'
    '目录文件夹路径名称后缀扩展格式编码解析生成导入导出上传下载同步异步并发线程进程锁队列缓存'
    '版本分支合并提交推送拉取冲突回滚标签发布部署构建打包安装卸载依赖环境配置变量'
    '界面窗口按钮菜单列表表格输入框滚动刷新加载显示隐藏切换选中取消确认提示弹窗'
    '需求设计方案架构模块组织结构流程步骤阶段计划目标范围边界约束假设风险问题原因分析'
    '整个具体大概差不多至少最多完全基本主要次要重要关键核心简单复杂清楚明白理解解决处理'
    '所以但是不过而且然后接着首先其次最后另外例如比如因为由于关于对于按照根据通过'
    '现在刚才马上立刻已经正在将要曾经从来一直总是有时偶尔经常很少从不'
)

# Path furniture that rides in on window titles and recent-document names but
# says nothing about vocabulary. The login name is added at import: it prefixes
# every path on the machine, so it would otherwise rank near the top.
_PATH_STOP = {
    'home', 'usr', 'var', 'tmp', 'opt', 'etc', 'src', 'lib', 'bin', 'doc', 'docs',
    'downloads', 'desktop', 'documents', 'pictures', 'videos', 'music', 'screenshots',
    'untitled', 'index', 'main', 'test', 'tests', 'build', 'dist', 'node_modules',
    os.path.basename(HOME).lower(),
    # Application chrome and source-code literals. Same category — they ride in
    # on titles and on Claude's code blocks, recur in every single session, and
    # are never spoken. In the persistent store they had accumulated k=171
    # sessions apiece, which put `Visual` and `Studio` above every real term the
    # user has; and unlike ordinary jargon the said-counter cannot demote them,
    # because "never once uttered" is also what a permanently-misheard word
    # looks like.
    'visual', 'studio', 'vscode', 'chrome', 'firefox', 'google', 'nautilus',
    'none', 'true', 'false', 'null',
}

_CJK_RE = re.compile(r'[一-鿿]+')
# Identifiers and jargon: keeps snake_case, kebab-case, dotted names and CamelCase
# as single tokens ("train_need", "rime-ice", "Qwen3-ASR", "recently-used.xbel").
_LATIN_RE = re.compile(r'[A-Za-z][A-Za-z0-9]*(?:[_\-.][A-Za-z0-9]+)*')

# Clipboard contents matching any of these are skipped wholesale — a clipboard is
# where passwords and tokens live, and biasing text is not worth leaking one into
# a prompt (or into the dictation debug dump).
_SECRET_RE = re.compile(
    r'(?i)(?:'
    r'-----BEGIN[ A-Z]*PRIVATE KEY|'
    r'\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}|'
    r'\bgh[pousr]_[A-Za-z0-9]{20,}|'
    r'\bAKIA[0-9A-Z]{12,}|'
    r'\bxox[baprs]-[A-Za-z0-9\-]{10,}|'
    r'\b(?:pass(?:wo?rd)?|passwd|secret|token|api[_\-]?key)\b\s*[:=]|'
    r'\b[A-Fa-f0-9]{40,}\b'
    r')'
)


def _is_secretish(text: str) -> bool:
    return bool(_SECRET_RE.search(text))


def _norm_for_match(text: str) -> str:
    """Whitespace/punctuation-insensitive key for 'is this the same utterance?'.

    Claude Code stores a pasted transcript with its own trimming, and a trailing
    Enter can shave characters, so exact equality misses.
    """
    return re.sub(r'[\s，。,.、！？!?；;：:「」“”"\'`]+', '', text)


def _is_prose(line: str) -> bool:
    """Rough 'is this a sentence rather than a path/URL/identifier?' test."""
    if line.startswith(('/', '~', 'http://', 'https://', 'file://')):
        return False
    has_cjk = bool(_CJK_RE.search(line))
    return has_cjk or ' ' in line.strip()


_SENT_RE = re.compile(r'[\n。！？；]|(?<=[.!?])\s')
_MD_RE = re.compile(r'[`*#>|_\[\]]+')


def _assistant_snippets(assistants: list[str], max_chars: int, reject=None) -> list[str]:
    """Quotable prose lines out of Claude's replies, newest first.

    The snippet channel used to be fed by user turns alone, on the reasoning that
    assistant replies are too verbose to quote and written in the wrong register.
    That holds for a whole reply — but it silently assumed user prose would be
    *there*, and for a user who dictates nearly every message it is not: every
    user turn is our own transcript, so note_dictation() subtracts all of them and
    the prose section comes out empty. Measured live: 0 chars of context.

    A reply split to sentences survives *that* objection — a single sentence is
    still prose the model has seen, topical, carrying the jargon's correct
    spelling in situ. What it does not survive is measurement: as a top-up to the
    user's own prose it is neutral, and on its own it is worse than sending
    nothing. So collect() gates it on there being user prose to top up, and the
    dictation-only case this was written for stays empty on purpose. See the
    comment at the call site for the numbers.

    `reject` is ContextBuilder._is_own_output. Claude quotes the user, so a reply
    can contain an ASR error verbatim; without this the self-reinforcement loop
    that note_dictation() exists to cut would simply reopen one source over.
    """
    out, seen = [], set()
    for text in assistants:
        for s in _SENT_RE.split(text):
            s = _MD_RE.sub('', s or '').strip()
            if not (8 <= len(s) <= max_chars) or not _is_prose(s):
                continue
            if s in seen or (reject is not None and reject(s)):
                continue
            seen.add(s)
            out.append(s)
    return out


# ---------------- sources ----------------

def _read_clipboard(is_own, max_chars: int) -> str:
    """Current clipboard, or '' if unusable.

    stop_dictation() copies each transcript to the clipboard, so `is_own` is what
    keeps the previous utterance from feeding itself back in — see
    ContextBuilder.note_dictation for why that loop is worth blocking.
    """
    try:
        import pyperclip
        text = pyperclip.paste() or ''
    except Exception:
        return ''
    text = text.strip()
    if not text or is_own(text) or _is_secretish(text):
        return ''
    return text[:max_chars]


_x_display = None


def _active_window_title() -> str:
    """Active X11 window title via python-xlib (already present as a pynput dep).

    The display connection is cached — reconnecting per dictation is pure latency —
    and dropped on any error so a restarted X server heals on the next call.
    """
    global _x_display
    try:
        from Xlib import display, X
        if _x_display is None:
            _x_display = display.Display()
        d = _x_display
        root = d.screen().root
        prop = root.get_full_property(d.intern_atom('_NET_ACTIVE_WINDOW'), X.AnyPropertyType)
        if not prop or not prop.value:
            return ''
        win = d.create_resource_object('window', prop.value[0])
        for atom in ('_NET_WM_NAME', 'WM_NAME'):
            p = win.get_full_property(d.intern_atom(atom), 0)
            if p and p.value:
                v = p.value
                return v.decode('utf-8', 'replace') if isinstance(v, bytes) else str(v)
        return ''
    except Exception:
        _x_display = None
        return ''


# Recent-documents entries that are noise, not vocabulary: screenshots dominate
# the list and their names are all timestamp.
_DOC_SKIP_RE = re.compile(r'(?i)screenshot|\.(?:png|jpe?g|gif|webp|mp4|mkv|pdf|zip|tar|gz)$')


def _recent_docs(limit: int) -> list[str]:
    """Basenames (and their parent dir) of the newest GTK recently-used entries."""
    path = os.path.join(HOME, '.local/share/recently-used.xbel')
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return []
    items = []
    for bm in root.iter('bookmark'):
        href = bm.get('href') or ''
        if not href.startswith('file://'):
            continue
        items.append((bm.get('modified') or '', href))
    items.sort(reverse=True)
    out: list[str] = []
    for _, href in items:
        p = urllib.parse.unquote(href[7:])
        name = os.path.basename(p.rstrip('/'))
        if not name or _DOC_SKIP_RE.search(name):
            continue
        parent = os.path.basename(os.path.dirname(p.rstrip('/')))
        out.append(f'{parent}/{name}' if parent else name)
        if len(out) >= limit:
            break
    return out


# Claude Code writes one JSONL per session under a directory named after the cwd
# with separators flattened to '-'. Lines we want are user turns whose content
# holds a plain text block; tool results, slash commands, IDE selections and
# injected reminders are machinery, not the user's voice.
_META_RE = re.compile(
    r'<(?:system-reminder|command-name|command-message|command-args|ide_selection|'
    r'local-command-stdout|task-notification)[\s>]', re.I)
_TAG_RE = re.compile(r'<[^>]{1,80}>')

# Claude Code injects prose into the *user* role that the user never typed: IDE
# state, retry nudges, interruption notices. It reads like ordinary English, so
# nothing above catches it — and it is pure noise that once put "Please retry
# tool call" into the term list. Match on the fixed opening phrases.
_HARNESS_RE = re.compile(
    r'^(?:'
    r'The user (?:opened|selected|closed)\b|'
    r'The previous response failed\b|'
    r'Caveat: The messages below\b|'
    r'This session is being continued\b|'
    r'Your task is to create\b|'
    r'API Error\b|'
    r'Request interrupted\b|'
    r'Continue from where\b|'
    r'No response requested\b|'
    r'\[Request interrupted'
    r')', re.I)


# Words that appear in *every* window title or *every* project path here, so
# matching on them tells you nothing about which project you are typing into.
# 'code' is the load-bearing one twice over: it ends every "… - Visual Studio
# Code" title and it names the directory every project lives in.
_PROJECT_STOP = _PATH_STOP | {'code', 'window', 'terminal',
                              'claude', 'project', 'projects'}


def _match_words(words) -> set:
    return {w for w in words if len(w) >= 3 and w not in _PROJECT_STOP}


def _newest_session_file(window_title: str, max_age_sec: float = 1800) -> str:
    """Pick the live Claude Code session, or '' when none is confidently live.

    Getting this wrong is the feature's main way to do harm. Measured on the
    capture corpus (context_ablation.py, 9 probes with known-correct wording):

        no context            4/9 correct
        right session         5/9   (+3 fixes, -2 new errors)
        wrong session         2/9   (-2 net — every 雾凇 probe regressed)

    So a stale or unrelated session is worse than no context at all, and the
    honest default when nothing looks live is to bias nothing. Two gates:
    recency (you are dictating *into* the session, so its file was written
    moments ago) and, among recent ones, a project-name match with the window
    you are typing into.
    """
    root = os.path.join(HOME, '.claude/projects')
    try:
        entries = list(os.scandir(root))
    except OSError:
        return ''
    cutoff = time.time() - max_age_sec
    files: list[tuple[float, str, str]] = []  # (mtime, project_dir, path)
    for proj in entries:
        if not proj.is_dir():
            continue
        try:
            for f in os.scandir(proj.path):
                if not f.name.endswith('.jsonl'):
                    continue
                mtime = f.stat().st_mtime
                if mtime >= cutoff:
                    files.append((mtime, proj.name, f.path))
        except OSError:
            continue
    if not files:
        return ''
    files.sort(reverse=True)

    # Project dirs flatten '/' AND '_' to '-', so match on the same normalization.
    #
    # Both sides must be filtered by _PROJECT_STOP or the gate silently does
    # nothing: every VSCode title ends "Visual Studio Code" and every project
    # here lives under ~/code, so the word `code` alone matched all 14 projects
    # and the loop just returned files[0] — the same answer as the fallback.
    # With many sessions open at once (background agents append constantly),
    # "globally newest" is whichever agent flushed last, i.e. a coin flip, and
    # the docstring above says a wrong session is worse than no context at all.
    title_words = _match_words(re.split(r'[^A-Za-z0-9]+', window_title.lower()))
    if title_words:
        for _mtime, proj, path in files:
            if title_words & _match_words(proj.lower().split('-')):
                return path
    return files[0][2]


def _session_texts(path: str, user_limit: int, user_max_chars: int,
                   assistant_limit: int, assistant_max_chars: int,
                   reject=None) -> tuple[list[str], list[str]]:
    """(user_texts, assistant_texts) from a session file, newest first.

    Only the tail of the file is read: sessions reach tens of MB and only the
    recent turns describe what is being worked on *now*.

    User turns become both terms and verbatim snippets — they are literally the
    user's speaking vocabulary. Assistant turns feed term extraction only: they
    spell jargon more consistently ("雾凇拼音 / rime-ice") but they are far too
    verbose to quote, and their register is not the user's.

    `reject(text) -> bool` drops user turns that are our own earlier dictation
    echoed back (see ContextBuilder.note_dictation).
    """
    import json
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            if size > 512 * 1024:
                f.seek(size - 512 * 1024)
                f.readline()  # drop the partial line the seek landed inside
            raw = f.read().decode('utf-8', 'replace')
    except OSError:
        return [], []

    users: list[str] = []
    assistants: list[str] = []
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        kind = rec.get('type')
        if kind not in ('user', 'assistant'):
            continue
        content = rec.get('message', {}).get('content')
        blocks = [content] if isinstance(content, str) else [
            b.get('text', '') for b in content or [] if isinstance(b, dict) and b.get('type') == 'text'
        ]
        for text in blocks:
            text = (text or '').strip()
            if not text or _META_RE.search(text) or _HARNESS_RE.match(text):
                continue
            text = _TAG_RE.sub(' ', text).strip()
            if len(text) < 4 or _HARNESS_RE.match(text):
                continue
            if kind == 'assistant':
                if len(assistants) < assistant_limit:
                    assistants.append(text[:assistant_max_chars])
            else:
                # Slash commands and bracketed harness notices ("[Request
                # interrupted by user]") are not things the user said out loud.
                if text.startswith('/') or text.startswith('['):
                    continue
                if reject is not None and reject(text):
                    continue  # our own transcript, echoed back — see note_dictation
                if len(users) < user_limit:
                    users.append(text[:user_max_chars])
        if len(users) >= user_limit and len(assistants) >= assistant_limit:
            break
    return users, assistants


# ---------------- term extraction ----------------

_CJK_TERM_MAX = 8


def _cjk_terms(pool: str, min_count: int) -> Counter:
    """Frequency-ranked CJK n-grams that look like *terms*.

    No segmenter is available (and adding one for this is not worth a dependency),
    so this uses **maximal repeats**: an n-gram is a term only if no one-character
    extension of it (left or right) occurs just as often. A naive
    "keep every repeated n-gram" pass instead shreds "生产代码路径" into the
    overlapping fragments 生产代码 / 产代码路 / 代码路径, and those fragments then
    burn term slots and teach the model nonsense word boundaries.

    Terms must also carry at least one character outside `_CJK_COMMON`; an n-gram
    of nothing but frequent characters is ordinary speech ("的时候", "就是说"),
    not vocabulary worth biasing toward.
    """
    counts: Counter = Counter()
    runs = _CJK_RE.findall(pool)
    for run in runs:
        # +1 so every kept term's one-char extensions are counted too, which is
        # what makes maximality decidable.
        for n in range(2, _CJK_TERM_MAX + 2):
            for i in range(len(run) - n + 1):
                counts[run[i:i + n]] += 1

    # Characters seen immediately left/right of each occurrence, so maximality is
    # a dict lookup instead of a scan over every counted n-gram.
    result = Counter()
    for term, count in counts.items():
        if count < min_count or len(term) > _CJK_TERM_MAX:
            continue
        if all(ch in _CJK_COMMON for ch in term):
            continue
        # Extending on either side must lose occurrences, or `term` is just a
        # fragment of a longer phrase that always contains it.
        if any(counts.get(term + ch, 0) >= count or counts.get(ch + term, 0) >= count
               for ch in _neighbors(runs, term)):
            continue
        result[term] = count
    return result


def _neighbors(runs: list[str], term: str) -> set[str]:
    """Characters that ever sit directly before or after `term` in `runs`."""
    out: set[str] = set()
    for run in runs:
        start = run.find(term)
        while start != -1:
            if start > 0:
                out.add(run[start - 1])
            end = start + len(term)
            if end < len(run):
                out.add(run[end])
            start = run.find(term, start + 1)
    return out


def difficulty(term: str) -> float:
    """How badly does this term need biasing? ~0.2 (easy) .. 1.0 (hard).

    Context length is the cost and lexical anchoring is the benefit, so a term
    only earns its slot if the recognizer would plausibly get it wrong. "config"
    and "backend" come out right unaided — biasing them buys nothing and spends
    prefix. "雾凇" and "fcitx5" are exactly what the hint is for.

    The rarity signal already exists as `_CJK_COMMON` but was only ever used as a
    yes/no filter; difficulty is a better *ranking* than raw frequency, which
    systematically favours the common words that need no help.
    """
    cjk = [c for c in term if _CJK_RE.match(c)]
    if cjk:
        rare = sum(1 for c in cjk if c not in _CJK_COMMON)
        return min(1.0, 0.35 + 0.65 * rare / len(cjk))
    low = term.lower()
    if any(c in term for c in '_-.') or any(c.isdigit() for c in term) or term[1:] != term[1:].lower():
        return 1.0            # identifier-shaped: spelling is arbitrary, easy to miss
    if low in _EASY_LATIN:
        return 0.2            # ordinary tech English the model already emits correctly
    return 0.6


# Tech vocabulary Qwen already transcribes reliably. Biasing these is pure cost:
# they take term slots and prefix length and change nothing.
_EASY_LATIN = frozenset('''
config model server client python code file path data test error debug build run
class function method object string list dict array value key name type import
git commit branch merge push pull repo docker image build deploy install update
api http json yaml token cache queue thread process memory disk network port
input output start stop open close read write send load save print log level
user admin system window screen mouse audio video image video text prompt agent
context session message request response result status enable disable default
'''.split())


_LINE_REF_RE = re.compile(r'^L\d+(-L?\d+)?$')
# Git short SHAs and hash-suffixed branch names. Measured live, these took the
# top slots of a 30-term correction list: dd1ca45, b15a24, a559f49, mlclaw-80,
# mlclaw-d2 — scored high because difficulty() rewards digit/letter mixtures,
# and nobody has ever said one out loud. Requiring a digit keeps real words made
# of hex letters (`face`, `decade`, `beef`).
_HASHISH_RE = re.compile(
    r'(?i)^(?=.*[0-9])(?:[0-9a-f]{6,40}|[A-Za-z]+-[0-9a-f]{1,8})$')

# _LATIN_STOP minus the entries whose easiness is a *writing* property, not a
# speech one. `json` sits in that list and is exactly the word that comes back as
# "Jason"; the list was built to keep ordinary prose out of a term bag, and it
# does that job, but it also silently made the worst code-switching offenders
# unbiasable. Only the correction path uses this narrowed set — the general term
# ranking still wants the full list, because there the words are competing on
# frequency and `data`/`file`/`code` really would crowd it out.
_SPEECH_HARD = frozenset('''
json yaml html code text data file files line lines net org com
'''.split())


def _speakable(term: str) -> bool:
    """Would a person ever say this token out loud?

    A biasing slot is only worth spending on a word that can be *spoken and
    misheard*. difficulty() ranks the opposite way — it scores identifier shape
    (underscores, dots, digits, inner capitals) at 1.0 because such spellings are
    arbitrary and easy to get wrong in writing. Measured, that inverted ranking
    filled every slot with `no_thin_cloud`, `nvidia-cuda-mps-control`, `g.20gb`,
    `sku_505x336x194`, `L16-L19` — and pushed out `loss`, which was sitting in the
    same candidate pool. config.yaml already documents this trap for recent_docs
    ("sfm_260729_s001.rrd 这种名字根本没人会念出来"); this is the same trap reached
    from the ranking side instead of the source side.

    Kept: pronounceable names, including alnum ones people do say — H100, L40S,
    RT-DETR, COCO, IoU. Dropped: paths, filenames, flags, long snake_case.
    """
    if '_' in term or '.' in term or term.count('-') > 1 or len(term) > 12:
        return False
    if _LINE_REF_RE.match(term) or _HASHISH_RE.match(term):
        return False
    return term.lower() not in _PATH_STOP


def correction_terms(assistant_pool: str, own_output_raw: list[str], limit: int,
                     *, cjk_min_count: int = 2, latin_only: bool = True) -> list[str]:
    """Words Claude used that our own transcripts never once produced.

    This is the recognizer's own error list, read off the conversation for free.
    You say 拉斯, Claude answers about `loss` — so `loss` is the correct spelling
    of a word we demonstrably cannot produce, which is precisely what a biasing
    slot is for.

    The general ranking cannot surface these, and not by accident: rank_terms
    scores evidence × difficulty, and a word the recognizer already gets right
    appears in BOTH the user pool and the assistant pool, doubling its evidence
    and taking the slot, while the corrected word appears only on Claude's side at
    weight 2 and gets crowded out. The ranking systematically demotes exactly the
    terms worth spending on.

    `own_output_raw` is the note_dictation() window, used the other way round:
    there it says "never quote this back to me", here it says "whatever is missing
    from this is what the recognizer failed to produce".

    Ranked by count × difficulty. Frequency is normally the wrong signal — it
    promotes the common words that need no help — but that objection is spent once
    the pool is filtered to terms we never produced. Inside that pool, how often
    Claude writes a word is clean evidence of what the conversation is about.

    NOTE the filter order: speakability and latin_only are applied BEFORE the cap.
    Capping first spends slots on terms that are then discarded — measured, that
    left barely a dozen terms at limit=30 and dropped `loss` (rank 15 of 55) and
    `JSON` (rank 22 of 87) off a list both comfortably fit.
    """
    if limit <= 0:
        return []
    said: set = set()
    for text in own_output_raw:
        said |= {t.lower() for t in _LATIN_RE.findall(text)}
        said |= set(_cjk_terms(text, 1))
    counts: Counter = Counter(
        t for t in _LATIN_RE.findall(assistant_pool)
        if len(t) >= 3 and t.lower() not in (_LATIN_STOP - _SPEECH_HARD) and _speakable(t))
    if not latin_only:
        counts.update(_cjk_terms(assistant_pool, cjk_min_count))
    scored = {t: c * difficulty(t) for t, c in counts.items()
              if t.lower() not in said and t not in said}
    return [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:limit]]


def _latin_terms(pool: str, min_len: int) -> Counter:
    counts: Counter = Counter()
    for tok in _LATIN_RE.findall(pool):
        low = tok.lower()
        if low in _LATIN_STOP or low in _PATH_STOP:
            continue
        # Jargon signal: an internal separator, a digit, or inner capitals mark a
        # name; otherwise demand length so ordinary prose words don't crowd in.
        distinctive = (any(c in tok for c in '_-.')
                       or any(c.isdigit() for c in tok)
                       or tok[1:] != tok[1:].lower())
        if len(tok) < 3 and not any(c in tok for c in '_-.'):
            continue  # "VS", "os" — too short to bias anything, pure slot waste
        if not distinctive and len(tok) < min_len:
            continue
        counts[tok] += 1
    # Case-fold duplicates onto the most frequent spelling ("Config" vs "config").
    best: dict[str, tuple[int, str]] = {}
    for tok, c in counts.items():
        key = tok.lower()
        prev = best.get(key)
        total = c + (prev[0] if prev else 0)
        spelling = tok if not prev or c > prev[0] else prev[1]
        best[key] = (total, spelling)
    return Counter({spelling: total for total, spelling in best.values()})


# ---------------- assembly ----------------

class Lexicon:
    """A personal vocabulary that accumulates across sessions and days.

    The live context can only see the last handful of messages, which is why a
    term discussed yesterday in another project ("sidecar") is missed even though
    it is squarely part of the user's working vocabulary. This keeps a persistent
    store so that recurring jargon stays available.

    **Only clean sources may write here.** Nothing the recognizer produced ever
    enters: a misheard word that got written down would be re-suggested forever,
    and would look *more* authoritative each time it recurred. Assistant replies
    are the primary feed precisely because they are already corrected — the user's
    point: 相当于 Claude 自己给修过一次了. The clipboard and window titles qualify
    too; the user's own session messages do not, because most of them are
    dictation output.

    Recurrence across *distinct sessions* is the quality gate. A term seen in one
    conversation is a topic; a term seen in several is vocabulary. That gate is
    what lets the lexicon make the hint shorter rather than longer: it competes
    for the same slots as live terms and generally deserves them more.
    """

    VERSION = 1

    def __init__(self, path: str, *, min_sessions: int = 2, max_terms: int = 4000,
                 max_age_days: float = 120):
        self.path = os.path.expanduser(path) if path else ''
        self.min_sessions = min_sessions
        self.max_terms = max_terms
        self.max_age_days = max_age_days
        self.terms: dict[str, dict] = {}
        self._dirty = False
        self._load()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        import json
        try:
            with open(self.path, encoding='utf-8') as f:
                data = json.load(f)
            if data.get('version') == self.VERSION:
                self.terms = data.get('terms') or {}
        except Exception:
            self.terms = {}   # corrupt store is not worth a crash; start over

    def save(self):
        if not self.path or not self._dirty:
            return
        import json
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'version': self.VERSION, 'terms': self.terms}, f,
                          ensure_ascii=False)
            os.replace(tmp, self.path)   # atomic: a crash never truncates the store
            self._dirty = False
        except OSError:
            pass

    def observe(self, terms: Counter, session_key: str, now: float):
        """Record terms seen in one clean-source pass.

        `session_key` identifies the conversation they came from, so recurrence is
        counted across sessions rather than across repeated builds within one.
        """
        if not self.path:
            return
        day = now / 86400.0
        for term, count in terms.items():
            e = self.terms.get(term)
            if e is None:
                self.terms[term] = {'n': count, 'k': 1, 'last': day, 'sess': session_key}
            else:
                e['n'] += count
                e['last'] = day
                if e.get('sess') != session_key:
                    e['k'] = e.get('k', 1) + 1
                    e['sess'] = session_key
        self._dirty = True
        if len(self.terms) > self.max_terms * 1.5:
            self._prune(day)

    def observe_said(self, text: str, now: float):
        """Record terms our OWN transcripts produced — the other half of the store.

        `observe()` answers "is this word part of the user's world"; this answers
        "can the recognizer already say it". A term needs both to be scored: the
        first alone promotes `skill` and `plugin`, which are recognized fine and
        need no slot. Kept decayed (30-day half-life, same as `_score`) so a word
        the recognizer stopped producing re-arms instead of staying suppressed by
        one lucky hit last month.

        This is the ONLY place recognizer output touches the store, and it only
        ever *demotes* — no spelling from the ASR is ever written as a term, which
        is the rule the class docstring exists to protect.
        """
        if not self.path or not text:
            return
        day = now / 86400.0
        # Latin case-insensitively, because the store keeps Claude's
        # capitalization ("SKU", "MLClaw") and the recognizer picks its own.
        # CJK by substring: _cjk_terms is a *discovery* pass and needs a maximal
        # repeat to fire, so a term said once in one short utterance never
        # registers through it — which silently left every Chinese term in the
        # store looking unsayable (干净, 仓库, 只剩 all sat in the top 30).
        # Asking "did this known term appear" is a different, easier question.
        latin = {t.lower() for t in _LATIN_RE.findall(text)}
        for term, e in self.terms.items():
            if _CJK_RE.search(term):
                if term not in text:
                    continue
            elif term.lower() not in latin:
                continue
            e['s'] = self._said(e, day) + 1.0
            e['sday'] = day
        self._dirty = True

    @staticmethod
    def _said(e: dict, day: float) -> float:
        s = e.get('s', 0.0)
        if not s:
            return 0.0
        return s * 0.5 ** (max(0.0, day - e.get('sday', day)) / 30.0)

    def corrections(self, limit: int, now: float, speakable=None) -> list[str]:
        """Long-horizon version of correction_terms(): words we cannot say.

        correction_terms() asks the same question of a 30-message window and a
        20-utterance window, and that short horizon is why a hard word flaps: the
        first time the recognizer gets `MLClaw` right it lands in the own-output
        window, the term is subtracted, the next utterance goes out unbiased and
        it comes back as 「M R Claw」. Measured in the dictation log for 8/18:
        MLClaw 6 right / 4 wrong, alternating.

        Here the same ratio is taken over months instead: `k` sessions of clean
        evidence against a decayed count of how often we actually produced it. A
        word Claude writes in 62 sessions that our transcripts produce six times
        stays near the top; `skill`, said in most utterances, falls off by itself.

        Unmeasured as of this writing — config.yaml's replay numbers cover the
        *generic* term bag (ranked by 证据×难度), which lost, and this is a
        different list. It ships behind `lexicon_corrections` for that reason.
        """
        if not self.path or limit <= 0:
            return []
        day = now / 86400.0
        scored = []
        for t, e in self.terms.items():
            if e.get('k', 1) < self.min_sessions:
                continue
            if speakable is not None and not speakable(t):
                continue
            if t.lower() in (_LATIN_STOP - _SPEECH_HARD):
                continue
            scored.append((t, self._score(e, day) * difficulty(t)
                              / (1.0 + self._said(e, day))))
        scored.sort(key=lambda kv: -kv[1])
        return [t for t, _ in scored[:limit]]

    def _prune(self, day: float):
        """Drop the stale one-offs, keep anything that recurred."""
        keep = {}
        for term, e in self.terms.items():
            age = day - e.get('last', day)
            if e.get('k', 1) >= self.min_sessions or age <= self.max_age_days:
                keep[term] = e
        if len(keep) > self.max_terms:
            ranked = sorted(keep.items(), key=lambda kv: -self._score(kv[1], day))
            keep = dict(ranked[:self.max_terms])
        self.terms = keep

    def _score(self, e: dict, day: float) -> float:
        # Recency halves the weight every 30 days, so vocabulary that fell out of
        # use fades instead of competing forever with what is current.
        age = max(0.0, day - e.get('last', day))
        return e.get('k', 1) * (1 + e.get('n', 1)) ** 0.5 * 0.5 ** (age / 30.0)

    def top(self, limit: int, now: float) -> Counter:
        """Graduated terms, scored for merging against the live context."""
        if not self.path or limit <= 0:
            return Counter()
        day = now / 86400.0
        scored = [(t, self._score(e, day) * difficulty(t))
                  for t, e in self.terms.items() if e.get('k', 1) >= self.min_sessions]
        scored.sort(key=lambda kv: -kv[1])
        return Counter(dict(scored[:limit]))


class ContextBuilder:
    """Assembles the biasing string off the hotkey path.

    `request()` (dictation start) kicks a worker; `get()` (dictation stop) returns
    the freshest value it can, waiting only briefly. A miss is a non-event — the
    transcription simply runs unbiased — so nothing here is allowed to raise or
    block the dictation path.
    """

    def __init__(self, cfg: dict):
        c = cfg.get('asr_context') or {}
        self.enabled = bool(c.get('enabled', True))
        src = c.get('sources') or {}
        self.use_session = bool(src.get('claude_session', True))
        self.use_clipboard = bool(src.get('clipboard', True))
        self.use_window = bool(src.get('window', True))

        self.max_chars = int(c.get('max_chars', 1200))
        self.max_terms = int(c.get('max_terms', 60))
        self.min_term_weight = int(c.get('min_term_weight', 1))
        self.max_snippets = int(c.get('max_snippets', 6))
        self.snippet_max_chars = int(c.get('snippet_max_chars', 160))
        self.session_messages = int(c.get('session_messages', 12))
        self.session_max_age_min = float(c.get('session_max_age_min', 30))
        self.lexicon_terms = int(c.get('lexicon_terms', 20))
        self.lexicon = Lexicon(
            c.get('lexicon_path', '~/.local/state/whisper_input/lexicon.json'),
            min_sessions=int(c.get('lexicon_min_sessions', 2)),
            max_terms=int(c.get('lexicon_max_terms', 4000)),
            max_age_days=float(c.get('lexicon_max_age_days', 120)))
        # Assistant turns are the one source that is topical AND provably not
        # written by the ASR, so they get a wide window: they are what is left
        # after note_dictation() subtracts our own transcripts, and the terms only
        # surface once enough of the conversation is in view (at 8x600 the recent
        # turns were all "restart VSCode"; at 30x800 "雾凇 / 雾凇拼音 / 雾凇词库"
        # appear — spelled correctly, by something other than the recognizer).
        self.session_assistant_messages = int(c.get('session_assistant_messages', 30))
        self.session_assistant_max_chars = int(c.get('session_assistant_max_chars', 800))
        self.assistant_snippets = bool(c.get('assistant_snippets', True))
        self.max_correction_terms = int(c.get('correction_terms', 30))
        self.lexicon_corrections = int(c.get('lexicon_corrections', 0))
        self.correction_latin_only = bool(c.get('correction_latin_only', True))
        self.clipboard_max_chars = int(c.get('clipboard_max_chars', 1500))
        self.recent_docs = int(c.get('recent_docs', 8))
        self.cjk_min_count = int(c.get('cjk_min_count', 2))
        self.latin_min_len = int(c.get('latin_min_len', 4))
        self.ttl_sec = float(c.get('ttl_sec', 20))
        self.wait_sec = float(c.get('wait_sec', 1.5))
        self.debug = bool(c.get('debug', False))

        self._lock = threading.Lock()
        self._value = ''
        self._built_at = 0.0
        self._ready = threading.Event()
        self._worker: threading.Thread | None = None
        # Recent dictation outputs, normalized — see note_dictation().
        self._own_output: list[str] = []
        # Same window, unnormalized. correction_terms() tokenizes it, and
        # _norm_for_match strips the spaces that separate words ("the loss value"
        # -> "thelossvalue"), so it cannot be tokenized.
        self._own_raw: list[str] = []
        self.own_output_keep = int(c.get('own_output_keep', 20))

    # -- self-contamination guard -------------------------------------------

    def note_dictation(self, text: str):
        """Record what we just pasted, so no source can feed it back to us.

        Dictated text does not stay outside the loop: it lands in the clipboard,
        and then in the Claude Code session as a user message. Both are context
        sources. Without this, every utterance biases the next one toward itself —
        and a word we got *wrong* becomes a high-weight term that makes the same
        mistake more likely next time, and more likely still the time after. The
        loop only ever tightens, because a misrecognition never corrects itself.

        Biasing is only worth anything when it comes from a source the ASR did not
        write. So: remember our own output and subtract it everywhere.
        """
        norm = _norm_for_match(text)
        if not norm:
            return
        self._own_output.append(norm)
        self._own_raw.append(text)
        # Teach the store what the recognizer CAN say. Demote-only: observe_said
        # never creates an entry, so no ASR spelling can enter the vocabulary.
        try:
            self.lexicon.observe_said(text, time.time())
        except Exception:
            pass
        del self._own_output[:-self.own_output_keep]
        del self._own_raw[:-self.own_output_keep]

    def _is_own_output(self, text: str) -> bool:
        norm = _norm_for_match(text)
        return bool(norm) and any(norm == o or (len(norm) > 12 and norm in o)
                                  for o in self._own_output)

    # -- public API ---------------------------------------------------------

    def request(self):
        """Start building in the background (called at dictation start)."""
        if not self.enabled:
            return
        with self._lock:
            if time.time() - self._built_at < self.ttl_sec:
                return  # still fresh; reuse
            if self._worker is not None and self._worker.is_alive():
                return
            self._ready.clear()
            self._worker = threading.Thread(target=self._build_safe, daemon=True)
            self._worker.start()

    def get(self) -> str:
        """Freshest context, waiting at most `wait_sec` for an in-flight build."""
        if not self.enabled:
            return ''
        self._ready.wait(self.wait_sec)
        with self._lock:
            return self._value

    # -- internals ----------------------------------------------------------

    def _build_safe(self):
        try:
            value = self._build()
        except Exception as e:
            value = ''
            if self.debug:
                print(f'[asr_context] build failed: {e!r}')
        with self._lock:
            self._value = value
            self._built_at = time.time()
        self._ready.set()

    # Term slots are scarce (`max_terms`), so sources compete on a weight rather
    # than on raw frequency: without this, a directory full of screenshot names
    # outranks the jargon of the session you are actually dictating into.
    _W_SESSION_USER = 4
    _W_CLIPBOARD = 3
    _W_SESSION_ASSISTANT = 2
    _W_WINDOW = 2
    _W_DOCS = 1

    def collect(self):
        """Read every enabled source.

        Returns (pools, snippets, clean_pools, session_key). `clean_pools` is the
        subset that provably never passed through the recognizer — the only text
        allowed to teach the persistent Lexicon a spelling.

        Split out from `assemble` so the ablation harness can score sources and
        output shapes against the real capture corpus without re-implementing any
        of this.
        """
        pools: list[tuple[int, str]] = []   # (weight, text) feeding term extraction
        clean: list[tuple[int, str]] = []   # subset safe to persist
        snippets: list[str] = []            # verbatim lines, newest first
        session_key = ''

        title = _active_window_title() if self.use_window else ''
        if title:
            pools.append((self._W_WINDOW, title))
            # NOT clean. A window title is provably not ASR output, but the
            # persistent store scores by cross-session recurrence and the title
            # furniture recurs in *every* session: measured on the live store,
            # `Studio` (k=171) and `Visual` (k=171) ranked #2 and #3 of the whole
            # lexicon, above MLClaw. It is the recent_docs trap again — a source
            # that is clean, topical-looking, and never spoken aloud.
        # Recent documents is off by default: the list is dominated by screenshots
        # and build artifacts, and a name like `sfm_260729_s001.rrd` is never said
        # out loud. Measured as the source that crowded real jargon out of the
        # term list, with nothing to show for the slots it took.
        if self.use_window and self.recent_docs:
            docs = _recent_docs(self.recent_docs)
            if docs:
                pools.append((self._W_DOCS, ' '.join(docs)))

        if self.use_session:
            path = _newest_session_file(title, self.session_max_age_min * 60)
            if path:
                session_key = os.path.basename(path)
                users, assistants = _session_texts(
                    path, self.session_messages, self.snippet_max_chars,
                    self.session_assistant_messages, self.session_assistant_max_chars,
                    reject=self._is_own_output)
                if users:
                    # NOT clean: most user turns in a session are dictation output.
                    pools.append((self._W_SESSION_USER, '\n'.join(users)))
                    snippets.extend(users)
                if assistants:
                    pools.append((self._W_SESSION_ASSISTANT, '\n'.join(assistants)))
                    clean.append((self._W_SESSION_ASSISTANT, '\n'.join(assistants)))
                    # Top up the snippet slots the user's own turns left over —
                    # but ONLY as a top-up. `snippets` being non-empty is the
                    # gate, and it is load-bearing, not defensive: measured over
                    # the 120-clip replay, assistant prose *added to* user prose
                    # is neutral (CER +0.0004, 12 better / 14 worse = noise),
                    # while assistant prose ALONE is worse than no context at all
                    # (CER +0.0110, hard words 3/9 vs 4/9 for empty). Claude's
                    # register drags the output distribution, and with nothing of
                    # the user's to anchor against, that drag is all you get.
                    if self.assistant_snippets and snippets:
                        snippets.extend(_assistant_snippets(
                            assistants, self.snippet_max_chars, self._is_own_output))

        if self.use_clipboard:
            clip = _read_clipboard(self._is_own_output, self.clipboard_max_chars)
            if clip:
                pools.append((self._W_CLIPBOARD, clip))
                clean.append((self._W_CLIPBOARD, clip))
                first_line = clip.strip().splitlines()[0].strip()
                # Only quote the clipboard when it reads like something a person
                # said. A copied path or URL is fine as term material but makes a
                # nonsense "最近上下文" line, and the prose section exists to give
                # the model register and sentence shape, which a path has none of.
                if 4 <= len(first_line) <= self.snippet_max_chars and _is_prose(first_line):
                    snippets.insert(0, first_line)

        return pools, snippets, clean, session_key

    def raw_weights(self, pools: list[tuple[int, str]]) -> Counter:
        """Evidence weight per term, before any difficulty adjustment."""
        terms: Counter = Counter()
        for weight, text in pools:
            for tok, c in _latin_terms(text, self.latin_min_len).items():
                terms[tok] += c * weight
        merged = '\n'.join(t for _, t in pools)
        for tok, c in _cjk_terms(merged, self.cjk_min_count).items():
            terms[tok] += c * self._W_SESSION_USER
        return terms

    def rank_terms(self, pools: list[tuple[int, str]], lexicon: Counter | None = None) -> list[str]:
        """Term list, best first, merging the live context with the Lexicon.

        Ranking is `evidence × difficulty`, not evidence alone. Frequency on its
        own promotes exactly the words that need no help — "config", "backend" —
        because those are what a technical conversation is full of. The slot is
        only worth spending where the recognizer would plausibly fail, so
        `difficulty()` decides the order and `min_term_weight` remains the floor
        on how much evidence a term needs to be considered at all.

        Live and persistent scores are each normalised against their own maximum
        and then combined with `max()`, not summed: the two are different kinds of
        evidence, and a term present in both should rank high once, not twice.
        """
        if self.max_terms <= 0:
            return []          # prose-only (the measured default) — skip the work
        live = self.raw_weights(pools)
        scored: dict[str, float] = {}
        if live:
            top = max(live.values())
            for t, w in live.items():
                if w >= self.min_term_weight:
                    scored[t] = (w / top) * difficulty(t)
        if lexicon:
            top = max(lexicon.values())
            for t, s in lexicon.items():
                # Slightly discounted so the conversation in front of you outranks
                # equally-scored history when they compete for the last slot.
                v = (s / top) * 0.8
                if v > scored.get(t, 0.0):
                    scored[t] = v
        return [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:self.max_terms]]

    def assemble(self, terms: list[str], snippets: list[str]) -> str:
        """Render the final hint: term list first, then verbatim recent lines."""
        lines: list[str] = []
        if terms:
            lines.append(' '.join(terms))
        used = sum(len(l) + 1 for l in lines)
        picked: list[str] = []
        for s in snippets[:self.max_snippets]:
            if used + len(s) + 1 > self.max_chars:
                break
            picked.append(s)
            used += len(s) + 1
        if picked:
            lines.append('最近上下文：')
            lines.extend(picked)
        return '\n'.join(lines)[:self.max_chars]

    def _build(self) -> str:
        now = time.time()
        pools, snippets, clean, session_key = self.collect()

        # Teach the lexicon from clean sources only, then read it back merged with
        # the live context. Learning happens before the read so a term appearing
        # for the second time is available immediately. Accumulation continues
        # even with injection off (lexicon_terms: 0), so the store is already
        # populated whenever the term-list shape is worth re-evaluating.
        if clean:
            self.lexicon.observe(self.raw_weights(clean), session_key or 'adhoc', now)
        # Outside the `if`: note_dictation() also dirties the store (the said
        # side), and that half would otherwise only reach disk on a build that
        # happened to have clean sources.
        self.lexicon.save()
        lex = self.lexicon.top(self.lexicon_terms, now)

        if not pools and not lex:
            return ''
        terms = self.rank_terms(pools, lex)

        # Corrections go in FRONT of the ranked terms, and are computed
        # separately rather than folded into rank_terms, because the two answer
        # different questions: rank_terms asks "what is this conversation about",
        # correction_terms asks "what has the recognizer been unable to say". The
        # second list is short, disjoint by construction, and the only one with
        # direct evidence attached, so it takes the slots first.
        asst_pool = '\n'.join(t for w, t in pools if w == self._W_SESSION_ASSISTANT)
        if asst_pool and self.max_correction_terms:
            corr = correction_terms(
                asst_pool, self._own_raw, self.max_correction_terms,
                cjk_min_count=self.cjk_min_count,
                latin_only=self.correction_latin_only)
            if corr:
                seen = set(corr)
                terms = corr + [t for t in terms if t not in seen]

        # Then the long-horizon version of the same list. The in-session one
        # above knows what is being talked about right now; this one knows what
        # has been unsayable for months, which is the part a 30-message window
        # structurally cannot see (the 「sidecar」 miss in the original notes).
        if self.lexicon_corrections:
            seen = set(terms)
            terms += [t for t in self.lexicon.corrections(
                self.lexicon_corrections, now, speakable=_speakable)
                if t not in seen]

        out = self.assemble(terms, snippets)
        if self.debug:
            print(f'[asr_context] {len(terms)} terms ({len(lex)} from lexicon of '
                  f'{len(self.lexicon.terms)}), {len(out)} chars\n{out}\n---')
        return out
