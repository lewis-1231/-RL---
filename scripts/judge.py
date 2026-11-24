"""共用判分器：\\boxed 提取 + 数值/字符串归一化匹配。

GRPO 奖励函数和 CMath 评测必须用同一份逻辑（口径不一致 = reward hacking 温床）。
"""
import re
from fractions import Fraction


def extract_boxed(text: str) -> str | None:
    """取最后一个 \\boxed{...} 的内容，花括号配平。找不到返回 None。"""
    if text is None:
        return None
    key = "\\boxed{"
    idx = text.rfind(key)
    if idx == -1:
        return None
    i = idx + len(key)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() if depth == 0 else None


_LATEX_MAP = [
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\!|\\,|\\;|\\ "), ""),
    (re.compile(r"\\dfrac|\\tfrac"), "\\frac"),
    (re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}"), r"(\1)/(\2)"),
    (re.compile(r"\\(%|cdot|times)"), ""),
    (re.compile(r"\\text\{[^{}]*\}"), ""),
    (re.compile(r"\$|\\pi|°|\\degree|\\circ"), ""),
    (re.compile(r"[,:；;。]"), ""),
]


_CJK_UNITS = (
    "平方厘米|平方米|立方厘米|立方分米|立方厘米|厘米|毫米|千米|公里|分米|米|"
    "千克|克|吨|公斤|升|毫升|元|角|分|个|只|本|人|天|日|小时|分钟|秒|岁|度|"
    "辆|棵|层|页|筐|箱|支|块|件|次|名|篇|道|颗|朵|张|台|盘|双|杯|瓶|袋|亩|倍"
)


def normalize_answer(ans: str | None) -> str:
    """归一化：剥 \\boxed 壳、去 LaTeX 装饰、去千分位逗号、剥中文单位、处理万/亿。"""
    if ans is None:
        return ""
    s = str(ans)
    s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)
    for pat, rep in _LATEX_MAP:
        s = pat.sub(rep, s)
    s = s.replace(",", "").replace("，", "").strip()
    if s.startswith("+"):
        s = s[1:]
    s = s.strip()
    # 剥中文单位（可能叠多个：如 "3千米500米" 只处理尾部会漏，简单场景够用）
    s = re.sub(rf"({_CJK_UNITS})+$", "", s)
    # 万 / 亿 换算成数值
    m = re.fullmatch(r"([\d.]+)(万|亿)", s)
    if m:
        s = str(float(m.group(1)) * (1e4 if m.group(2) == "万" else 1e8))
    return s.strip()


_ARITH_RE = re.compile(r"[\d\s+\-*/().]+")


def _try_number(s: str):
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    # 归一化后的分数表达式如 "(3)/(4)"：白名单字符内安全求值
    if _ARITH_RE.fullmatch(s) and any(op in s for op in "+-*/"):
        try:
            return float(eval(s, {"__builtins__": {}}, {}))
        except Exception:
            return None
    return None


def answers_equal(pred: str | None, gold: str | None, tol: float = 1e-4) -> bool:
    """先数值比较（容差 tol），失败再比归一化字符串。"""
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if not np_ or not ng:
        return False
    a, b = _try_number(np_), _try_number(ng)
    if a is not None and b is not None:
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return np_ == ng
