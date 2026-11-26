"""Matematik ifodalarni xavfsiz tahlil qilish.

eval()/exec()/sympify() ishlatilmaydi. Matn avval tokenlarga ajratiladi (faqat ruxsat etilgan belgilar),
keyin ast.parse() bilan daraxtga aylantiriladi (ast.parse kodni bajarmaydi) va faqat ruxsat etilgan
tugunlar o'zimizning Node daraxtiga ko'chiriladi. Hisoblash shu daraxt ustida bajariladi.
"""
import ast
import math
import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import special

MAX_LEN = 600        # bitta ifoda uzunligi
MAX_NODES = 250
MAX_DEPTH = 40
MAX_DEGREE = 1000    # ichma-ich darajalar ko'paytmasi chegarasi
MAX_FACTORIAL = 170  # float chegarasi
MAX_LOG10 = 300      # o'zgarmas qism kattaligi: 10^300 gacha
MAX_MATRIX = 50
MAX_DATA = 100_000


class MathParseError(ValueError):
    """Ifoda noto'g'ri yoki ruxsat etilmagan. Xabar foydalanuvchiga ko'rsatiladi."""


@dataclass(frozen=True)
class Node:
    kind: str              # num | var | const | neg | add | sub | mul | div | pow | call
    value: str = ""        # son matni, nom yoki funksiya nomi
    args: tuple = ()


CONSTANTS = {"pi": math.pi, "e": math.e}

# nom -> (argumentlar soni (min, max), float/complex hisoblovchi)
FUNCTIONS = {
    "sin": ((1, 1), np.sin), "cos": ((1, 1), np.cos), "tan": ((1, 1), np.tan),
    "cot": ((1, 1), lambda x: np.cos(x) / np.sin(x)),
    "sec": ((1, 1), lambda x: 1 / np.cos(x)), "csc": ((1, 1), lambda x: 1 / np.sin(x)),
    "asin": ((1, 1), np.arcsin), "acos": ((1, 1), np.arccos), "atan": ((1, 1), np.arctan),
    "sinh": ((1, 1), np.sinh), "cosh": ((1, 1), np.cosh), "tanh": ((1, 1), np.tanh),
    "asinh": ((1, 1), np.arcsinh), "acosh": ((1, 1), np.arccosh), "atanh": ((1, 1), np.arctanh),
    "exp": ((1, 1), np.exp), "sqrt": ((1, 1), np.sqrt), "cbrt": ((1, 1), np.cbrt),
    "abs": ((1, 1), np.abs),
    "log": ((1, 2), lambda x, b=None: np.log(x) if b is None else np.log(x) / np.log(b)),
    "log10": ((1, 1), np.log10), "log2": ((1, 1), np.log2),
    "floor": ((1, 1), np.floor), "ceil": ((1, 1), np.ceil),
    "factorial": ((1, 1), lambda x: np.where((x >= 0) & (x == np.floor(x)), special.gamma(x + 1), np.nan)),
    "gamma": ((1, 1), special.gamma), "erf": ((1, 1), special.erf),
    "heaviside": ((1, 1), lambda x: np.heaviside(x, 0.5)),
}
ALIASES = {"ln": "log", "lg": "log10", "tg": "tan", "ctg": "cot", "arcsin": "asin", "arccos": "acos",
           "arctan": "atan", "arctg": "atan", "sh": "sinh", "ch": "cosh", "th": "tanh"}
GREEK = ("alpha", "beta", "gamma_", "delta", "theta", "lambda_", "mu", "sigma", "phi", "omega", "tau", "rho")
_VAR_RE = re.compile(r"^(?:[A-Za-z][0-9]{0,2}|alpha|beta|delta|theta|mu|sigma|phi|omega|tau|rho)$")
_UNICODE = {"×": "*", "·": "*", "⋅": "*", "÷": "/", "−": "-", "–": "-", "—": "-", "π": " pi ", "²": "^2",
            "³": "^3", "≤": "<=", "≥": ">=", "√": " sqrt", "∞": " oo "}
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|(?P<name>[A-Za-z][A-Za-z0-9]*)|(?P<op>\*\*|[-+*/^(),]))")


def normalize(text):
    for old, new in _UNICODE.items():
        text = text.replace(old, new)
    return text.strip()


def _tokenize(text):
    tokens, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            bad = text[pos:].strip()[:1]
            raise MathParseError(f"Ruxsat etilmagan belgi: «{bad}». Daraja uchun ^, modul uchun abs(x) yozing.")
        kind = m.lastgroup
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


def _prepare(text):
    """Tokenlardan Python sintaksisidagi satr yasaydi: ^ -> **, 2x -> 2*x, sin/ln taxalluslari."""
    tokens = _tokenize(text)
    out = []
    prev = None  # num | value | func | open | close | op
    for i, (kind, tok) in enumerate(tokens):
        nxt = tokens[i + 1][1] if i + 1 < len(tokens) else None
        if kind == "num":
            if prev in ("num", "value", "close"):
                if prev == "num":
                    raise MathParseError("Ikki son orasida amal yo'q. Ko'paytirish uchun * yozing.")
                out.append("*")
            # 007 kabi yozuvlar Python sintaksisida xato bo'ladi
            tok = re.sub(r"^0+(?=\d)", "", tok)
            out.append(tok)
            prev = "num"
        elif kind == "name":
            name = ALIASES.get(tok, tok)
            if prev in ("num", "value", "close"):
                out.append("*")
            if name in FUNCTIONS and nxt == "(":
                out.append(name)
                prev = "func"
            elif name in FUNCTIONS and name != "gamma":
                raise MathParseError(f"«{tok}» funksiyasidan keyin qavs yozing: {tok}(x).")
            elif name in CONSTANTS or _VAR_RE.match(name):
                out.append("v_" + name)  # Python kalit so'zlari bilan to'qnashmasin
                prev = "value"
            else:
                raise MathParseError(
                    f"Noma'lum nom: «{tok[:30]}». O'zgaruvchi bitta harf (x, y, t, x1) bo'lsin; ko'paytma uchun x*y yozing.")
        else:
            if tok == "(":
                if prev in ("num", "value", "close"):
                    out.append("*")
                prev = "open"
            elif tok == ")":
                prev = "close"
            else:
                prev = "op"
            out.append("**" if tok == "^" else tok)
    return " ".join(out)


_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}


def parse(text):
    """Matnni xavfsiz Node daraxtiga aylantiradi."""
    if not isinstance(text, str):
        raise MathParseError("Ifoda matn ko'rinishida bo'lishi kerak.")
    if len(text) > MAX_LEN:
        raise MathParseError(f"Ifoda juda uzun (chegara {MAX_LEN} belgi).")
    text = normalize(text)
    if not text:
        raise MathParseError("Ifoda bo'sh.")
    source = _prepare(text)
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise MathParseError("Ifoda sintaksisi noto'g'ri: qavslar va amallarni tekshiring.") from None
    count = [0]

    def convert(n, depth):
        count[0] += 1
        if depth > MAX_DEPTH or count[0] > MAX_NODES:
            raise MathParseError("Ifoda juda murakkab (ichma-ichlik yoki hadlar soni chegaradan oshdi).")
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            literal = source[n.col_offset:n.end_col_offset]  # manba ASCII, offsetlar mos
            if not math.isfinite(float(literal)):
                raise MathParseError("Son juda katta.")
            return Node("num", literal)
        if isinstance(n, ast.Name) and n.id.startswith("v_"):
            name = n.id[2:]
            return Node("const" if name in CONSTANTS else "var", name)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            inner = convert(n.operand, depth + 1)
            return inner if isinstance(n.op, ast.UAdd) else Node("neg", args=(inner,))
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return Node(_BINOPS[type(n.op)], args=(convert(n.left, depth + 1), convert(n.right, depth + 1)))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCTIONS and not n.keywords:
            lo, hi = FUNCTIONS[n.func.id][0]
            if not lo <= len(n.args) <= hi:
                raise MathParseError(f"«{n.func.id}» funksiyasi argumentlari soni noto'g'ri.")
            return Node("call", n.func.id, tuple(convert(a, depth + 1) for a in n.args))
        if isinstance(n, ast.Tuple):
            raise MathParseError("Vergul faqat funksiya argumentlarini ajratadi. O'nlik kasr uchun nuqta yozing (1.5).")
        raise MathParseError("Ruxsat etilmagan konstruksiya.")

    node = convert(tree.body, 0)
    _check_limits(node)
    return node


def _check_limits(node):
    """Juda katta sonlar va darajalarni rad etadi (hisoblash osilib qolmasligi uchun).

    Qaytaradi: ifodaning o'zgaruvchi bo'yicha darajasiga yuqori chegara (taxminiy).
    """
    degrees = [_check_limits(a) for a in node.args]
    degree = max(degrees, default=1)
    if node.kind == "pow":
        base, exp = node.args
        if not variables(exp):
            k = _const_value(exp)
            if abs(k) > MAX_DEGREE:
                raise MathParseError(f"Daraja ko'rsatkichi juda katta (chegara {MAX_DEGREE}).")
            degree = degrees[0] * max(abs(k), 1)
            if degree > MAX_DEGREE:
                raise MathParseError(f"Ichma-ich darajalar juda katta (chegara {MAX_DEGREE}).")
    elif node.kind == "mul":
        degree = sum(degrees) if all(variables(a) for a in node.args) else degree
        if degree > MAX_DEGREE:
            raise MathParseError("Ko'phad darajasi juda katta.")
    if node.kind == "call" and node.value in ("factorial", "gamma") and not variables(node.args[0]):
        if abs(_const_value(node.args[0])) > MAX_FACTORIAL:
            raise MathParseError(f"Faktorial/gamma argumenti juda katta (chegara {MAX_FACTORIAL}).")
    if node.kind in ("pow", "mul", "call") and not variables(node):
        v = _const_value(node)
        if math.isinf(v) or (math.isfinite(v) and v != 0 and abs(math.log10(abs(v))) > MAX_LOG10):
            raise MathParseError("Son qiymati juda katta yoki juda kichik (10^±300 chegarasi).")
    return degree


def _const_value(node):
    v = evaluate(node, {})
    return float(v)


def variables(node):
    if node.kind == "var":
        return {node.value}
    out = set()
    for a in node.args:
        out |= variables(a)
    return out


def evaluate(node, env, complex_mode=False):
    """Node daraxtini NumPy bilan hisoblaydi. Aniqlanmagan nuqtalar nan/inf bo'ladi."""
    dtype = complex if complex_mode else float

    def go(n):
        if n.kind == "num":
            return np.asarray(float(n.value), dtype=dtype)
        if n.kind == "const":
            return np.asarray(CONSTANTS[n.value], dtype=dtype)
        if n.kind == "var":
            if n.value not in env:
                raise MathParseError(f"«{n.value}» o'zgaruvchisiga qiymat berilmagan.")
            return np.asarray(env[n.value], dtype=dtype)
        a = [go(x) for x in n.args]
        if n.kind == "neg":
            return -a[0]
        if n.kind == "add":
            return a[0] + a[1]
        if n.kind == "sub":
            return a[0] - a[1]
        if n.kind == "mul":
            return a[0] * a[1]
        if n.kind == "div":
            return a[0] / a[1]
        if n.kind == "pow":
            return np.power(a[0], a[1])
        try:
            return np.asarray(FUNCTIONS[n.value][1](*a))
        except TypeError:
            raise MathParseError(f"«{n.value}» funksiyasi bu turdagi (kompleks) son uchun qo'llanmaydi.") from None

    with np.errstate(all="ignore"):
        return go(node)


def make_function(node, *names):
    """f(x, ...) ko'rinishidagi oddiy Python funksiyasi."""
    extra = variables(node) - set(names)
    if extra:
        raise MathParseError(f"Ifodada kutilmagan o'zgaruvchi bor: {', '.join(sorted(extra))}.")

    def f(*values):
        out = evaluate(node, dict(zip(names, values)))
        shape = np.broadcast(*values).shape if values else ()
        return np.broadcast_to(out, shape) if shape else float(out)

    return f


def exact_value(node):
    """Faqat ratsional amallardan iborat o'zgarmas ifodani Fraction bilan aniq hisoblaydi, aks holda None."""
    if node.kind == "num":
        m = re.search(r"[eE]([+-]?\d+)$", node.value)
        if m and abs(int(m.group(1))) > MAX_LOG10:
            return None
        return Fraction(node.value)
    if node.kind not in ("neg", "add", "sub", "mul", "div", "pow"):
        return None
    a = [exact_value(x) for x in node.args]
    if any(x is None for x in a):
        return None
    if node.kind == "neg":
        return -a[0]
    if node.kind == "add":
        r = a[0] + a[1]
    elif node.kind == "sub":
        r = a[0] - a[1]
    elif node.kind == "mul":
        r = a[0] * a[1]
    elif node.kind == "div":
        if a[1] == 0:
            raise MathParseError("Nolga bo'lish mumkin emas.")
        r = a[0] / a[1]
    else:
        base, k = a
        if k.denominator != 1 or abs(k) > MAX_DEGREE:
            return None
        if base == 0 and k < 0:
            raise MathParseError("Nolga bo'lish mumkin emas (0 ning manfiy darajasi).")
        if base.numerator.bit_length() + base.denominator.bit_length() > 20000 / max(abs(k), 1):
            return None
        r = base ** int(k)
    if r.numerator.bit_length() + r.denominator.bit_length() > 20000:
        return None
    return r


_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2, "neg": 3, "pow": 4}


def to_string(node):
    if node.kind in ("num", "var", "const"):
        return node.value
    if node.kind == "call":
        return f"{node.value}({', '.join(to_string(a) for a in node.args)})"

    def wrap(child, need):
        s = to_string(child)
        return f"({s})" if _PREC.get(child.kind, 9) < need else s

    p = _PREC[node.kind]
    if node.kind == "neg":
        return "-" + wrap(node.args[0], p)
    sym = {"add": " + ", "sub": " - ", "mul": "*", "div": "/", "pow": "^"}[node.kind]
    left = wrap(node.args[0], p + 1 if node.kind == "pow" else p)
    right = wrap(node.args[1], p + 1 if node.kind in ("sub", "div") else p)
    return left + sym + right


def parse_relation(text, default_zero=True):
    """'chap = o'ng' yoki tengsizlikni (chap, belgi, o'ng) ga ajratadi."""
    text = normalize(str(text)).replace("==", "=")
    parts = re.split(r"(<=|>=|<|>|=)", text)
    if len(parts) == 1 and default_zero:
        return parse(parts[0]), "=", Node("num", "0")
    if len(parts) != 3:
        raise MathParseError("Aynan bitta tenglik yoki tengsizlik belgisi bo'lishi kerak (qo'sh tengsizlikni ikkiga ajrating).")
    return parse(parts[0]), parts[1], parse(parts[2])


def split_items(text):
    """Tenglamalar tizimini ';' yoki yangi qator bo'yicha ajratadi."""
    return [p.strip() for p in re.split(r"[;\n]", str(text)) if p.strip()]


def parse_number(text):
    """Bitta o'zgarmas qiymat: 5, -1/2, pi/4, sqrt(2). 'inf' / 'oo' ham qabul qilinadi."""
    t = normalize(str(text)).replace(" ", "").lower()
    if t in ("inf", "+inf", "oo", "+oo", "infinity"):
        return math.inf
    if t in ("-inf", "-oo", "-infinity"):
        return -math.inf
    node = parse(str(text))
    if variables(node):
        raise MathParseError(f"Bu yerda son kerak: «{str(text)[:40]}».")
    v = float(evaluate(node, {}))
    if not math.isfinite(v):
        raise MathParseError(f"Qiymat aniqlanmagan yoki cheksiz: «{str(text)[:40]}».")
    return v


def _split_entries(row):
    """Qatorni vergul yoki probel bo'yicha ajratadi (qavs ichidagilar ajratilmaydi)."""
    sep = "," if "," in row else None
    out, depth, cur = [], 0, ""
    for ch in row:
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and (ch == sep or (sep is None and ch.isspace())):
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out if x.strip()]


def parse_matrix(text):
    """'1 2; 3 4', '1,2;3,4' yoki '[[1,2],[3,4]]' -> Node'lar jadvali."""
    t = normalize(str(text))
    if len(t) > 20000:
        raise MathParseError("Matritsa matni juda uzun.")
    if t.startswith("["):
        body = t[1:-1] if re.match(r"^\[\s*\[", t) and t.endswith("]") else t
        rows = re.findall(r"\[([^\[\]]*)\]", body)
        if not rows or re.sub(r"\[[^\[\]]*\]", "", body).strip(" ,;\n"):
            raise MathParseError("Matritsa formati noto'g'ri. Masalan: [[1,2],[3,4]] yoki 1 2; 3 4.")
    else:
        rows = split_items(t)
    matrix = [[parse(x) for x in _split_entries(r)] for r in rows]
    if not matrix or not matrix[0]:
        raise MathParseError("Matritsa bo'sh.")
    if len(matrix) > MAX_MATRIX or len(matrix[0]) > MAX_MATRIX:
        raise MathParseError(f"Matritsa juda katta (chegara {MAX_MATRIX}×{MAX_MATRIX}).")
    if any(len(r) != len(matrix[0]) for r in matrix):
        raise MathParseError("Matritsa qatorlari uzunligi har xil.")
    if any(variables(x) for r in matrix for x in r):
        raise MathParseError("Matritsa elementlari son bo'lishi kerak.")
    return matrix


def parse_data(text):
    """Statistik ma'lumot: sonlar vergul, probel, ';' yoki yangi qator bilan ajratiladi."""
    if isinstance(text, (list, tuple)):
        items = [str(x) for x in text]
    else:
        items = [x for x in re.split(r"[;,\s]+", normalize(str(text)).strip("[]")) if x]
    if not items:
        raise MathParseError("Ma'lumotlar bo'sh.")
    if len(items) > MAX_DATA:
        raise MathParseError(f"Ma'lumot juda ko'p (chegara {MAX_DATA}).")
    return [parse_number(x) for x in items]
