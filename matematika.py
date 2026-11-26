"""Mathematics AI: hisoblash amallari, Groq AI bilan yechish va Streamlit paneli.

Qatlamlar:
  math_parser    - xavfsiz ifoda tahlili
  math_symbolic  - aniq natija (SymPy, alohida jarayonda)
  math_verifier  - mustaqil sonli tekshiruv
  math_visualizer- grafiklar
Streamlit faqat panel() ichida import qilinadi, shuning uchun hisoblash qismi alohida test qilinadi.
"""
import json
import math
import os
import re
import statistics
from fractions import Fraction

import numpy as np
from scipy import integrate, linalg, optimize, stats

import math_parser as mp
import math_symbolic as sym
import math_verifier as mv
from math_parser import MathParseError, Node
from math_verifier import FAIL, PASS, UNKNOWN, Check

DISCLAIMER = "Hech qaysi natija 100% aniqlik kafolati emas: muhim hisoblarni qayta tekshiring."


class MathError(ValueError):
    """Amalni bajarib bo'lmadi. Xabar foydalanuvchiga ko'rsatiladi."""


def _num(v, digits=12):
    if isinstance(v, (complex, np.complexfloating)):
        if abs(v.imag) < 1e-12 * max(1.0, abs(v.real)):
            return _num(float(v.real), digits)
        return f"{v.real:.{digits}g} {'+' if v.imag >= 0 else '-'} {abs(v.imag):.{digits}g}i"
    return f"{float(v):.{digits}g}"


def _result(title, checks=(), exact=None, approx=None, notes=(), plot=None, data=None):
    return {"title": title, "exact": exact, "approx": approx, "checks": list(checks), "notes": list(notes),
            "plot": plot, "data": data or {}}


def _variable(node_list, wanted=None):
    names = set()
    for n in node_list:
        names |= mp.variables(n)
    if wanted:
        wanted = str(wanted).strip()
        if names and wanted not in names:
            raise MathError(f"Ifodada «{wanted}» o'zgaruvchisi yo'q.")
        return wanted
    if len(names) > 1:
        raise MathError(f"Qaysi o'zgaruvchi bo'yicha hisoblashni ko'rsating: {', '.join(sorted(names))}.")
    return names.pop() if names else "x"


def _range(lower, upper, default=(-100.0, 100.0)):
    a = default[0] if lower in (None, "") else mp.parse_number(lower)
    b = default[1] if upper in (None, "") else mp.parse_number(upper)
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise MathError("Qidiruv oralig'i noto'g'ri: quyi chegara yuqoridan kichik va chekli bo'lsin.")
    return a, b


def _given(x):
    return x not in (None, "")


# ====================== ARIFMETIKA, SODDALASHTIRISH ======================
def op_evaluate(expression, values=None):
    node = mp.parse(expression)
    env = {}
    for item in re.split(r"[;,]", values or ""):
        if item.strip():
            if "=" not in item:
                raise MathError("Qiymatlarni x=2; y=3 ko'rinishida yozing.")
            k, v = item.split("=", 1)
            env[k.strip()] = mp.parse_number(v)
    missing = mp.variables(node) - set(env)
    if missing:
        raise MathError(f"Qiymat berilmagan o'zgaruvchi: {', '.join(sorted(missing))}. Soddalashtirish amalini tanlang yoki qiymat kiriting.")
    value = float(mp.evaluate(node, env))
    if not math.isfinite(value):
        raise MathError("Ifoda bu qiymatlarda aniqlanmagan (masalan nolga bo'lish yoki manfiy sondan ildiz).")
    exact, checks, notes = None, [], []
    if not env:
        frac = mp.exact_value(node)
        if frac is not None:
            exact = str(frac)
            checks.append(mv.check_agreement("aniq kasr arifmetikasi va float hisobini solishtirish", float(frac), value, 1e-9))
            if checks[-1].status == FAIL:
                notes.append("Float hisobida aniqlik yo'qolgan: aniq (kasr) natijaga ishoning.")
                checks[-1] = Check(UNKNOWN, "float bilan solishtirish", "Natija faqat aniq kasr arifmetikasidan olindi.")
                value = float(frac)
        else:
            r = sym.run("evaluate", expression=expression)
            if r["ok"]:
                exact = r["result"]["text"]
                if "value" in r["result"]:
                    checks.append(mv.check_agreement("SymPy (30 xonali) va NumPy natijalarini solishtirish",
                                                     r["result"]["value"][0], value, 1e-9))
            else:
                notes.append("Aniq (simvolik) natija olinmadi: " + r["error"])
    if not checks:
        checks.append(Check(UNKNOWN, "mustaqil tekshiruv", "Faqat bitta sonli usul qo'llandi."))
    notes.append("Taqribiy qiymat ikki baravar aniqlikdagi float (nisbiy xato odatda ~1e-15, katta ayirmalarda ko'proq).")
    return _result("Ifoda qiymati", checks, exact, _num(value, 15), notes)


def op_simplify(expression):
    node = mp.parse(expression)
    r = sym.run("simplify", expression=expression)
    if not r["ok"]:
        raise MathError("Soddalashtirish uchun SymPy kerak. " + r["error"])
    checks = []
    plain = r["result"].get("plain")
    if plain:
        other = mp.parse(plain)
        names = sorted(mp.variables(node) | mp.variables(other))
        rng = np.random.default_rng(7)
        pairs = []
        for _ in range(60):
            env = {n: rng.uniform(-3, 3) for n in names}
            a, b = float(mp.evaluate(node, env)), float(mp.evaluate(other, env))
            if math.isfinite(a) and math.isfinite(b):
                pairs.append((a, b))
        if len(pairs) >= 5:
            a, b = zip(*pairs)
            checks.append(mv.check_agreement("tasodifiy nuqtalarda asl va soddalashgan ifodani solishtirish", a, b, 1e-7))
    if not checks:
        checks.append(Check(UNKNOWN, "mustaqil tekshiruv", "Natijani sonli solishtirib bo'lmadi."))
    return _result("Soddalashtirish", checks, f"{r['result']['text']}   (ko'paytuvchilar: {r['factored']['text']})",
                   notes=["Aniqlanish sohasi asl ifodanikidan farq qilishi mumkin (masalan qisqartirilgan maxraj)."])


# ====================== TENGLAMA VA TENGSIZLIK ======================
def _poly_coeffs(node, var):
    """Ko'phad bo'lsa koeffitsiyentlar (o'sish tartibida), aks holda None."""
    if var not in mp.variables(node):
        v = float(mp.evaluate(node, {})) if not mp.variables(node) else math.nan
        return np.array([v]) if math.isfinite(v) else None
    if node.kind == "var":
        return np.array([0.0, 1.0])
    parts = [_poly_coeffs(a, var) for a in node.args]
    P = np.polynomial.polynomial
    if node.kind == "neg" and parts[0] is not None:
        return -parts[0]
    if node.kind in ("add", "sub", "mul") and all(p is not None for p in parts):
        if node.kind == "mul":
            return P.polymul(*parts) if len(parts[0]) + len(parts[1]) <= 32 else None
        return P.polyadd(*parts) if node.kind == "add" else P.polysub(*parts)
    if node.kind == "div" and parts[0] is not None and parts[1] is not None and len(parts[1]) == 1 and parts[1][0] != 0:
        return parts[0] / parts[1][0]
    if node.kind == "pow" and parts[0] is not None and not mp.variables(node.args[1]):
        k = float(mp.evaluate(node.args[1], {}))
        if k == int(k) and 0 <= k <= 30 and (len(parts[0]) - 1) * k <= 30:
            return P.polypow(parts[0], int(k))
    return None


def _merge_multiple_roots(c, roots):
    """Karrali ildiz sonli usulda bir-biriga yaqin soxta ildizlarga bo'linadi. Klaster o'rtachasi
    ko'phad VA uning (k−1)-tartibgacha hosilalarini nolga aylantirsa, bitta ildiz deb olinadi."""
    P = np.polynomial.polynomial
    left, out = list(roots), []
    while left:
        first = left.pop(0)
        group = [first] + [r for r in left if abs(r - first) <= 1e-2 * max(1.0, abs(first))]
        center = sum(group) / len(group)
        if abs(center.imag) < 1e-7 * max(1.0, abs(center)):
            center = complex(center.real, 0)
        multiple = len(group) > 1 and all(
            abs(P.polyval(center, P.polyder(c, d))) <= 1e-9 * P.polyval(abs(center), np.abs(P.polyder(c, d)))
            for d in range(len(group)))
        if multiple:
            left = [r for r in left if r not in group]
        else:
            center = first
            if abs(center.imag) < 1e-7 * max(1.0, abs(center)):
                center = complex(center.real, 0)
        out.append(center.real if center.imag == 0 else center)
    return out


def find_roots(node, var, lo, hi, samples=20001):
    """[lo, hi] oralig'ida f(x)=0 ning haqiqiy ildizlarini sonli qidiradi."""
    xs = np.linspace(lo, hi, samples)
    ys = np.array(np.broadcast_to(mp.evaluate(node, {var: xs}), xs.shape), dtype=float)
    f = lambda x: float(mp.evaluate(node, {var: x}))
    candidates = list(xs[ys == 0][:50])
    for i in np.where(np.isfinite(ys[:-1]) & np.isfinite(ys[1:]) & (ys[:-1] * ys[1:] < 0))[0][:500]:
        try:
            candidates.append(optimize.brentq(f, xs[i], xs[i + 1], xtol=1e-14, maxiter=200))
        except (ValueError, RuntimeError):
            pass
    a = np.abs(ys)  # ishorasi o'zgarmaydigan (juft karrali) ildizlar: |f| ning lokal minimumlari
    for i in np.where((a[1:-1] < a[:-2]) & (a[1:-1] <= a[2:]) & (a[1:-1] < 1))[0][:200] + 1:
        r = optimize.minimize_scalar(lambda x: abs(f(x)), bounds=(xs[i - 1], xs[i + 1]), method="bounded",
                                     options={"xatol": 1e-12})
        if r.fun < 1e-8:
            candidates.append(r.x)
    roots = []
    zero = Node("num", "0")
    for c in sorted(candidates):
        if abs(c - round(c)) < 1e-7 and f(round(c)) == 0:
            c = float(round(c))
        if any(abs(c - r) <= 1e-6 * max(1.0, abs(c)) for r in roots):
            continue
        if mv.check_root(node, zero, {var: c}).status == PASS:
            roots.append(float(c))
    return roots


def op_solve_equation(equation, variable=None, lower=None, upper=None):
    lhs, op, rhs = mp.parse_relation(equation)
    if op != "=":
        return op_inequality(equation, variable, lower, upper)
    var = _variable([lhs, rhs], variable)
    if mp.variables(lhs) | mp.variables(rhs) != {var}:
        raise MathError("Bu amal bitta o'zgaruvchili tenglama uchun. Parametrli yoki ko'p noma'lumli holda tizim amalidan foydalaning.")
    diff = Node("sub", args=(lhs, rhs))
    lo, hi = _range(lower, upper)
    coeffs = _poly_coeffs(diff, var)
    notes, roots = [], []
    if coeffs is not None:
        c = np.trim_zeros(coeffs, "b")
        if len(c) <= 1:
            msg = "Tenglama ayniyat: har qanday qiymat yechim." if len(c) == 0 else "Tenglamaning yechimi yo'q."
            return _result("Tenglama", [Check(PASS, "ko'phad koeffitsiyentlari", msg)], msg)
        roots = [complex(r) for r in np.polynomial.polynomial.polyroots(c)]
        roots = _merge_multiple_roots(c, roots)
        roots.sort(key=lambda r: (np.imag(r) != 0, np.real(r), np.imag(r)))
        notes.append(f"Ko'phad ({len(c) - 1}-daraja): barcha ildizlar, jumladan komplekslari, sonli usulda topildi. Karrali ildizlarda aniqlik pasayadi.")
    else:
        roots = find_roots(diff, var, lo, hi)
        notes.append(f"Sonli qidiruv faqat [{lo:g}; {hi:g}] oralig'idagi haqiqiy ildizlarni qamraydi; juda yaqin yoki oraliqdan tashqi ildizlar qolib ketishi mumkin.")
    checks, shown = [], []
    if coeffs is None and len(roots) >= 40:
        checks.append(Check(UNKNOWN, "ildizlar to'liqligi", f"Ildizlar juda ko'p ({len(roots)}+): ro'yxat to'liq bo'lmasligi mumkin, oraliqni toraytiring."))
        notes.insert(0, "Faqat dastlabki 40 ta ildiz ko'rsatildi.")
        roots = roots[:40]
    for r in roots:
        ch = mv.check_root(lhs, rhs, {var: r}, tol=1e-8 if coeffs is None else 1e-6)
        shown.append(f"{var} ≈ {_num(r, 10)}")
        ch.detail = f"{var} = {_num(r, 8)}: {ch.detail}"
        checks.append(ch)
    exact = None
    s = sym.run("solve", equations=equation, variables=[var])
    if s["ok"] and s["solutions"]:
        items = []
        for sol in s["solutions"]:
            v = sol.get(var)
            if v and "value" in v:
                value = complex(*v["value"])
                ch = mv.check_root(lhs, rhs, {var: value.real if abs(value.imag) < 1e-20 else value})
                if ch.status == FAIL:
                    notes.append(f"SymPy taklif qilgan {v['text']} asl tenglamani qanoatlantirmadi va olib tashlandi.")
                    continue
                ch.detail = f"aniq ildiz {v['text']}: {ch.detail}"
                checks.append(ch)
                items.append(f"{var} = {v['text']}")
        exact = ";  ".join(items) or None
        if coeffs is None and exact:
            notes.append("SymPy aniq ildizlari barcha (masalan davriy) yechimlarni qamramasligi mumkin.")
    elif not s["ok"]:
        notes.append("Aniq ildiz olinmadi: " + s["error"])
    if not shown and not exact:
        checks.append(Check(UNKNOWN, "ildiz qidirish", "Ko'rsatilgan oraliqda ildiz topilmadi; bu yechim yo'qligini isbotlamaydi."))
    plot = {"kind": "function", "expressions": [mp.to_string(diff)], "var": var, "lo": -10.0, "hi": 10.0,
            "points": [(float(np.real(r)), 0.0) for r in roots if np.imag(r) == 0]}
    real = [float(np.real(r)) for r in roots if np.imag(r) == 0]
    if real and (min(real) < -9 or max(real) > 9):
        pad = 0.15 * max(max(real) - min(real), 5)
        plot.update(lo=min(real) - pad, hi=max(real) + pad)
    return _result("Tenglama", checks, exact, ";  ".join(shown) or "ildiz topilmadi", notes, plot,
                   {"roots": [_num(r) for r in roots]})


def op_inequality(inequality, variable=None, lower=None, upper=None):
    lhs, op, rhs = mp.parse_relation(inequality, default_zero=False)
    if op == "=":
        raise MathError("Bu tenglama: tenglama amalini tanlang.")
    var = _variable([lhs, rhs], variable)
    if mp.variables(lhs) | mp.variables(rhs) != {var}:
        raise MathError("Tengsizlik bitta o'zgaruvchili bo'lishi kerak.")
    lo, hi = _range(lower, upper)
    diff = Node("sub", args=(lhs, rhs))
    f = lambda x: float(mp.evaluate(diff, {var: x}))
    xs = np.linspace(lo, hi, 20001)
    ys = np.array(np.broadcast_to(mp.evaluate(diff, {var: xs}), xs.shape), dtype=float)
    cuts = find_roots(diff, var, lo, hi)
    ok = np.isfinite(ys)
    for i in np.where(ok[:-1] != ok[1:])[0][:100]:  # aniqlanish sohasi chegaralari
        a, b = xs[i], xs[i + 1]
        for _ in range(60):
            m = (a + b) / 2
            a, b = (m, b) if math.isfinite(f(m)) == ok[i] else (a, m)
        cuts.append((a + b) / 2)
    for i in np.where(ok[:-1] & ok[1:] & (ys[:-1] * ys[1:] < 0))[0][:100]:  # uzilishlar (1/x kabi)
        a, b = xs[i], xs[i + 1]
        for _ in range(60):
            m = (a + b) / 2
            a, b = (m, b) if (f(m) < 0) == (ys[i] < 0) else (a, m)
        cuts.append((a + b) / 2)
    edges = [lo]
    for c in sorted(cuts):
        c = float(round(c)) if abs(c - round(c)) < 1e-9 else float(c)
        if c - edges[-1] > 1e-7 and hi - c > 1e-7:
            edges.append(c)
    edges.append(hi)
    cmp = {"<": lambda v: v < 0, "<=": lambda v: v <= 0, ">": lambda v: v > 0, ">=": lambda v: v >= 0}[op]
    holds = lambda x: math.isfinite(f(x)) and cmp(f(x))

    def point_holds(c):
        v = f(c)
        if not math.isfinite(v) or abs(v) > 1e9:  # qutb (masalan tan(x) da pi/2) yechimga kirmaydi
            return False
        is_root = mv.check_root(lhs, rhs, {var: c}).status == PASS
        return op in ("<=", ">=") if is_root else cmp(v)

    intervals = []
    for a, b in zip(edges, edges[1:]):
        if holds((a + b) / 2):
            left, right = point_holds(a), point_holds(b)
            if intervals and intervals[-1][1] == a and left:
                intervals[-1] = [intervals[-1][0], b, intervals[-1][2], right]
            else:
                intervals.append([a, b, left, right])
    singles = [c for c in edges[1:-1] if point_holds(c) and not any(i[0] <= c <= i[1] for i in intervals)]
    parts = [f"{'[' if a_in else '('}{_num(a, 8)}; {_num(b, 8)}{']' if b_in else ')'}" for a, b, a_in, b_in in intervals]
    parts += [f"{{{_num(c, 8)}}}" for c in singles]
    approx = " ∪ ".join(parts) if parts else "bu oraliqda yechim topilmadi (∅)"
    checks = [mv.check_inequality(lhs, op, rhs, var, intervals, lo, hi)]
    notes = [f"Natija faqat [{lo:g}; {hi:g}] qidiruv oralig'i uchun: chegaradagi {lo:g} yoki {hi:g} oraliq cheksiz davom etishi mumkinligini bildiradi.",
             "Interval uchlari sonli usulda topilgan."]
    exact = None
    s = sym.run("inequality", inequality=inequality, variable=var)
    if s["ok"]:
        exact = s["result"]["text"]
        notes.append("Aniq to'plam (SymPy) avtomatik tekshirilmaydi: tekshiruv faqat taqribiy intervallarga tegishli.")
    else:
        notes.append("Aniq to'plam olinmadi: " + s["error"])
    return _result("Tengsizlik", checks, exact, approx, notes,
                   {"kind": "function", "expressions": [mp.to_string(diff)], "var": var, "lo": -10.0, "hi": 10.0},
                   {"intervals": intervals, "points": singles})


def op_system(equations, variables=None):
    items = mp.split_items(equations)
    rels = [mp.parse_relation(t) for t in items]
    if any(op != "=" for _, op, _ in rels):
        raise MathError("Tizimda faqat tengliklar bo'lishi kerak.")
    names = sorted(set().union(*[mp.variables(a) | mp.variables(b) for a, _, b in rels]))
    if variables:
        wanted = [v.strip() for v in re.split(r"[,;\s]+", variables) if v.strip()]
        if sorted(wanted) != names:
            raise MathError(f"Noma'lumlar tizimdagi o'zgaruvchilar bilan mos emas: {', '.join(names)}.")
    if not names or len(names) > 10 or len(rels) > 10:
        raise MathError("Tizimda 1–10 ta noma'lum va tenglama bo'lishi kerak.")
    if len(rels) != len(names):
        raise MathError("Sonli yechim uchun tenglamalar soni noma'lumlar soniga teng bo'lishi kerak.")
    F = lambda v: np.array([float(mp.evaluate(a, dict(zip(names, v))) - mp.evaluate(b, dict(zip(names, v)))) for a, _, b in rels])
    n = len(names)
    f0 = F(np.zeros(n))
    A = np.column_stack([F(np.eye(n)[i]) - f0 for i in range(n)])
    rng = np.random.default_rng(1)
    probes = rng.uniform(-5, 5, (6, n))
    linear = np.all(np.isfinite(A)) and np.all(np.isfinite(f0)) and all(
        np.allclose(F(p), A @ p + f0, rtol=1e-9, atol=1e-8) for p in probes)
    notes, solutions = [], []
    if linear:
        rank = np.linalg.matrix_rank(A)
        if rank < n:
            consistent = np.linalg.matrix_rank(np.column_stack([A, -f0])) == rank
            msg = "cheksiz ko'p yechim bor" if consistent else "yechim yo'q"
            return _result("Chiziqli tizim", [Check(PASS, "rang sharti (Kroneker–Kapelli)", f"rang(A) = {rank} < {n}: {msg}.")],
                           approx=msg, notes=["Rang sonli usulda aniqlangan."])
        solutions = [np.linalg.solve(A, -f0)]
        notes.append(f"Chiziqli tizim. Shartlanganlik soni: {np.linalg.cond(A):.3g}.")
    else:
        starts = [np.ones(n), -np.ones(n), np.zeros(n) + 0.5] + list(rng.uniform(-10, 10, (60, n)))
        for x0 in starts:
            r = optimize.root(F, x0, tol=1e-13)
            if np.all(np.isfinite(r.x)) and np.all(np.isfinite(F(r.x))) and np.max(np.abs(F(r.x))) < 1e-8:
                if not any(np.allclose(r.x, s, atol=1e-6, rtol=1e-6) for s in solutions):
                    solutions.append(r.x)
        notes.append("Nochiziqli tizim: ko'p boshlang'ich nuqtali sonli qidiruv. Barcha yechimlar topilganiga kafolat yo'q.")
    checks, shown = [], []
    for s in solutions[:20]:
        env = dict(zip(names, map(float, s)))
        sub = [mv.check_root(a, b, env, tol=1e-7) for a, _, b in rels]
        status = mv.overall(sub)
        label = ", ".join(f"{k} ≈ {_num(0.0 if abs(v) < 1e-12 else v, 10)}" for k, v in env.items())
        shown.append(label)
        checks.append(Check(status, "yechimni barcha tenglamalarga qo'yish", label, max(c.error or 0 for c in sub), 1e-7))
    if not checks:
        checks.append(Check(UNKNOWN, "ildiz qidirish", "Yechim topilmadi; bu yechim yo'qligini isbotlamaydi."))
    exact = None
    r = sym.run("solve", equations=equations, variables=names)
    if r["ok"] and r["solutions"]:
        exact = " | ".join(", ".join(f"{k} = {v['text']}" for k, v in s.items()) for s in r["solutions"])
        for s_ in r["solutions"][:10]:
            if set(s_) == set(names) and all("value" in v for v in s_.values()):
                env = {k: complex(*v["value"]) for k, v in s_.items()}
                sub = [mv.check_root(a, b, env, tol=1e-7) for a, _, b in rels]
                checks.append(Check(mv.overall(sub), "aniq (SymPy) yechimni tenglamalarga qo'yish", ""))
            else:
                checks.append(Check(UNKNOWN, "aniq (SymPy) yechim", "Parametrli yechim avtomatik tekshirilmadi."))
    elif not r["ok"]:
        notes.append("Aniq yechim olinmadi: " + r["error"])
    return _result("Tenglamalar tizimi", checks, exact, " | ".join(shown) or "topilmadi", notes,
                   data={"solutions": [list(map(float, s)) for s in solutions[:20]], "variables": names})


# ====================== MATEMATIK ANALIZ ======================
def op_derivative(expression, variable=None, order=1, at=None):
    node = mp.parse(expression)
    var = _variable([node], variable)
    order = int(order or 1)
    if not 1 <= order <= 10:
        raise MathError("Hosila tartibi 1 dan 10 gacha bo'lsin.")
    notes, checks, exact, approx = [], [], None, None
    r = sym.run("derivative", expression=expression, variable=var, order=order)
    if r["ok"]:
        exact = r["result"]["text"]
        plain = r["result"].get("plain")
        checks.append(mv.check_derivative(node, mp.parse(plain), var, order) if plain else
                      Check(UNKNOWN, "hosilani tekshirish", "Natija sonli tekshiruvchi formatiga o'tmadi."))
    else:
        notes.append("Aniq hosila olinmadi: " + r["error"])
    if _given(at):
        point = mp.parse_number(at)
        if mp.variables(node) - {var}:
            raise MathError("Nuqtadagi sonli hosila uchun ifodada bitta o'zgaruvchi bo'lsin.")
        f = lambda x: float(mp.evaluate(node, {var: x}))
        num = mv.numeric_derivative(f, point, order)
        if num is None:
            raise MathError("Bu nuqtada sonli hosilani hisoblab bo'lmadi (tartib 4 dan katta yoki aniqlanish sohasi chegarasi).")
        approx = f"f^({order})({_num(point)}) ≈ {_num(num[0], 8)}  (xato bahosi {num[1]:.1e})"
        h = 1e-6 * max(1.0, abs(point))
        left, right = (f(point) - f(point - h)) / h, (f(point + h) - f(point)) / h
        smooth = mv.check_agreement("chap va o'ng hosilalarni solishtirish", left, right, 1e-3)
        if smooth.status != PASS:
            raise MathError(f"{var} = {_num(point)} nuqtada chap va o'ng hosilalar mos kelmadi ({_num(left, 5)} va {_num(right, 5)}): "
                            "hosila mavjud emas yoki sonli hisoblab bo'lmaydi.")
        if order == 1:
            checks.append(smooth)
    elif not r["ok"]:
        raise MathError("SymPy'siz faqat nuqtadagi sonli hosila hisoblanadi: «nuqta» maydonini to'ldiring. " + r["error"])
    plot = None
    if mp.variables(node) == {var}:
        exprs = [expression] + ([r["result"]["plain"]] if r["ok"] and r["result"].get("plain") else [])
        plot = {"kind": "function", "expressions": exprs, "var": var, "lo": -5.0, "hi": 5.0}
    return _result(f"Hosila ({order}-tartib)", checks or [Check(UNKNOWN, "tekshiruv", "Mustaqil tekshiruv bajarilmadi.")],
                   exact, approx, notes, plot)


def op_integral(expression, variable=None, lower=None, upper=None):
    node = mp.parse(expression)
    var = _variable([node], variable)
    if _given(lower) != _given(upper):
        raise MathError("Aniq integral uchun ikkala chegarani kiriting.")
    definite = _given(lower)
    notes, checks, exact, approx = [], [], None, None
    r = sym.run("integral", expression=expression, variable=var,
                lower=str(lower) if definite else None, upper=str(upper) if definite else None)
    plot = None
    if not definite:
        if not r["ok"]:
            raise MathError("Aniqmas integral uchun SymPy kerak. " + r["error"])
        exact = r["result"]["text"] + " + C"
        plain = r["result"].get("plain")
        checks.append(mv.check_antiderivative(node, mp.parse(plain), var) if plain else
                      Check(UNKNOWN, "differensiallab tekshirish", "Natija maxsus funksiyali: sonli tekshirilmadi."))
        return _result("Aniqmas integral", checks, exact, notes=notes)
    if mp.variables(node) - {var}:
        raise MathError("Aniq integralda bitta o'zgaruvchi bo'lsin.")
    a, b = mp.parse_number(lower), mp.parse_number(upper)
    f = lambda x: float(mp.evaluate(node, {var: x}))
    value, err, problems = mv.quadrature(f, a, b)
    reliable = not problems and math.isfinite(value) and err <= 1e-6 * max(1.0, abs(value))
    if reliable:
        approx = f"{_num(value)}  (kvadratura xato bahosi {err:.1e})"
        notes.append("Juda tor cho'qqili funksiyalarda sonli kvadratura cho'qqini sezmay qolishi mumkin.")
    else:
        notes.append("Sonli kvadratura yaqinlashmadi: integral uzoqlashuvchi yoki maxsus nuqtali bo'lishi mumkin.")
    if r["ok"]:
        exact = r["result"]["text"]
        if "value" in r["result"] and abs(r["result"]["value"][1]) < 1e-12:
            checks.append(mv.check_definite_integral(node, var, a, b, r["result"]["value"][0]))
        else:
            checks.append(Check(UNKNOWN, "kvadratura bilan solishtirish", "Aniq natija chekli haqiqiy son emas."))
    else:
        notes.append("Aniq natija olinmadi: " + r["error"])
        if reliable:  # ikkinchi mustaqil sonli usul
            if math.isfinite(a) and math.isfinite(b):
                nodes, weights = np.polynomial.legendre.leggauss(40)
                edges = np.linspace(a, b, 201)
                mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
                X = mid[:, None] + half[:, None] * nodes[None, :]
                other = float(np.sum(half[:, None] * weights[None, :] * np.broadcast_to(mp.evaluate(node, {var: X}), X.shape)))
                checks.append(mv.check_agreement("Gauss–Legendre kvadraturasi bilan solishtirish", value, other, 1e-7))
            else:
                checks.append(Check(UNKNOWN, "ikkinchi usul", "Cheksiz chegarada faqat bitta sonli usul qo'llandi."))
    if not checks:
        checks.append(Check(UNKNOWN, "tekshiruv", "Natija ishonchli tasdiqlanmadi."))
    if math.isfinite(a) and math.isfinite(b) and a != b:
        pad = 0.2 * abs(b - a)
        plot = {"kind": "function", "expressions": [expression], "var": var, "lo": min(a, b) - pad, "hi": max(a, b) + pad}
    return _result("Aniq integral", checks, exact, approx, notes, plot, {"value": value if reliable else None})


def op_limit(expression, point, variable=None, direction="both"):
    node = mp.parse(expression)
    var = _variable([node], variable)
    if direction not in ("both", "left", "right"):
        raise MathError("Yo'nalish both, left yoki right bo'lsin.")
    if mp.variables(node) - {var}:
        raise MathError("Limitda bitta o'zgaruvchi bo'lsin.")
    p = mp.parse_number(point)
    r = sym.run("limit", expression=expression, variable=var, point=str(point), direction=direction)
    notes, exact, approx = [], None, None
    if r["ok"]:
        exact = r["result"]["text"]
        if "value" in r["result"]:
            value = r["result"]["value"][0]
        else:
            value = math.inf if exact == "oo" else -math.inf if exact == "-oo" else None
        checks = [mv.check_limit(node, var, p, value, direction) if value is not None else
                  Check(UNKNOWN, "sonli yaqinlashish", "Natija son emas.")]
    else:
        notes.append("Aniq limit olinmadi: " + r["error"])
        sides = ["right" if p < 0 else "left"] if math.isinf(p) else (["left", "right"] if direction == "both" else [direction])
        estimates = []
        for side in sides:
            xs = ([math.copysign(10.0 ** k, p) for k in range(2, 8)] if math.isinf(p) else
                  [p + (-1 if side == "left" else 1) * 10.0 ** -k * max(1.0, abs(p)) for k in range(3, 8)])
            ys = [float(mp.evaluate(node, {var: x})) for x in xs]
            ys = [y for y in ys if math.isfinite(y)]
            if len(ys) < 3:
                raise MathError("Nuqta atrofida funksiya aniqlanmagan.")
            diffs = [abs(b - a) for a, b in zip(ys, ys[1:])]
            best = int(np.argmin(diffs))
            if diffs[best] > 1e-4 * max(1.0, abs(ys[best + 1])):
                raise MathError("Sonli qiymatlar yaqinlashmayapti: limit cheksiz, mavjud emas yoki juda sekin.")
            estimates.append(ys[best + 1])
        if len(estimates) == 2 and abs(estimates[0] - estimates[1]) > 1e-3 * max(1.0, abs(estimates[0])):
            raise MathError(f"Bir tomonlama limitlar har xil ({_num(estimates[0], 6)} va {_num(estimates[1], 6)}): ikki tomonlama limit mavjud emas.")
        approx = _num(estimates[-1], 6)
        checks = [Check(UNKNOWN, "sonli baho", "Bu faqat sonli taxmin, isbot yoki mustaqil tasdiq emas.")]
    return _result("Limit", checks, exact, approx, notes)


def op_gradient(expression, point):
    node = mp.parse(expression)
    names = sorted(mp.variables(node))
    if not 1 <= len(names) <= 8:
        raise MathError("Funksiyada 1–8 ta o'zgaruvchi bo'lsin.")
    env = {}
    for item in re.split(r"[;,]", point or ""):
        if item.strip():
            k, _, v = item.partition("=")
            env[k.strip()] = mp.parse_number(v)
    if set(env) != set(names):
        raise MathError(f"Nuqtani barcha o'zgaruvchilar uchun bering: {', '.join(n + '=...' for n in names)}.")
    f = lambda e: float(mp.evaluate(node, e))
    grad, checks, notes, exact = [], [], [], []
    for n in names:
        num = mv.numeric_derivative(lambda x: f(dict(env, **{n: x})), env[n], 1)
        if num is None:
            raise MathError(f"{n} bo'yicha xususiy hosila bu nuqtada aniqlanmadi.")
        grad.append(num[0])
        r = sym.run("derivative", expression=expression, variable=n, order=1)
        if r["ok"] and r["result"].get("plain"):
            exact.append(f"∂f/∂{n} = {r['result']['text']}")
            claimed = float(mp.evaluate(mp.parse(r["result"]["plain"]), env))
            checks.append(mv.check_agreement(f"∂f/∂{n}: SymPy va chekli ayirma", claimed, num[0], 1e-6))
        else:
            checks.append(Check(UNKNOWN, f"∂f/∂{n}", f"Faqat sonli usul (xato bahosi {num[1]:.1e})."))
    approx = f"f = {_num(f(env), 10)};  ∇f = (" + ", ".join(f"∂{n}: {_num(g, 8)}" for n, g in zip(names, grad)) + ")"
    return _result("Gradiyent", checks, ";  ".join(exact) or None, approx, notes, data={"gradient": grad})


# ====================== CHIZIQLI ALGEBRA ======================
def _gauss(rows, ncols):
    """Fraction bilan aniq Gauss–Jordan. (RREF, rang, determinant)."""
    M = [list(r) for r in rows]
    rank, det = 0, Fraction(1)
    for c in range(ncols):
        pivot = next((i for i in range(rank, len(M)) if M[i][c] != 0), None)
        if pivot is None:
            det = Fraction(0)
            continue
        if pivot != rank:
            M[rank], M[pivot] = M[pivot], M[rank]
            det = -det
        det *= M[rank][c]
        inv = 1 / M[rank][c]
        M[rank] = [x * inv for x in M[rank]]
        for i in range(len(M)):
            if i != rank and M[i][c] != 0:
                factor = M[i][c]
                M[i] = [a - factor * b for a, b in zip(M[i], M[rank])]
        rank += 1
    return M, rank, det


def _fmt_matrix(M):
    return "[" + "; ".join(" ".join(str(x) if isinstance(x, Fraction) else _num(x, 8) for x in r) for r in M) + "]"


def op_matrix(matrix, operation="det", other=None):
    nodes = mp.parse_matrix(matrix)
    A = np.array([[float(mp.evaluate(x, {})) for x in r] for r in nodes])
    if not np.all(np.isfinite(A)):
        raise MathError("Matritsada aniqlanmagan element bor.")
    m, n = A.shape
    exact_rows = None
    if m <= 12 and n <= 12:
        fr = [[mp.exact_value(x) for x in r] for r in nodes]
        if all(x is not None for r in fr for x in r):
            exact_rows = fr
    notes, checks, exact, approx = [], [], None, None
    operation = str(operation).strip().lower()
    if operation in ("det", "inverse", "eigen") and m != n:
        raise MathError("Bu amal uchun kvadrat matritsa kerak.")
    if operation == "det":
        d = float(np.linalg.det(A))
        approx = _num(d)
        checks.append(mv.check_determinant(A, d))
        if exact_rows:
            ed = _gauss(exact_rows, n)[2]
            exact = str(ed)
            checks = [mv.check_agreement("aniq (kasrli) Gauss usuli va LU sonli usulini solishtirish", float(ed), d, 1e-6)]
            if checks[0].status == FAIL:
                notes.append("Sonli (float) natija yomon shartlanganlik sabab noaniq; aniq natijadan foydalaning.")
                checks = [mv.check_agreement("determinantni transpozitsiya orqali aniq qayta hisoblash",
                                             float(_gauss(list(zip(*exact_rows)), n)[2]), float(ed), 0.0)]
    elif operation == "inverse":
        if exact_rows:
            aug = [list(r) + [Fraction(int(i == j)) for j in range(n)] for i, r in enumerate(exact_rows)]
            R = _gauss(aug, n)[0]
            if _gauss(exact_rows, n)[1] < n:
                raise MathError("Matritsa xos (determinant = 0): teskari matritsa mavjud emas.")
            inv = [r[n:] for r in R]
            product = [[sum(exact_rows[i][k] * inv[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            ok = all(product[i][j] == int(i == j) for i in range(n) for j in range(n))
            exact = _fmt_matrix(inv)
            approx = _fmt_matrix([[float(x) for x in r] for r in inv])
            checks.append(Check(PASS if ok else FAIL, "A·A⁻¹ = I (aniq kasr arifmetikasida)", "", 0.0 if ok else 1.0, 0.0))
        else:
            if np.linalg.matrix_rank(A) < n:
                raise MathError("Matritsa sonli jihatdan xos: teskari matritsa topilmadi.")
            inv = np.linalg.inv(A)
            approx = _fmt_matrix(inv)
            checks.append(mv.check_inverse(A, inv))
        notes.append(f"Shartlanganlik soni: {np.linalg.cond(A):.3g}.")
    elif operation == "rank":
        sv = np.linalg.svd(A, compute_uv=False)
        r = int(np.linalg.matrix_rank(A))
        approx = str(r)
        if exact_rows:
            er = _gauss(exact_rows, n)[1]
            exact = str(er)
            tr = _gauss(list(zip(*exact_rows)), m)[1]
            checks.append(Check(PASS if er == tr else FAIL, "rang(A) = rang(Aᵀ), aniq Gauss usuli", f"SVD bo'yicha sonli rang: {r}."))
        else:
            checks.append(mv.check_agreement("rang(A) = rang(Aᵀ) (SVD)", r, int(np.linalg.matrix_rank(A.T)), 0.0))
        notes.append(f"Eng kichik singulyar qiymat: {sv.min():.3g}.")
    elif operation == "eigen":
        values, vectors = np.linalg.eig(A)
        approx = ";  ".join(f"λ{i + 1} = {_num(v, 10)}, v{i + 1} = ({', '.join(_num(x, 6) for x in vectors[:, i])})"
                            for i, v in enumerate(values))
        checks += mv.check_eigen(A, values, vectors)
        if np.linalg.cond(vectors) > 1e8:
            notes.append("Xos vektorlar deyarli chiziqli bog'liq: matritsa diagonallashmasligi mumkin, aniqlik past.")
    elif operation == "solve":
        if not other:
            raise MathError("A·x = b uchun b vektorini kiriting.")
        bn = mp.parse_matrix(other)
        if len(bn) == 1 and len(bn[0]) == m:
            bn = [[x] for x in bn[0]]
        if len(bn) != m or len(bn[0]) != 1:
            raise MathError("b vektori uzunligi A qatorlari soniga teng bo'lsin.")
        b = np.array([float(mp.evaluate(r[0], {})) for r in bn])
        eb = [mp.exact_value(r[0]) for r in bn]
        if exact_rows and all(x is not None for x in eb):
            R, _, _ = _gauss([list(r) + [v] for r, v in zip(exact_rows, eb)], n)
            rank = _gauss(exact_rows, n)[1]
            if any(all(x == 0 for x in r[:n]) and r[n] != 0 for r in R):
                return _result("A·x = b", [Check(PASS, "Kroneker–Kapelli sharti (aniq)", "rang(A) < rang(A|b)")], "yechim yo'q")
            if rank < n:
                return _result("A·x = b", [Check(PASS, "Kroneker–Kapelli sharti (aniq)", f"rang = {rank} < {n}")],
                               f"cheksiz ko'p yechim ({n - rank} ta erkin o'zgaruvchi). Kengaytirilgan matritsaning RREF ko'rinishi: {_fmt_matrix(R)}")
            x = [R[i][n] for i in range(n)]
            ok = all(sum(a * v for a, v in zip(row, x)) == bv for row, bv in zip(exact_rows, eb))
            exact = ", ".join(f"x{i + 1} = {v}" for i, v in enumerate(x))
            approx = ", ".join(f"x{i + 1} ≈ {_num(float(v), 10)}" for i, v in enumerate(x))
            checks.append(Check(PASS if ok else FAIL, "A·x = b (aniq kasr arifmetikasida)", "", 0.0 if ok else 1.0, 0.0))
        else:
            if m != n or np.linalg.matrix_rank(A) < n:
                raise MathError("Bu hajm/ko'rinishdagi tizim uchun faqat yagona yechimli kvadrat tizim qo'llanadi.")
            x = linalg.solve(A, b)
            approx = ", ".join(f"x{i + 1} ≈ {_num(v, 10)}" for i, v in enumerate(x))
            checks.append(mv.check_linear_solution(A, x, b))
            notes.append(f"Shartlanganlik soni: {np.linalg.cond(A):.3g}.")
    elif operation == "multiply":
        if not other:
            raise MathError("Ikkinchi matritsani kiriting.")
        B = np.array([[float(mp.evaluate(x, {})) for x in r] for r in mp.parse_matrix(other)])
        if n != B.shape[0]:
            raise MathError("O'lchamlar mos emas: A ustunlari soni B qatorlari soniga teng bo'lsin.")
        C = A @ B
        approx = _fmt_matrix(C)
        v = np.random.default_rng(3).standard_normal(B.shape[1])
        checks.append(mv.check_agreement("Freivalds tekshiruvi: C·v = A·(B·v)", C @ v, A @ (B @ v), 1e-8))
    elif operation == "transpose":
        approx = _fmt_matrix(A.T)
        checks.append(mv.check_agreement("(Aᵀ)ᵀ = A", A.T.T, A, 0.0))
    else:
        raise MathError("Matritsa amali: det, inverse, rank, eigen, solve, multiply, transpose.")
    return _result(f"Matritsa: {operation} ({m}×{n})", checks, exact, approx, notes)


# ====================== DIFFERENSIAL TENGLAMALAR ======================
class _TooLong(Exception):
    pass


def op_ode(rhs, t0=0, y0=1, t_end=5, order=1, v0=None, variable="t"):
    """y' = rhs(t, y) yoki y'' = rhs(t, y, v), v = y'."""
    node = mp.parse(rhs)
    var = str(variable or "t").strip()
    order = int(order or 1)
    if order not in (1, 2) or var in ("y", "v"):
        raise MathError("Tartib 1 yoki 2; erkin o'zgaruvchi y yoki v bo'lmasin.")
    allowed = {var, "y"} | ({"v"} if order == 2 else set())
    if mp.variables(node) - allowed:
        raise MathError(f"O'ng tomonda faqat {', '.join(sorted(allowed))} qatnashsin.")
    a, b = mp.parse_number(t0), mp.parse_number(t_end)
    if not (math.isfinite(a) and math.isfinite(b) and a != b):
        raise MathError("Vaqt oralig'i chekli va nolga teng bo'lmasin.")
    init = [mp.parse_number(y0)]
    if order == 2:
        if not _given(v0):
            raise MathError("2-tartibli tenglama uchun y'(t0) qiymatini kiriting.")
        init.append(mp.parse_number(v0))
    calls = [0]

    def system(t, s):
        calls[0] += 1
        if calls[0] > 100_000:
            raise _TooLong
        env = {var: t, "y": s[0]}
        if order == 2:
            env["v"] = s[1]
        d = float(mp.evaluate(node, env))
        if not math.isfinite(d):
            raise MathError(f"O'ng tomon {var} = {t:.6g} nuqtada aniqlanmagan (yechim portlashi yoki soha chegarasi).")
        return [d] if order == 1 else [s[1], d]

    grid = np.linspace(a, b, 301)
    sols = []
    try:
        for method, tol in (("DOP853", 1e-10), ("Radau", 1e-9)):
            calls[0] = 0
            r = integrate.solve_ivp(system, (a, b), init, method=method, t_eval=grid, rtol=tol, atol=tol * 1e-2)
            if not r.success:
                raise MathError("Sonli yechuvchi oxirigacha yetmadi (yechim cheksizlikka ketishi mumkin): " + r.message)
            sols.append(r.y)
    except _TooLong:
        raise MathError("Hisoblash juda cho'zildi: oraliqni qisqartiring (tenglama qattiq yoki yechim portlashi mumkin).") from None
    check = mv.check_agreement("ikki mustaqil sonli usul: DOP853 va Radau", sols[0][0], sols[1][0], 1e-5)
    checks, notes, exact = [check], ["Sonli yechim: rtol = 1e-10; farq yuqoridagi tekshiruvda ko'rsatilgan."], None
    s = sym.run("ode", rhs=rhs, variable=var, order=order, t0=str(t0), y0=str(y0), v0=None if v0 is None else str(v0))
    if s["ok"]:
        exact = " ;  ".join(x["text"] for x in s["solutions"])
        plain = s["solutions"][0].get("plain")
        if len(s["solutions"]) == 1 and plain and mp.variables(mp.parse(plain)) <= {var}:
            values = np.broadcast_to(mp.evaluate(mp.parse(plain), {var: grid}), grid.shape)
            checks.append(mv.check_agreement("SymPy yechimi va sonli yechim", sols[0][0], values, 1e-5))
        else:
            notes.append("Aniq yechim sonli yechim bilan avtomatik solishtirilmadi.")
    else:
        notes.append("Aniq yechim olinmadi: " + s["error"])
    approx = f"y({_num(b)}) ≈ {_num(sols[0][0][-1], 9)}" + (f",  y'({_num(b)}) ≈ {_num(sols[0][1][-1], 9)}" if order == 2 else "")
    order_x = np.argsort(grid)
    plot = {"kind": "series", "x": grid[order_x].tolist(), "series": {"y": sols[0][0][order_x].tolist()}, "xlabel": var}
    return _result("Oddiy differensial tenglama", checks, exact, approx, notes, plot, {"y_end": float(sols[0][0][-1])})


def _heat_cn(f, alpha, L, T, nx, nt, left, right):
    x = np.linspace(0, L, nx + 1)
    u = np.array(np.broadcast_to(f(x), x.shape), dtype=float)
    u[0], u[-1] = left, right
    dx, dt = L / nx, T / nt
    r = alpha * dt / (2 * dx * dx)
    ab = np.zeros((3, nx - 1))
    ab[0, 1:], ab[1, :], ab[2, :-1] = -r, 1 + 2 * r, -r

    def step(u, A, rr):
        d = u[1:-1] + rr * (u[2:] - 2 * u[1:-1] + u[:-2])
        d[0] += rr * left
        d[-1] += rr * right
        u[1:-1] = linalg.solve_banded((1, 1), A, d)

    # Rannacher: dastlabki ikki qadam yarim qadamli oshkormas Eyler (uzilishli boshlang'ich shartdagi tebranishni so'ndiradi)
    for _ in range(4):
        d = u[1:-1].copy()
        d[0] += r * left
        d[-1] += r * right
        u[1:-1] = linalg.solve_banded((1, 1), ab, d)
    for _ in range(nt - 2):
        step(u, ab, r)
    return x, u


def op_heat(initial, alpha=1, length=1, t_end=0.1, left=0, right=0):
    """u_t = α·u_xx, 0<x<L, u(x,0)=initial(x), u(0,t)=left, u(L,t)=right."""
    node = mp.parse(initial)
    if mp.variables(node) - {"x"}:
        raise MathError("Boshlang'ich shart faqat x ga bog'liq bo'lsin.")
    a, L, T, lv, rv = (mp.parse_number(v) for v in (alpha, length, t_end, left, right))
    if not (a > 0 and L > 0 and T > 0 and math.isfinite(a * L * T)):
        raise MathError("α, L va T musbat bo'lsin.")
    f = mp.make_function(node, "x")
    if not np.all(np.isfinite(f(np.linspace(0, L, 801)))):
        raise MathError("Boshlang'ich funksiya [0, L] da aniqlanmagan.")
    x1, u1 = _heat_cn(f, a, L, T, 200, 400, lv, rv)
    x2, u2 = _heat_cn(f, a, L, T, 400, 800, lv, rv)
    diff = float(np.max(np.abs(u2[::2] - u1)))
    check = mv.check_agreement("to'rni 2 marta zichlashtirib solishtirish (200×400 va 400×800)", u2[::2], u1, 1e-3)
    notes = [f"Crank–Nicolson usuli. Diskretlash xatosi bahosi (maks. farq): {diff:.2e}.",
             "Faqat 1D issiqlik tenglamasi (Dirixle shartlari) qo'llab-quvvatlanadi."]
    if abs(f(0.0) - lv) > 1e-6 or abs(f(L) - rv) > 1e-6:
        notes.append("Boshlang'ich va chegaraviy shartlar uchlarda mos emas: t≈0 yaqinida aniqlik past.")
    approx = ";  ".join(f"u({_num(x2[i], 4)}, {_num(T, 4)}) ≈ {_num(u2[i], 6)}" for i in (100, 200, 300))
    plot = {"kind": "series", "x": x2.tolist(), "xlabel": "x",
            "series": {"t = 0": np.array(np.broadcast_to(f(x2), x2.shape)).tolist(), f"t = {T:g}": u2.tolist()}}
    return _result("Issiqlik o'tkazuvchanlik tenglamasi (PDE)", [check], None, approx, notes, plot,
                   {"x": x2.tolist(), "u": u2.tolist()})


# ====================== EHTIMOLLIK VA STATISTIKA ======================
def op_statistics(data, data_y=None):
    x = np.array(mp.parse_data(data))
    values = x.tolist()
    mean, med = float(np.mean(x)), float(np.median(x))
    rows = [("n", len(x)), ("o'rtacha", mean), ("mediana", med), ("min", x.min()), ("max", x.max()),
            ("Q1", np.percentile(x, 25)), ("Q3", np.percentile(x, 75))]
    checks = [mv.check_agreement("NumPy va statistics moduli: o'rtacha, mediana",
                                 [mean, med], [statistics.fmean(values), statistics.median(values)], 1e-9)]
    if len(x) > 1:
        var_s = float(np.var(x, ddof=1))
        rows += [("dispersiya (tanlanma, n−1)", var_s), ("standart og'ish (tanlanma)", math.sqrt(var_s)),
                 ("dispersiya (bosh to'plam, n)", float(np.var(x)))]
        checks.append(mv.check_agreement("NumPy va statistics moduli: dispersiya", var_s, statistics.variance(values), 1e-7))
    notes = ["Kvartillar chiziqli interpolyatsiya usulida (boshqa dasturlarda biroz farq qilishi mumkin)."]
    if _given(data_y):
        y = np.array(mp.parse_data(data_y))
        if len(y) != len(x) or len(x) < 3 or np.ptp(x) == 0:
            raise MathError("Regressiya uchun x va y uzunligi teng (kamida 3) va x o'zgaruvchan bo'lsin.")
        reg = stats.linregress(x, y)
        coef = np.polyfit(x, y, 1)
        rows += [("regressiya: qiyalik b", reg.slope), ("regressiya: ozod had a", reg.intercept),
                 ("korrelyatsiya r", reg.rvalue), ("R²", reg.rvalue ** 2), ("p-qiymat (b=0)", reg.pvalue)]
        checks.append(mv.check_agreement("linregress va polyfit koeffitsiyentlari", [reg.slope, reg.intercept], coef, 1e-6))
    approx = ";  ".join(f"{k} = {_num(v, 10)}" for k, v in rows)
    return _result("Statistika", checks, None, approx, notes, {"kind": "histogram", "data": values},
                   {k: float(v) for k, v in rows})


def op_probability(distribution, k=None, n=None, p=None, mean=0, std=1, lam=None, kind="le"):
    """kind: eq (=k), le (≤k), ge (≥k). Normal uchun eq yo'q."""
    d = str(distribution).strip().lower()
    if not _given(k) or kind not in ("eq", "le", "ge"):
        raise MathError("k qiymati va tur (eq, le, ge) kerak.")
    x = mp.parse_number(k)
    sign = {"eq": "=", "le": "≤", "ge": "≥"}[kind]
    if d in ("binomial", "poisson"):
        if x != int(x) or x < 0:
            raise MathError("k manfiy bo'lmagan butun son bo'lsin.")
        x = int(x)
        if d == "binomial":
            if not (_given(n) and _given(p)):
                raise MathError("Binomial uchun n va p kerak.")
            nv, pv = mp.parse_number(n), mp.parse_number(p)
            if nv != int(nv) or not 1 <= nv <= 100_000 or not 0 <= pv <= 1:
                raise MathError("n 1..100000 butun son, p 0..1 oralig'ida bo'lsin.")
            nv = int(nv)
            dist, top = stats.binom(nv, pv), nv

            def pmf(i):
                if pv in (0.0, 1.0):
                    return float(i == round(nv * pv))
                return math.exp(math.lgamma(nv + 1) - math.lgamma(i + 1) - math.lgamma(nv - i + 1)
                                + i * math.log(pv) + (nv - i) * math.log1p(-pv))
        else:
            if not _given(lam):
                raise MathError("Puasson uchun λ kerak.")
            lv = mp.parse_number(lam)
            if not 0 < lv <= 100_000:
                raise MathError("λ musbat va 100000 dan oshmasin.")
            dist, top = stats.poisson(lv), None
            pmf = lambda i: math.exp(-lv + i * math.log(lv) - math.lgamma(i + 1))
        if x > 200_000:
            raise MathError("k juda katta.")
        x_in = x if top is None else min(x, top)
        if kind == "eq":
            value = float(dist.pmf(x))
            other = pmf(x) if top is None or x <= top else 0.0
        elif kind == "le":
            value = float(dist.cdf(x))
            other = math.fsum(pmf(i) for i in range(x_in + 1))
        else:
            value = float(dist.sf(x - 1))
            other = (math.fsum(pmf(i) for i in range(x, top + 1)) if top is not None
                     else 1 - math.fsum(pmf(i) for i in range(x)))
        label = f"P(X {sign} {x})"
        method = "SciPy va ehtimolliklarni formula bo'yicha yig'ish"
    elif d == "normal":
        mu, s = mp.parse_number(mean), mp.parse_number(std)
        if s <= 0 or kind == "eq":
            raise MathError("Normal taqsimotda σ > 0 va tur le yoki ge bo'lsin (P(X = k) = 0).")
        z = (x - mu) / s
        value = float(stats.norm.cdf(z) if kind == "le" else stats.norm.sf(z))
        other = 0.5 * math.erfc((-z if kind == "le" else z) / math.sqrt(2))
        label = f"P(X {sign} {_num(x)})"
        method = "SciPy va erfc formulasi"
    else:
        raise MathError("Taqsimot: binomial, poisson, normal.")
    return _result("Ehtimollik", [mv.check_agreement(method, value, other, 1e-8)], None, f"{label} ≈ {_num(value, 10)}",
                   ["Mutlaq xato chegarasi taxminan 1e-8."], data={"value": value})


# ====================== OPTIMALLASHTIRISH, TRANSFORMATSIYALAR ======================
def op_optimize(expression, bounds, goal="min"):
    """bounds: 'x: -5, 5; y: 0, 3'."""
    node = mp.parse(expression)
    names = sorted(mp.variables(node))
    box = {}
    for item in mp.split_items(bounds):
        name, _, rest = item.partition(":")
        parts = rest.split(",")
        if len(parts) != 2:
            raise MathError("Chegaralarni 'x: -5, 5; y: 0, 3' ko'rinishida yozing.")
        box[name.strip()] = _range(parts[0].strip(), parts[1].strip())
    if not 1 <= len(names) <= 5 or set(box) != set(names):
        raise MathError("1–5 o'zgaruvchi va har biri uchun chegara kerak: " + ", ".join(names))
    if goal not in ("min", "max"):
        raise MathError("Maqsad min yoki max.")
    sign = 1.0 if goal == "min" else -1.0
    bnds = [box[n] for n in names]
    f = lambda v: float(mp.evaluate(node, dict(zip(names, v))))

    def objective(v):
        y = sign * f(v)
        return y if math.isfinite(y) else 1e300

    r = optimize.differential_evolution(objective, bnds, seed=1, tol=1e-10, maxiter=300, polish=True)
    best = r.x
    for x0 in (np.array([b[0] for b in bnds]), np.array([b[1] for b in bnds]), r.x):
        q = optimize.minimize(objective, x0, bounds=bnds, method="L-BFGS-B")
        if q.fun < objective(best):
            best = q.x
    if objective(best) >= 1e299 or not math.isfinite(f(best)):
        raise MathError("Berilgan sohada chekli qiymat topilmadi.")
    checks = [mv.check_minimum(f, best, bnds, maximize=goal == "max")]
    notes = ["Sonli global qidiruv: topilgan nuqta global optimum ekaniga kafolat yo'q."]
    if abs(f(best)) > 1e12:
        checks = [Check(UNKNOWN, "chegaralanganlik", "Optimum tasdiqlanmadi.")]
        notes.append("Qiymat juda katta: funksiya bu sohada chegaralanmagan bo'lishi mumkin.")
    if any(min(abs(v - lo), abs(v - hi)) < 1e-6 * (hi - lo) for v, (lo, hi) in zip(best, bnds)):
        notes.append("Optimum soha chegarasida.")
    approx = ", ".join(f"{n} ≈ {_num(v, 8)}" for n, v in zip(names, best)) + f";  f{goal} ≈ {_num(f(best), 10)}"
    plot = None
    if len(names) == 1:
        plot = {"kind": "function", "expressions": [expression], "var": names[0], "lo": bnds[0][0], "hi": bnds[0][1],
                "points": [(float(best[0]), f(best))]}
    elif len(names) == 2:
        plot = {"kind": "contour", "expression": expression, "var1": names[0], "var2": names[1],
                "bounds": bnds, "point": [float(v) for v in best]}
    return _result(f"Optimallashtirish ({goal})", checks, None, approx, notes, plot,
                   {"x": [float(v) for v in best], "value": f(best)})


def op_fourier_series(expression, half_period="pi", terms=5):
    """[-L, L] da Fourier qatori: a0/2 + Σ (an·cos(nπx/L) + bn·sin(nπx/L))."""
    node = mp.parse(expression)
    if mp.variables(node) - {"x"}:
        raise MathError("Funksiya faqat x ga bog'liq bo'lsin.")
    L, N = mp.parse_number(half_period), int(terms)
    if not (L > 0 and math.isfinite(L)) or not 1 <= N <= 30:
        raise MathError("L > 0, hadlar soni 1..30.")
    f = lambda x: float(mp.evaluate(node, {"x": x}))
    M = 20000  # o'rta nuqtalar (FFT'ga o'xshash mustaqil usul)
    xm = -L + (np.arange(M) + 0.5) * 2 * L / M
    ym = np.array(np.broadcast_to(mp.evaluate(node, {"x": xm}), xm.shape), dtype=float)
    if not np.all(np.isfinite(ym)):
        raise MathError("Funksiya oraliqda aniqlanmagan.")
    a, b, a2, b2, bad = [], [], [], [], False
    for n in range(N + 1):
        for trig, out, out2 in ((math.cos, a, a2), (math.sin, b, b2)):
            v, err, problems = mv.quadrature(lambda x: f(x) * trig(n * math.pi * x / L), -L, L)
            bad |= bool(problems) and err > 1e-6
            out.append(v / L)
            npt = np.cos if trig is math.cos else np.sin
            out2.append(float(np.sum(ym * npt(n * math.pi * xm / L)) * (2 * L / M) / L))
    check = mv.check_agreement("koeffitsiyentlar: adaptiv kvadratura va o'rta nuqta usuli", a + b, a2 + b2, 1e-4)
    if bad:
        check = Check(UNKNOWN, check.method, "Kvadratura yaqinlashishida muammo bor.")
    clean = lambda v: 0.0 if abs(v) < 1e-10 else v
    approx = f"a0 = {_num(clean(a[0]), 8)};  " + ";  ".join(
        f"a{n} = {_num(clean(a[n]), 8)}, b{n} = {_num(clean(b[n]), 8)}" for n in range(1, N + 1))
    xs = np.linspace(-L, L, 600)
    S = a[0] / 2 + sum(a[n] * np.cos(n * np.pi * xs / L) + b[n] * np.sin(n * np.pi * xs / L) for n in range(1, N + 1))
    fx = np.array(np.broadcast_to(mp.evaluate(node, {"x": xs}), xs.shape), dtype=float)
    rmse = float(np.sqrt(np.mean((S - fx) ** 2)))
    return _result("Fourier qatori", [check], None, approx,
                   [f"{N} hadli yig'indining o'rtacha kvadratik farqi: {rmse:.3g} (uzilishlarda Gibbs hodisasi)."],
                   {"kind": "series", "x": xs.tolist(), "xlabel": "x", "series": {"f(x)": fx.tolist(), f"S{N}(x)": S.tolist()}},
                   {"a": a, "b": b})


def op_laplace(expression, direction="forward", s_value=None):
    if direction not in ("forward", "inverse"):
        raise MathError("Yo'nalish forward yoki inverse.")
    node = mp.parse(expression)
    var = "t" if direction == "forward" else "s"
    if mp.variables(node) - {var}:
        raise MathError(f"Ifoda faqat {var} ga bog'liq bo'lsin.")
    notes, checks, exact, approx = [], [], None, None

    def transform(f_node, s):
        v, err, problems = mv.quadrature(lambda t: float(math.exp(-s * t) * mp.evaluate(f_node, {"t": t})) if s * t < 700 else 0.0,
                                         0, math.inf)
        return None if problems or not math.isfinite(v) or err > 1e-6 * max(1, abs(v)) else v

    r = sym.run("laplace" if direction == "forward" else "inverse_laplace", expression=expression)
    if r["ok"]:
        exact = r["result"]["text"]
        plain = r["result"].get("plain")
        pairs = []
        if plain:
            other = mp.parse(plain)
            for s in (2.5, 3.7, 6.0, 11.0):
                f_node, F_node = (node, other) if direction == "forward" else (other, node)
                num = transform(f_node, s) if mp.variables(f_node) <= {"t"} else None
                claimed = float(mp.evaluate(F_node, {"s": s}))
                if num is not None and math.isfinite(claimed):
                    pairs.append((claimed, num))
        if len(pairs) >= 2:
            checks.append(mv.check_agreement("Laplas integralini s ning bir nechta qiymatida sonli hisoblash", *zip(*pairs), 1e-6))
        else:
            checks.append(Check(UNKNOWN, "sonli integral bilan solishtirish", "Yetarli sinov nuqtasi olinmadi."))
    else:
        notes.append("Simvolik natija olinmadi: " + r["error"])
        if direction == "inverse" or not _given(s_value):
            raise MathError("SymPy'siz faqat to'g'ri transformatsiyaning berilgan s dagi sonli qiymati hisoblanadi. " + r["error"])
    if direction == "forward" and _given(s_value):
        s = mp.parse_number(s_value)
        v = transform(node, s)
        if v is None:
            raise MathError("Bu s qiymatida Laplas integrali sonli yaqinlashmadi.")
        approx = f"F({_num(s)}) ≈ {_num(v, 10)}"
        if not checks:
            checks.append(Check(UNKNOWN, "sonli integral", "Faqat bitta sonli usul."))
    return _result("Laplas transformatsiyasi" if direction == "forward" else "Teskari Laplas", checks, exact, approx, notes)


def op_fourier_transform(expression, frequency=None):
    """F(k) = ∫ f(x)·e^(−2πikx) dx (SymPy konvensiyasi)."""
    node = mp.parse(expression)
    if mp.variables(node) - {"x"}:
        raise MathError("Funksiya faqat x ga bog'liq bo'lsin.")
    notes, checks, exact, approx = ["Konvensiya: F(k) = ∫ f(x)·e^(−2πikx) dx."], [], None, None
    r = sym.run("fourier", expression=expression)
    if r["ok"]:
        exact = r["result"]["text"]
        notes.append("Simvolik natija avtomatik tekshirilmaydi; k qiymatini berib sonli natija bilan solishtiring.")
        checks.append(Check(UNKNOWN, "simvolik natija", "Mustaqil tekshirilmadi."))
    else:
        notes.append("Simvolik natija olinmadi: " + r["error"])
    if _given(frequency):
        k = mp.parse_number(frequency)
        parts = []
        for trig in (math.cos, lambda a: -math.sin(a)):
            v, err, problems = mv.quadrature(lambda x: float(mp.evaluate(node, {"x": x})) * trig(2 * math.pi * k * x),
                                             -math.inf, math.inf)
            if problems or err > 1e-6:
                raise MathError("Sonli integral yaqinlashmadi: funksiya cheksizlikda yetarlicha tez so'nsin.")
            parts.append(v)
        approx = f"F({_num(k)}) ≈ {_num(complex(*parts), 10)}"
        xs = np.linspace(-60, 60, 600001)
        ys = np.broadcast_to(mp.evaluate(node, {"x": xs}), xs.shape) * np.exp(-2j * np.pi * k * xs)
        other = np.trapezoid(ys, xs)
        checks.append(mv.check_agreement("trapetsiya usuli ([-60, 60]) bilan solishtirish", parts, [other.real, other.imag], 1e-5))
    elif not r["ok"]:
        raise MathError("SymPy'siz chastota (k) qiymatini kiriting. " + r["error"])
    return _result("Fourier transformatsiyasi", checks, exact, approx, notes)


# ====================== AMALLAR RO'YXATI ======================
def _f(name, label, default="", required=False, options=None):
    return {"name": name, "label": label, "default": default, "required": required, "options": options}


ALL, UNI, ADV = ("School", "University", "Advanced"), ("University", "Advanced"), ("Advanced",)
OPERATIONS = {
    "evaluate": ("Ifodani hisoblash", op_evaluate, ALL, [_f("expression", "Ifoda", "(3/4 + 1/6)^2", True), _f("values", "Qiymatlar (x=2; y=3)")]),
    "simplify": ("Soddalashtirish", op_simplify, ALL, [_f("expression", "Ifoda", "(x^2 - 1)/(x - 1)", True)]),
    "solve_equation": ("Tenglama", op_solve_equation, ALL, [_f("equation", "Tenglama", "x^2 - 5x + 6 = 0", True), _f("variable", "O'zgaruvchi"), _f("lower", "Qidiruv: dan"), _f("upper", "gacha")]),
    "inequality": ("Tengsizlik", op_inequality, ALL, [_f("inequality", "Tengsizlik", "x^2 - 4 > 0", True), _f("variable", "O'zgaruvchi"), _f("lower", "Qidiruv: dan"), _f("upper", "gacha")]),
    "system": ("Tenglamalar tizimi", op_system, ALL, [_f("equations", "Tenglamalar (; bilan)", "x + y = 5; x - y = 1", True), _f("variables", "Noma'lumlar")]),
    "derivative": ("Hosila", op_derivative, UNI, [_f("expression", "f", "x^3*sin(x)", True), _f("variable", "O'zgaruvchi"), _f("order", "Tartib", "1"), _f("at", "Nuqta (ixtiyoriy)", "1")]),
    "integral": ("Integral", op_integral, UNI, [_f("expression", "f", "x*exp(-x)", True), _f("variable", "O'zgaruvchi"), _f("lower", "Quyi chegara", "0"), _f("upper", "Yuqori chegara", "inf")]),
    "limit": ("Limit", op_limit, UNI, [_f("expression", "f", "sin(x)/x", True), _f("point", "Nuqta", "0", True), _f("variable", "O'zgaruvchi"), _f("direction", "Yo'nalish", "both", options=["both", "left", "right"])]),
    "matrix": ("Matritsa", op_matrix, UNI, [_f("matrix", "A (qatorlar ; bilan)", "2 1; 1 3", True), _f("operation", "Amal", "det", True, ["det", "inverse", "rank", "eigen", "solve", "multiply", "transpose"]), _f("other", "b yoki B (solve/multiply uchun)")]),
    "ode": ("Differensial tenglama (ODE)", op_ode, UNI, [_f("rhs", "y' = ... (2-tartibda y'' = ..., v = y')", "-2*y", True), _f("t0", "t0", "0"), _f("y0", "y(t0)", "1"), _f("t_end", "t oxiri", "3"), _f("order", "Tartib", "1", options=["1", "2"]), _f("v0", "y'(t0)"), _f("variable", "Erkin o'zgaruvchi", "t")]),
    "statistics": ("Statistika", op_statistics, ALL, [_f("data", "Ma'lumotlar", "12, 15, 11, 19, 14", True), _f("data_y", "y (regressiya uchun)")]),
    "probability": ("Ehtimollik", op_probability, UNI, [_f("distribution", "Taqsimot", "binomial", True, ["binomial", "poisson", "normal"]), _f("k", "k", "3", True), _f("kind", "Tur", "le", options=["eq", "le", "ge"]), _f("n", "n", "10"), _f("p", "p", "0.5"), _f("lam", "λ"), _f("mean", "μ", "0"), _f("std", "σ", "1")]),
    "gradient": ("Gradiyent (ko'p o'zgaruvchi)", op_gradient, ADV, [_f("expression", "f(x, y, ...)", "x^2*y + sin(y)", True), _f("point", "Nuqta", "x=1; y=2", True)]),
    "optimize": ("Optimallashtirish", op_optimize, ADV, [_f("expression", "f", "(x-1)^2 + (y+2)^2", True), _f("bounds", "Chegaralar", "x: -5, 5; y: -5, 5", True), _f("goal", "Maqsad", "min", options=["min", "max"])]),
    "heat": ("PDE: issiqlik tenglamasi", op_heat, ADV, [_f("initial", "u(x,0)", "sin(pi*x)", True), _f("alpha", "α", "1"), _f("length", "L", "1"), _f("t_end", "T", "0.1"), _f("left", "u(0,t)", "0"), _f("right", "u(L,t)", "0")]),
    "fourier_series": ("Fourier qatori", op_fourier_series, ADV, [_f("expression", "f(x)", "x", True), _f("half_period", "L", "pi"), _f("terms", "Hadlar soni", "5")]),
    "fourier_transform": ("Fourier transformatsiyasi", op_fourier_transform, ADV, [_f("expression", "f(x)", "exp(-x^2)", True), _f("frequency", "k", "0.5")]),
    "laplace": ("Laplas transformatsiyasi", op_laplace, ADV, [_f("expression", "f(t) yoki F(s)", "t*exp(-t)", True), _f("direction", "Yo'nalish", "forward", options=["forward", "inverse"]), _f("s_value", "s (sonli qiymat)", "2")]),
}


def run_operation(name, params):
    """Amalni bajaradi. Xato bo'lsa {'error': ...} qaytaradi (hech qachon istisno tashlamaydi)."""
    try:
        if name not in OPERATIONS:
            raise MathError("Noma'lum amal: " + str(name)[:40])
        _, func, _, fields = OPERATIONS[name]
        allowed = {f["name"] for f in fields}
        if not isinstance(params, dict) or set(params) - allowed:
            raise MathError("Noma'lum parametr. Ruxsat etilganlari: " + ", ".join(sorted(allowed)))
        clean = {k: v for k, v in params.items() if v not in (None, "")}
        if any(len(str(v)) > 20000 for v in clean.values()):
            raise MathError("Kirish matni juda uzun.")
        missing = [f["name"] for f in fields if f["required"] and f["name"] not in clean]
        if missing:
            raise MathError("To'ldirilmagan maydon: " + ", ".join(missing))
        return func(**clean)
    except (MathParseError, MathError) as e:
        return {"error": str(e)}
    except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError, RecursionError, MemoryError) as e:
        return {"error": f"Hisoblab bo'lmadi: {type(e).__name__}: {e}"[:300]}


_ICON = {PASS: "✅ tekshiruvdan o'tdi", FAIL: "❌ TEKSHIRUVDAN O'TMADI", UNKNOWN: "⚠️ to'liq tekshirib bo'lmadi"}


def _code(text):
    return "`" + str(text).replace("`", "'") + "`"


def format_result(res, steps=True):
    """Natijani markdown ko'rinishida: aniq va taqribiy qiymatlar alohida, tekshiruv holati bilan."""
    if "error" in res:
        return "❗ " + res["error"]
    lines = [f"**{res['title']}** — {_ICON[mv.overall(res['checks'])]}"]
    if res["exact"]:
        lines.append(f"- Aniq natija: {_code(res['exact'])}")
    if res["approx"]:
        lines.append(f"- Taqribiy natija: {_code(res['approx'])}")
    checks = res["checks"] if steps else [c for c in res["checks"] if c.status != PASS][:3]
    for c in checks[:12]:
        extra = ""
        if c.error is not None and c.tolerance is not None:
            extra = f" (xato {c.error:.1e}, chegara {c.tolerance:.1e})"
        lines.append(f"- {_ICON[c.status].split()[0]} Tekshiruv: {c.method}{extra}. {c.detail}".rstrip())
    notes = res["notes"] if steps else res["notes"][:2]
    lines += [f"- Eslatma: {n}" for n in notes]
    return "\n".join(lines)


# ====================== GROQ AI ======================
LEVELS = {
    "School": "Maktab darajasi: arifmetika, algebra, tenglama va tengsizliklar, geometriya, trigonometriya, funksiyalar, logarifmlar. Maktab o'quvchisi tushunadigan tilda yoz.",
    "University": "Universitet darajasi: matematik analiz, limit, hosila, integral, chiziqli algebra, matritsalar, differensial tenglamalar, ehtimollar nazariyasi va statistika.",
    "Advanced": "Ilg'or va amaliy daraja: murakkab analiz, ko'p o'zgaruvchili funksiyalar, ODE va PDE, sonli usullar, optimallashtirish, Fourier va Laplas transformatsiyalari, ilmiy hisoblashlar.",
}
LEVEL_NAMES = {"School": "🏫 School", "University": "🎓 University", "Advanced": "🔬 Advanced & Applied"}
ANSWER_MODES = {"quick": "⚡ Quick Answer", "steps": "🪜 Step-by-Step"}


def system_prompt(level, answer_mode):
    style = ("REJIM: tezkor javob. Natijani birinchi qatorda ber, ko'pi bilan 3–4 qisqa gap." if answer_mode == "quick" else
             "REJIM: bosqichma-bosqich. Berilganlar, ishlatiladigan formulalar, har bir qadam va uning izohi, yakuniy javob va tekshirish.")
    return (
        "Sen The Economics AI platformasining Mathematics AI yordamchisisan. Ravon o'zbek tilida (lotin) yoz.\n"
        + LEVELS[level] + "\n" + style + "\n"
        "Hisoblanadigan har qanday natijani math_calculate asbobi bilan ol va asbob natijasini o'zgartirma. "
        "Tekshiruv holati FAIL yoki UNKNOWN bo'lsa, buni ochiq ayt. Aniq va taqribiy qiymatlarni ajrat. "
        "Asbob qo'llanmaydigan masalada (isbot, geometriya mulohazasi) buni ayt. 100% aniqlik va'da qilma. "
        "Masala sharti to'liq yoki aniq bo'lmasa, taxmin qilmasdan aniqlashtiruvchi savol ber. "
        "Formulalarni LaTeX bilan $...$ yoki $$...$$ ichida yoz. Dasturiy tekshiruv bloki javobingga avtomatik qo'shiladi, uni o'zing yozma.\n"
        "Ifoda sintaksisi: ^ daraja, sqrt(x), log(x) natural, log(x, a), abs(x), pi, e, inf; o'zgaruvchilar bitta harf.\n"
        "Amallar va parametrlari (* majburiy):\n" +
        "\n".join(f"- {k}({', '.join(f['name'] + ('*' if f['required'] else '') + ('=' + '|'.join(f['options']) if f['options'] else '') for f in v[3])})"
                  for k, v in OPERATIONS.items())
    )


TOOLS = [{"type": "function", "function": {
    "name": "math_calculate",
    "description": "Matematik amalni dasturiy bajaradi va mustaqil tekshiradi.",
    "parameters": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "params": {"type": "object", "description": "Amal parametrlari (qiymatlar matn ko'rinishida)"}},
        "required": ["operation", "params"]}}}]


def pick_model(ids, vision=False, override=None):
    """Groq ro'yxatidan matematika uchun model tanlaydi. override: MATH_MODEL / MATH_VISION_MODEL."""
    if override:
        return override
    prefs = ("qwen3.8", "qwen3", "vision", "llama-4") if vision else ("gpt-oss-120b", "qwen3.8", "llama-3.3-70b", "gpt-oss-20b", "qwen")
    usable = [m for m in ids if not any(x in m for x in ("whisper", "guard", "tts", "orpheus"))]
    for pref in prefs:
        for m in usable:
            if pref in m:
                return m
    if vision:
        return None
    return usable[0] if usable else "openai/gpt-oss-120b"


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def read_image(client, model, image_urls, user_text):
    """Rasmdagi masalani ko'chirib oladi. (matn, None) yoki (None, aniqlashtirish xabari)."""
    instruction = (
        "Rasmdagi matematik masalani AYNAN ko'chir. Hech narsani yechma. Biror belgi, son, daraja, indeks yoki chegara "
        "xira, kesilgan yoki ikki xil o'qilishi mumkin bo'lsa, TAXMIN QILMA: clear=false qil va unclear ro'yxatida aynan "
        "qaysi joy noaniqligini o'zbek tilida yoz. Faqat JSON qaytar: "
        '{"clear": true|false, "problem": "masala matni", "unclear": ["..."]}\n'
        "Foydalanuvchi izohi (noaniqlikni hal qilishda ishlat): " + user_text)
    content = [{"type": "text", "text": instruction}] + [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    request = dict(model=model, messages=[{"role": "user", "content": content}], temperature=0)
    try:
        try:
            r = client.chat.completions.create(response_format={"type": "json_object"}, **request)
        except Exception:
            r = client.chat.completions.create(**request)
        text = _strip_think(r.choices[0].message.content)
    except Exception as e:
        return None, _api_error(e, "Rasmni o'qib bo'lmadi")
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        problem = str(data.get("problem") or "").strip()
        clear = data.get("clear") is True
        unclear = data.get("unclear") or []
        unclear = [str(u) for u in (unclear if isinstance(unclear, list) else [unclear])]
    except (ValueError, AttributeError):
        problem, clear, unclear = "", False, []
    if clear and problem and not unclear:
        return problem, None
    msg = "🔎 **Rasmni aniq o'qiy olmadim, shuning uchun taxmin qilmayman. Aniqlashtiring:**\n"
    msg += "".join(f"\n- {u}" for u in unclear) or "\n- Masalani matn ko'rinishida yozib yuboring yoki tiniqroq rasm yuklang."
    if problem:
        msg += f"\n\nO'qiy olgan qismim (tekshiring):\n\n> {problem}"
    msg += "\n\nTo'g'rilangan masalani to'liq matn qilib yozing."
    return None, msg


def _api_error(e, prefix="Xato"):
    if "rate" in str(e).lower() or "429" in str(e):
        return "So'rovlar limiti to'ldi. Bir daqiqadan keyin qayta urinib ko'ring."
    return f"{prefix}: {str(e)[:300]}"


def verification_block(results, answer_mode):
    if not results:
        return ("\n\n---\n⚠️ **Dasturiy tekshiruv bajarilmadi:** bu javob faqat AI matni. Muhim natijani "
                "«Matematik kalkulyator» bo'limida tekshiring.")
    body = "\n\n".join(format_result(r, steps=answer_mode == "steps") for r in results)
    return (f"\n\n---\n#### 🧪 Dasturiy tekshiruv\n{body}\n\n_Tekshiruv asbobga berilgan ifodalarga tegishli: masala shartdan "
            f"to'g'ri ko'chirilganini va AI izohini o'zingiz ham tekshiring. {DISCLAIMER}_")


def answer(client, prompt, level, answer_mode, history=(), image_urls=None, doc="", model=None, vision_model=None):
    """Savolga javob. (markdown matn, grafik tavsifi yoki None) qaytaradi."""
    level = level if level in LEVELS else "School"
    intro = ""
    if image_urls:
        if not vision_model:
            return "Rasm o'qiy oladigan Groq modeli topilmadi. Masalani matn ko'rinishida yozing.", None
        if len(image_urls) > 3:
            return "Bir martada ko'pi bilan 3 ta rasm yuboring.", None
        problem, clarify = read_image(client, vision_model, image_urls, prompt)
        if clarify:
            return clarify, None
        intro = f"📷 **Rasmdan o'qilgan masala** (to'g'riligini tekshiring):\n\n> {problem}\n\n"
        prompt = f"{prompt}\n\nRasmdagi masala:\n{problem}"
    system = system_prompt(level, answer_mode)
    if doc:
        system += "\nYuklangan fayl:\n" + doc[:4000]
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": m["role"], "content": m["content"][:3000]} for m in list(history)[-6:]]
    msgs.append({"role": "user", "content": prompt})
    results, text, warning = [], None, ""
    try:
        for step in range(6):
            r = client.chat.completions.create(model=model, messages=msgs, tools=TOOLS, tool_choice="auto",
                                               temperature=0.1).choices[0].message
            if not r.tool_calls:
                text = r.content
                break
            msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in r.tool_calls]})
            for t in r.tool_calls:
                try:
                    args = json.loads(t.function.arguments or "{}")
                    if t.function.name != "math_calculate":
                        raise ValueError
                    res = run_operation(args.get("operation"), args.get("params") or {})
                except (ValueError, AttributeError):
                    res = {"error": "Asbob so'rovi noto'g'ri."}
                if "error" not in res:
                    results.append(res)
                payload = ({"xato": res["error"]} if "error" in res else
                           {"aniq": res["exact"], "taqribiy": res["approx"], "tekshiruv": mv.overall(res["checks"]),
                            "izohlar": res["notes"], "tekshiruvlar": [c.as_dict() for c in res["checks"][:8]]})
                msgs.append({"role": "tool", "tool_call_id": t.id,
                             "content": json.dumps(payload, ensure_ascii=False, default=str)[:6000]})
        if text is None:
            text = "Yechim juda ko'p qadam talab qildi. Quyida bajarilgan hisoblar; savolni bo'laklab yuboring."
    except Exception as first:
        try:  # asbobsiz zaxira urinish; natija tekshirilmagan deb belgilanadi
            plain = [m for m in msgs if m["role"] != "tool" and not m.get("tool_calls")]
            text = client.chat.completions.create(model=model, messages=plain, temperature=0.1).choices[0].message.content
            warning = f"⚠️ Asbobli so'rov bajarilmadi ({_api_error(first)[:150]}). Quyidagi AI matni asbobsiz yozildi.\n\n"
        except Exception as e:
            return intro + _api_error(e) + (verification_block(results, answer_mode) if results else ""), None
    plot = next((r["plot"] for r in reversed(results) if r.get("plot")), None)
    return intro + warning + _strip_think(text) + verification_block(results, answer_mode), plot


# ====================== STREAMLIT PANELI ======================
def _show_plot(st, spec):
    import math_visualizer
    try:
        st.image(math_visualizer.render(spec))
    except Exception as e:
        st.caption(f"Grafikni chizib bo'lmadi: {e}")


def panel(st):
    """Daraja va rejim tanlash, to'g'ridan-to'g'ri kalkulyator. (daraja, rejim) qaytaradi."""
    c1, c2 = st.columns(2)
    level = c1.selectbox("Daraja", list(LEVELS), format_func=LEVEL_NAMES.get, key="math_level")
    mode = c2.radio("Javob rejimi", list(ANSWER_MODES), format_func=ANSWER_MODES.get, horizontal=True, key="math_answer_mode")
    with st.expander("🧮 Matematik kalkulyator (AI'siz, dasturiy tekshiruv bilan)"):
        names = [k for k, v in OPERATIONS.items() if level in v[2]]
        op = st.selectbox("Amal", names, format_func=lambda k: OPERATIONS[k][0], key="math_op_" + level)
        with st.form("math_calc_" + op):
            params = {}
            for f in OPERATIONS[op][3]:
                key = f"math_{op}_{f['name']}"
                if f["options"]:
                    params[f["name"]] = st.selectbox(f["label"], f["options"], index=f["options"].index(f["default"] or f["options"][0]), key=key)
                else:
                    params[f["name"]] = st.text_input(f["label"], value=f["default"], key=key)
            ok = st.form_submit_button("Hisoblash", use_container_width=True)
        if ok:
            with st.spinner("Hisoblanmoqda..."):
                res = run_operation(op, params)
            st.markdown(format_result(res, steps=mode == "steps"))
            if res.get("plot"):
                _show_plot(st, res["plot"])
            st.caption(DISCLAIMER)
    return level, mode


def chat_reply(st, client, prompt, level, answer_mode, history, imgs, doc, img_to_url):
    """app.py dagi chat uchun javob. Grafik tavsifini session_state ga yozadi."""
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception:
        ids = []
    model = pick_model(ids, override=os.getenv("MATH_MODEL"))
    vision = pick_model(ids, vision=True, override=os.getenv("MATH_VISION_MODEL"))
    try:
        urls = [img_to_url(f) for f in imgs] if imgs else None
    except Exception as e:
        return "Rasmni o'qib bo'lmadi: " + str(e)
    text, plot = answer(client, prompt, level, answer_mode, history, urls, doc, model, vision)
    st.session_state["math_plot"] = (prompt, plot) if plot else None
    return text


def show_chat_plot(st, messages):
    """Oxirgi javobga tegishli grafik (faqat o'sha suhbatda ko'rsatiladi; tarixga saqlanmaydi)."""
    saved = st.session_state.get("math_plot")
    if saved and len(messages) >= 2 and str(messages[-2].get("content", "")).startswith(saved[0]):
        _show_plot(st, saved[1])
