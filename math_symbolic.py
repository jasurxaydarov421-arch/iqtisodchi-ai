"""Aniq (simvolik) hisoblash: SymPy.

Alohida jarayonda ishga tushiriladi, shunda og'ir ifoda vaqt chegarasida to'xtatiladi:
    matematika.py  ->  python math_symbolic.py  (stdin: JSON so'rov, stdout: JSON javob)
SymPy ifodalari math_parser daraxtidan quriladi; sympify()/parse_expr() ishlatilmaydi.
Natijadagi ifodalar math_parser formatidagi satr sifatida ham qaytariladi, shunda ularni
math_verifier SymPy'dan mustaqil ravishda sonli tekshira oladi.
"""
import json
import math
import os
import re
import subprocess
import sys

import math_parser as mp

TIMEOUT = 20  # soniya


def available():
    try:
        import sympy  # noqa: F401
    except ImportError:
        return False
    return True


def run(task, **params):
    """Simvolik vazifani cheklangan vaqtli jarayonda bajaradi. {'ok': bool, ...} qaytaradi."""
    if not available():
        return {"ok": False, "error": "SymPy o'rnatilmagan (pip install -r requirements.txt)."}
    request = json.dumps({"task": task, "params": params})
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__)], input=request, capture_output=True,
                           text=True, timeout=TIMEOUT, encoding="utf-8")
        return json.loads(p.stdout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Aniq hisoblash {TIMEOUT} soniyada tugamadi."}
    except (OSError, ValueError):
        return {"ok": False, "error": "Aniq hisoblash jarayoni xato bilan tugadi."}


# ---------------------------------------------------------------- jarayon ichida
def _converter(sp):
    funcs = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
             "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh,
             "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh, "exp": sp.exp,
             "sqrt": sp.sqrt, "cbrt": lambda x: sp.real_root(x, 3), "abs": sp.Abs,
             "log": sp.log, "log10": lambda x: sp.log(x, 10), "log2": lambda x: sp.log(x, 2),
             "floor": sp.floor, "ceil": sp.ceiling, "factorial": sp.factorial, "gamma": sp.gamma,
             "erf": sp.erf, "heaviside": sp.Heaviside}

    def convert(n, symbols):
        if n.kind == "num":
            return sp.Rational(n.value)
        if n.kind == "const":
            return {"pi": sp.pi, "e": sp.E}[n.value]
        if n.kind == "var":
            return symbols.setdefault(n.value, sp.Symbol(n.value, real=True))
        a = [convert(x, symbols) for x in n.args]
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
            return a[0] ** a[1]
        return funcs[n.value](*a)

    return convert


def to_plain(sp, expr):
    """SymPy ifodasini math_parser tushunadigan satrga aylantiradi, bo'lmasa None."""
    try:
        if expr.has(sp.I):  # kompleks natija sonli tekshiruvchiga berilmaydi
            return None
        text = sp.sstr(expr.replace(sp.Abs, sp.Function("abs")))
        text = text.replace("**", "^").replace("ceiling(", "ceil(").replace("Heaviside(", "heaviside(")
        text = re.sub(r"\bE\b", "e", text)
        mp.parse(text)
        return text
    except Exception:
        return None


def _pack(sp, expr):
    out = {"text": sp.sstr(expr), "latex": sp.latex(expr), "plain": to_plain(sp, expr)}
    try:
        if not expr.free_symbols:
            v = complex(sp.N(expr, 30))
            out["value"] = [v.real, v.imag]
            out["digits"] = str(sp.N(expr, 30))
    except (TypeError, ValueError, AttributeError):
        pass
    return out


def _task(sp, task, p):
    convert = _converter(sp)
    symbols = {}
    ex = lambda text: convert(mp.parse(text), symbols)
    sym = lambda name: symbols.setdefault(name, sp.Symbol(name, real=True))

    def bound(text):
        v = mp.parse_number(text)
        if math.isinf(v):
            return sp.oo if v > 0 else -sp.oo
        return ex(text)

    if task == "evaluate":
        return {"result": _pack(sp, sp.simplify(ex(p["expression"])))}
    if task == "simplify":
        e = ex(p["expression"])
        return {"result": _pack(sp, sp.simplify(e)), "factored": _pack(sp, sp.factor(e))}
    if task == "solve":
        eqs = []
        for text in mp.split_items(p["equations"]):
            lhs, _, rhs = mp.parse_relation(text)
            eqs.append(convert(lhs, symbols) - convert(rhs, symbols))
        unknowns = [sym(v) for v in p["variables"]]
        sols = sp.solve(eqs, unknowns, dict=True)
        return {"solutions": [{str(k): _pack(sp, v) for k, v in s.items()} for s in sols[:30]]}
    if task == "inequality":
        lhs, op, rhs = mp.parse_relation(p["inequality"])
        a, b = convert(lhs, symbols), convert(rhs, symbols)
        rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](a, b)
        solset = sp.solveset(rel, sym(p["variable"]), domain=sp.S.Reals)
        if isinstance(solset, sp.ConditionSet):
            raise ValueError("SymPy bu tengsizlikni aniq yecha olmadi.")
        return {"result": {"text": sp.sstr(solset), "latex": sp.latex(solset)}}
    if task == "derivative":
        d = sp.diff(ex(p["expression"]), sym(p["variable"]), int(p["order"]))
        return {"result": _pack(sp, sp.simplify(d))}
    if task == "integral":
        f, x = ex(p["expression"]), sym(p["variable"])
        if p.get("lower") is None:
            r = sp.integrate(f, x)
        else:
            r = sp.integrate(f, (x, bound(p["lower"]), bound(p["upper"])))
        if r.has(sp.Integral):
            raise ValueError("SymPy bu integralning yopiq ko'rinishini topa olmadi.")
        return {"result": _pack(sp, sp.simplify(r))}
    if task == "limit":
        d = {"both": "+-", "left": "-", "right": "+"}[p["direction"]]
        point = bound(p["point"])
        if point in (sp.oo, -sp.oo):
            d = "-" if point == sp.oo else "+"
        r = sp.limit(ex(p["expression"]), sym(p["variable"]), point, d)
        if r.has(sp.Limit) or isinstance(r, sp.AccumBounds) or r in (sp.zoo, sp.nan):
            raise ValueError("Limit mavjud emas yoki aniqlanmadi.")
        return {"result": _pack(sp, r)}
    if task == "ode":
        t, y = sp.Symbol(p["variable"], real=True), sp.Function("y")
        order = int(p["order"])
        symbols[p["variable"]] = t
        symbols["y"], symbols["v"] = sp.Symbol("Y_"), sp.Symbol("V_")
        rhs = ex(p["rhs"]).subs({symbols["y"]: y(t), symbols["v"]: y(t).diff(t)})
        ics = {}
        if p.get("y0") is not None:
            t0 = ex(p["t0"])
            ics[y(t0)] = ex(p["y0"])
            if order == 2 and p.get("v0") is not None:
                ics[y(t).diff(t).subs(t, t0)] = ex(p["v0"])
        sol = sp.dsolve(sp.Eq(y(t).diff(t, order), rhs), y(t), ics=ics or None)
        sols = sol if isinstance(sol, list) else [sol]
        return {"solutions": [{"text": sp.sstr(s), "latex": sp.latex(s),
                               "plain": to_plain(sp, s.rhs) if s.lhs == y(t) else None} for s in sols]}
    if task in ("laplace", "inverse_laplace"):
        t, s = sp.Symbol("t", positive=True), sp.Symbol("s", positive=True)
        symbols["t"], symbols["s"] = t, s
        f = ex(p["expression"])
        if task == "laplace":
            r = sp.laplace_transform(f, t, s, noconds=True)
            bad = sp.LaplaceTransform
        else:
            r = sp.inverse_laplace_transform(f, s, t)
            bad = sp.InverseLaplaceTransform
        if r.has(bad):
            raise ValueError("SymPy bu transformatsiyani hisoblay olmadi.")
        return {"result": _pack(sp, sp.simplify(r))}
    if task == "fourier":
        x, k = sp.Symbol("x", real=True), sp.Symbol("k", real=True)
        symbols["x"] = x
        r = sp.fourier_transform(ex(p["expression"]), x, k)
        if r.has(sp.FourierTransform):
            raise ValueError("SymPy bu Fourier transformatsiyasini hisoblay olmadi.")
        return {"result": _pack(sp, r)}
    raise ValueError("Noma'lum simvolik vazifa.")


def main():
    try:
        try:  # Linux/macOS: xotirani cheklash
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3, 4 * 1024 ** 3))
        except (ImportError, ValueError, OSError):
            pass
        import sympy as sp
        request = json.load(sys.stdin)
        out = {"ok": True, **_task(sp, request["task"], request["params"])}
    except Exception as e:  # har qanday xato JSON bo'lib qaytadi
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
