"""Yechimlarni mustaqil tekshirish.

Har bir tekshiruv natijani boshqa usul bilan qayta sinaydi: ildizni asl tenglamaga qo'yish, hosilani
chekli ayirmalar bilan, integralni kvadratura bilan, matritsani shartlari bilan solishtirish va h.k.
Bu SONLI tekshiruv: «o'tdi» — xato topilmadi degani, matematik isbot emas.

Holatlar: PASS (o'tdi), FAIL (mos kelmadi), UNKNOWN (tekshirib bo'lmadi).
"""
import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import integrate

from math_parser import MathParseError, Node, evaluate, variables

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
SAMPLE_POINTS = (0.137, 0.419, 0.731, 1.093, 1.618, 2.236, 3.317, 5.099, -0.283, -0.577, -1.291, -2.718,
                 -4.123, 0.053, 0.887, 7.43, -6.91, 12.7)


@dataclass
class Check:
    status: str
    method: str
    detail: str = ""
    error: float = None      # kuzatilgan eng katta xato
    tolerance: float = None  # ruxsat etilgan chegara

    def as_dict(self):
        return {"status": self.status, "method": self.method, "detail": self.detail,
                "error": self.error, "tolerance": self.tolerance}


def _value(node, env, complex_mode=False):
    try:
        v = complex(evaluate(node, env, complex_mode))
    except (MathParseError, TypeError, OverflowError):
        return None
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    return v if complex_mode else v.real


def check_root(lhs, rhs, values, tol=1e-8):
    """Yechimni asl tenglamaga qo'yib tekshiradi. values: {o'zgaruvchi: son (kompleks bo'lishi mumkin)}."""
    method = "yechimni asl tenglamaga qo'yish"
    cm = any(isinstance(v, complex) and v.imag != 0 for v in values.values())
    if not cm:
        values = {k: float(np.real(v)) for k, v in values.items()}
    left, right = _value(lhs, values, cm), _value(rhs, values, cm)
    if left is None or right is None:
        return Check(FAIL, method, "Bu nuqtada tenglamaning biror tomoni aniqlanmagan (aniqlanish sohasidan tashqari).")
    scale = max(1.0, abs(left), abs(right))
    res = abs(left - right) / scale
    if res > tol:
        # Katta koeffitsiyentli tenglamada yaxlitlash qoldig'i: Nyuton qadami mashina aniqligida bo'lsa qabul qilinadi.
        if len(values) == 1 and res <= 1e-3:
            (name, x), = values.items()
            h = 1e-6 * max(1.0, abs(x))
            ends = [(_value(lhs, {name: x + d}, cm), _value(rhs, {name: x + d}, cm)) for d in (-h, h)]
            if all(a is not None and b is not None for a, b in ends):
                slope = ((ends[1][0] - ends[1][1]) - (ends[0][0] - ends[0][1])) / (2 * h)
                if slope != 0 and abs((left - right) / slope) <= 1e-12 * max(1.0, abs(x)):
                    return Check(PASS, method, "Ildizgacha masofa mashina aniqligida (Nyuton qadami).", res, tol)
        return Check(FAIL, method, f"Chap tomon {left:.10g}, o'ng tomon {right:.10g}.", res, tol)
    # Kichik qoldiq yetarli emas: exp(x)=0 yoki 1/x=0 kabi «soxta» ildizlarni Nyuton qadami bilan ushlaymiz.
    for name, x in values.items():
        h = 1e-6 * max(1.0, abs(x))
        around = []
        for d in (-h, h, -0.5, 0.5):
            env = dict(values, **{name: x + d})
            a, b = _value(lhs, env, cm), _value(rhs, env, cm)
            around.append(None if a is None or b is None else a - b)
        if all(v == 0 for v in around):
            return Check(UNKNOWN, method, f"Tenglama {name} atrofida ham nolga teng: ayniyat yoki sonli yo'qolish (underflow).", res, tol)
        if len(values) == 1 and around[0] is not None and around[1] is not None:
            slope = (around[1] - around[0]) / (2 * h)
            f0 = left - right
            if slope != 0 and abs(f0 / slope) > 1e-5 * max(1.0, abs(x)):
                return Check(FAIL, method, "Qoldiq kichik, lekin bu nuqta ildizga yaqin emas (funksiya nolga faqat intilmoqda).", res, tol)
    return Check(PASS, method, "Chap va o'ng tomon mos keldi (nisbiy qoldiq).", res, tol)


def numeric_derivative(f, x, order=1):
    """Richardson ekstrapolyatsiyali markaziy ayirma. (qiymat, xato bahosi) yoki None."""
    if order not in (1, 2, 3, 4):
        return None
    base = {1: 1e-3, 2: 4e-3, 3: 1e-2, 4: 2e-2}[order] * max(1.0, abs(x)) ** 0.5
    stencils = {1: ((-1, -0.5), (1, 0.5)), 2: ((-1, 1), (0, -2), (1, 1)),
                3: ((-2, -0.5), (-1, 1), (1, -1), (2, 0.5)), 4: ((-2, 1), (-1, -4), (0, 6), (1, -4), (2, 1))}
    table = []
    for i in range(4):
        h = base / 2 ** i
        vals = [f(x + k * h) for k, _ in stencils[order]]
        if any(v is None or not math.isfinite(v) for v in vals):
            return None
        row = [sum(c * v for (_, c), v in zip(stencils[order], vals)) / h ** order]
        for j in range(1, i + 1):
            row.append(row[j - 1] + (row[j - 1] - table[i - 1][j - 1]) / (4 ** j - 1))
        table.append(row)
    best, prev = table[-1][-1], table[-2][-1]
    return best, abs(best - prev)


def check_derivative(f_node, df_node, var, order=1, points=SAMPLE_POINTS, label="hosilani chekli ayirmalar bilan solishtirish"):
    """df_node haqiqatan f_node ning `order`-tartibli hosilasimi? Boshqa o'zgaruvchilarga sinov qiymati beriladi."""
    tol = {1: 1e-6, 2: 1e-5, 3: 1e-3, 4: 1e-2}.get(order)
    if tol is None:
        return Check(UNKNOWN, label, "4-tartibdan yuqori hosila sonli tekshirilmaydi.")
    others = sorted((variables(f_node) | variables(df_node)) - {var})
    env = {name: 0.7 + 0.45 * i for i, name in enumerate(others)}
    f = lambda x: _value(f_node, dict(env, **{var: x}))
    used, worst = 0, 0.0
    for x in points:
        claimed = _value(df_node, dict(env, **{var: x}))
        num = numeric_derivative(f, x, order)
        if claimed is None or num is None:
            continue
        value, err = num
        scale = max(1.0, abs(value), abs(claimed))
        if err / scale > tol:  # sonli hosilaning o'zi ishonchsiz nuqta
            continue
        diff = max(abs(value - claimed) - 10 * err, 0.0) / scale
        used += 1
        worst = max(worst, diff)
        if diff > tol:
            return Check(FAIL, label, f"{var} = {x} nuqtada: berilgan {claimed:.8g}, sonli {value:.8g}.", diff, tol)
    if used < 4:
        return Check(UNKNOWN, label, f"Tekshirishga yaroqli nuqta yetarli emas ({used} ta).")
    return Check(PASS, label, f"{used} ta sinov nuqtasida mos keldi.", worst, tol)


def check_antiderivative(integrand, antiderivative, var):
    return check_derivative(antiderivative, integrand, var, label="boshlang'ich funksiyani differensiallab tekshirish")


def quadrature(f, a, b):
    """scipy.quad: (qiymat, xato bahosi, ogohlantirishlar)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value, err = integrate.quad(f, a, b, limit=200, epsabs=1e-11, epsrel=1e-11)
    problems = [str(w.message).split("\n")[0] for w in caught if issubclass(w.category, integrate.IntegrationWarning)]
    return value, err, problems


def check_definite_integral(f_node, var, a, b, claimed, tol=1e-7):
    method = "integralni mustaqil sonli kvadratura bilan hisoblash"
    f = lambda x: float(evaluate(f_node, {var: x}))
    try:
        value, err, problems = quadrature(f, a, b)
    except (MathParseError, ValueError, ZeroDivisionError, OverflowError) as e:
        return Check(UNKNOWN, method, f"Kvadratura bajarilmadi: {e}")
    if problems or not math.isfinite(value) or err > 1e-6 * max(1.0, abs(value)):
        return Check(UNKNOWN, method, "Kvadratura ishonchli yaqinlashmadi (integral uzoqlashuvchi yoki maxsus nuqtali bo'lishi mumkin).")
    diff = abs(value - claimed) / max(1.0, abs(value))
    allowed = max(tol, 10 * err / max(1.0, abs(value)))
    status = PASS if diff <= allowed else FAIL
    return Check(status, method, f"Kvadratura natijasi {value:.12g} (xato bahosi {err:.1e}).", diff, allowed)


def check_limit(f_node, var, point, claimed, direction="both"):
    """Nuqtaga yaqinlashuvchi ketma-ketlikda f qiymatlari `claimed` ga intilishini tekshiradi."""
    method = "nuqtaga sonli yaqinlashish"
    sides = []
    if math.isinf(point):
        sides.append([math.copysign(10.0 ** k, point) for k in range(1, 8)])
    else:
        for sign, name in ((-1, "left"), (1, "right")):
            if direction in ("both", name):
                sides.append([point + sign * 10.0 ** -k * max(1.0, abs(point)) for k in range(1, 8)])
    for xs in sides:
        values = [_value(f_node, {var: x}) for x in xs]
        values = [v for v in values if v is not None]
        if len(values) < 4:
            return Check(UNKNOWN, method, "Nuqta atrofida funksiya qiymatlarini olib bo'lmadi.")
        if math.isinf(claimed):
            mags = [v * math.copysign(1, claimed) for v in values]
            steps = [b - a for a, b in zip(mags, mags[1:])]
            if all(s > 0 for s in steps) and mags[-1] > 5:
                if mags[-1] > 1e3:
                    continue
                return Check(UNKNOWN, method, "Funksiya o'smoqda, lekin juda sekin: cheksizlikni sonli tasdiqlab bo'lmaydi.")
            if mags[-1] < 0 or abs(steps[-1]) < 1e-6:
                return Check(FAIL, method, "Funksiya ko'rsatilgan cheksizlikka intilmayapti.")
            return Check(UNKNOWN, method, "Cheksiz limit sonli tasdiqlanmadi.")
        diffs = [abs(v - claimed) / max(1.0, abs(claimed)) for v in values]
        # uchta ketma-ket nuqta limit yaqinida va kamida bittasi juda yaqin bo'lsin
        if any(max(diffs[k:k + 3]) <= 1e-4 and min(diffs[k:k + 3]) <= 1e-6 for k in range(len(diffs) - 2)):
            continue
        if diffs[-1] < 0.7 * diffs[-3]:
            return Check(UNKNOWN, method, "Qiymatlar yaqinlashmoqda, lekin juda sekin.", diffs[-1], 1e-4)
        spread = max(values[-3:]) - min(values[-3:])
        if spread < 1e-3 * max(1.0, abs(values[-1])):
            return Check(FAIL, method, f"Qiymatlar {values[-1]:.8g} ga intilmoqda, berilgani {claimed:.8g}.", diffs[-1], 1e-4)
        return Check(UNKNOWN, method, "Qiymatlar barqaror yaqinlashmadi (tebranish yoki limit mavjud emas).")
    return Check(PASS, method, "Yaqinlashuvchi nuqtalarda qiymatlar limitga intildi.", None, 1e-4)


def _matrix_check(method, residual, scale, tol, detail=""):
    rel = float(residual) / max(float(scale), 1.0)
    return Check(PASS if rel <= tol else FAIL, method, detail, rel, tol)


def check_linear_solution(A, x, b, tol=1e-8):
    return _matrix_check("A·x = b qoldig'i", np.linalg.norm(A @ x - b),
                         np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(b), tol)


def check_inverse(A, inv, tol=1e-8):
    n = len(A)
    r = max(np.linalg.norm(A @ inv - np.eye(n)), np.linalg.norm(inv @ A - np.eye(n)))
    return _matrix_check("A·A⁻¹ = I va A⁻¹·A = I", r, np.linalg.norm(A) * np.linalg.norm(inv), tol)


def check_determinant(A, det, tol=1e-8):
    method = "determinantni xos qiymatlar ko'paytmasi bilan solishtirish"
    other = np.prod(np.linalg.eigvals(A))
    # Hadamard chegarasi: determinant uchun tabiiy masshtab
    bound = float(np.prod(np.linalg.norm(A, axis=1)))
    if not math.isfinite(bound):
        return Check(UNKNOWN, method, "Masshtab juda katta.")
    diff = abs(other - det) / max(abs(det), 1e-6 * bound, 1e-300)
    if diff <= tol:
        return Check(PASS, method, "", float(diff), tol)
    if np.linalg.cond(A) > 1e10:
        return Check(UNKNOWN, method, "Matritsa yomon shartlangan: sonli solishtirish ishonchsiz.", float(diff), tol)
    return Check(FAIL, method, f"Ikkinchi usul: {other:.10g}", float(diff), tol)


def check_eigen(A, values, vectors, tol=1e-7):
    residual = max(np.linalg.norm(A @ vectors[:, i] - values[i] * vectors[:, i]) for i in range(len(values)))
    checks = [_matrix_check("A·v = λ·v qoldig'i", residual, np.linalg.norm(A), tol),
              _matrix_check("iz(A) = Σλ", abs(np.trace(A) - np.sum(values)), np.linalg.norm(A), tol)]
    return checks


def check_minimum(f, x, bounds, maximize=False):
    """Topilgan nuqta atrofida (va tasodifiy nuqtalarda) yaxshiroq qiymat yo'qligini tekshiradi."""
    method = "atrofdagi va tasodifiy nuqtalar bilan solishtirish"
    sign = -1.0 if maximize else 1.0
    x = np.asarray(x, float)
    best = sign * f(x)
    rng = np.random.default_rng(12345)
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    candidates = [rng.uniform(lo, hi) for _ in range(4000)]
    for scale in (1e-2, 1e-4):
        candidates += [np.clip(x + scale * (hi - lo) * rng.standard_normal(len(x)), lo, hi) for _ in range(200)]
    tol = 1e-7 * max(1.0, abs(best))
    for c in candidates:
        v = sign * f(c)
        if math.isfinite(v) and v < best - tol:
            return Check(FAIL, method, f"Yaxshiroq nuqta topildi: {np.round(c, 6).tolist()}.", float(best - v), tol)
    return Check(PASS, method, f"{len(candidates)} ta nuqtada yaxshiroq qiymat topilmadi (global optimum isboti emas).", 0.0, tol)


def check_agreement(method, a, b, tol):
    """Ikki mustaqil sonli natijani solishtiradi."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return Check(UNKNOWN, method, "Natijalarni solishtirib bo'lmadi.")
    diff = float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(a)))) if a.size else 0.0
    return Check(PASS if diff <= tol else FAIL, method, "", diff, tol)


def check_inequality(lhs, op, rhs, var, intervals, lo, hi):
    """intervals: [(a, b, a_kiradi, b_kiradi)]. Zich sinov nuqtalarida tengsizlikni qayta hisoblaydi."""
    method = "tasodifiy sinov nuqtalarida tengsizlikni qayta hisoblash"
    rng = np.random.default_rng(2024)
    xs = np.concatenate([rng.uniform(lo, hi, 3000), np.linspace(lo, hi, 997)])
    diff = Node("sub", args=(lhs, rhs))
    values = np.broadcast_to(evaluate(diff, {var: xs}), xs.shape)
    cmp = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}[op]
    margin = 1e-7 * (hi - lo)
    wrong = 0
    for x, v in zip(xs, values):
        if any(abs(x - e) <= margin for i in intervals for e in i[:2]) or abs(v) < 1e-9:
            continue
        truth = bool(math.isfinite(v) and cmp(v, 0))
        claimed = any(a < x < b for a, b, _, _ in intervals)
        wrong += truth != claimed
    if wrong:
        return Check(FAIL, method, f"{wrong} ta sinov nuqtasida natija mos kelmadi.", wrong / len(xs), 0.0)
    return Check(PASS, method, f"[{lo:g}; {hi:g}] oralig'idagi sinov nuqtalarida mos keldi.", 0.0, 0.0)


def overall(checks):
    statuses = [c.status for c in checks]
    if FAIL in statuses:
        return FAIL
    if statuses and all(s == PASS for s in statuses):
        return PASS
    return UNKNOWN
