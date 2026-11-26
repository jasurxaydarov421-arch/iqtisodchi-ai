"""Grafiklar (Matplotlib). Streamlit'ga bog'liq emas: har bir funksiya PNG baytlarini qaytaradi."""
import io

import numpy as np
from matplotlib.figure import Figure  # pyplot ishlatilmaydi: global holat va GUI yo'q

from math_parser import MathParseError, evaluate, parse, variables


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    return buf.getvalue()


def _axes(title=""):
    fig = Figure(figsize=(7, 4.2))
    ax = fig.subplots()
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    return fig, ax


def _break_jumps(y):
    """Uzilish nuqtalarida (masalan tan(x), 1/x) soxta vertikal chiziqlarni olib tashlaydi."""
    y = np.array(y, dtype=float)
    y[~np.isfinite(y)] = np.nan
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        raise MathParseError("Bu oraliqda funksiya aniqlanmagan.")
    lo, hi = np.percentile(finite, [3, 97])
    span = max(hi - lo, 1e-9)
    jumps = np.where((np.abs(np.diff(y)) > span) & (np.sign(y[1:]) != np.sign(y[:-1])))[0]
    y[jumps + 1] = np.nan
    limits = None
    if finite.max() - finite.min() > 6 * span:
        limits = (lo - span, hi + span)
    return y, limits


def plot_function(expressions, var="x", lo=-10.0, hi=10.0, points=(), labels=None):
    """Bir yoki bir nechta funksiya grafigi. points: belgilab ko'rsatiladigan (x, y) nuqtalar."""
    if not lo < hi:
        raise MathParseError("Grafik oralig'i noto'g'ri.")
    fig, ax = _axes()
    xs = np.linspace(lo, hi, 2000)
    all_limits = []
    for i, text in enumerate(expressions):
        node = parse(text) if isinstance(text, str) else text
        extra = variables(node) - {var}
        if extra:
            raise MathParseError(f"Grafik uchun faqat bitta o'zgaruvchi bo'lishi kerak ({', '.join(sorted(extra))} ortiqcha).")
        y, limits = _break_jumps(np.broadcast_to(evaluate(node, {var: xs}), xs.shape))
        ax.plot(xs, y, label=(labels or [str(t) for t in expressions])[i])
        if limits:
            all_limits.append(limits)
    for px, py in points:
        if lo <= px <= hi:
            ax.plot(px, py, "o", color="#FF3B22")
    if all_limits:
        ax.set_ylim(min(a for a, _ in all_limits), max(b for _, b in all_limits))
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel(var)
    ax.legend(fontsize=8)
    return _png(fig)


def plot_series(x, series, xlabel="t", title=""):
    """series: {nom: qiymatlar}."""
    fig, ax = _axes(title)
    for name, y in series.items():
        ax.plot(x, y, label=name)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)
    return _png(fig)


def plot_contour(expression, var1="x", var2="y", bounds=((-5, 5), (-5, 5)), point=None):
    node = parse(expression) if isinstance(expression, str) else expression
    xs, ys = np.linspace(*bounds[0], 200), np.linspace(*bounds[1], 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array(np.broadcast_to(evaluate(node, {var1: X, var2: Y}), X.shape), dtype=float)
    Z[~np.isfinite(Z)] = np.nan
    if not np.isfinite(Z).any():
        raise MathParseError("Bu sohada funksiya aniqlanmagan.")
    fig, ax = _axes()
    cs = ax.contourf(X, Y, Z, levels=25)
    fig.colorbar(cs, ax=ax)
    if point is not None:
        ax.plot(*point, "o", color="#FF3B22")
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    return _png(fig)


def plot_histogram(data):
    fig, ax = _axes("Gistogramma")
    ax.hist(data, bins="auto", color="#16354D", edgecolor="white")
    return _png(fig)


def render(spec):
    """matematika.py natijasidagi grafik tavsifidan PNG yasaydi."""
    kind = spec["kind"]
    args = {k: v for k, v in spec.items() if k != "kind"}
    return {"function": plot_function, "series": plot_series, "contour": plot_contour,
            "histogram": plot_histogram}[kind](**args)
