"""Mathematics AI testlari. Kutilgan qiymatlar qo'lda / analitik usulda olingan.

SymPy talab qiladigan testlar kutubxona bo'lmasa SKIP bo'ladi (o'tdi deb hisoblanmaydi).
"""
import json
import math
import random
import types
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np

import math_parser as mp
import math_symbolic as sym
import math_verifier as mv
import math_visualizer as viz
import matematika as m

SYMPY = sym.available()
PASS, FAIL, UNKNOWN = mv.PASS, mv.FAIL, mv.UNKNOWN


def val(text, **env):
    return float(mp.evaluate(mp.parse(text), env))


class ParserTest(unittest.TestCase):
    def test_amallar_va_ustunlik(self):
        for text, want in [("2+3*4", 14), ("-2^2", -4), ("(-2)^2", 4), ("2^3^2", 512), ("10/4", 2.5),
                           ("2(3+1)", 8), ("(1+1)(2+2)", 8), ("2pi", 2 * math.pi), ("sqrt(16)", 4), ("√(9)", 3),
                           ("3×4÷2", 6), ("ln(e^2)", 2), ("lg(1000)", 3), ("log(8, 2)", 3), ("tg(pi/4)", 1),
                           ("ctg(pi/4)", 1), ("factorial(5)", 120), ("abs(-3)", 3), ("2e3", 2000), ("007+1", 8),
                           ("5²", 25), ("cbrt(-8)", -2)]:
            with self.subTest(text=text):
                self.assertAlmostEqual(val(text), want, places=9)

    def test_ozgaruvchilar(self):
        self.assertAlmostEqual(val("2x^2 + 3x y", x=2, y=5), 38)
        self.assertAlmostEqual(val("x(x+1)", x=3), 12)
        self.assertAlmostEqual(val("2e", ), 2 * math.e)
        self.assertEqual(mp.variables(mp.parse("a*x + b1 - theta")), {"a", "x", "b1", "theta"})

    def test_xavfli_va_notogri_kirishlar_rad_etiladi(self):
        bad = ["__import__('os').system('ls')", "os.system(1)", "x.__class__", "().__class__", "[1,2]", "x[0]",
               "lambda: 1", "open('a')", "eval(1)", "exec(1)", "'abc'", "x if 1 else 2", "1 and 2", "x==1",
               "{1:2}", "f(x)=1", "xy", "import os", "True", "1;2", "x := 2", "sin", "sin x", "2 3", "1,5",
               "", "   ", "(", "x**", "9**9**9", "10^10^10", "factorial(100000)", "((x^100)^100)^100",
               "x^100000", "1e999", "exp(exp(100))", "(" * 500 + "1" + ")" * 500, "-" * 590 + "1",
               "+".join(["x"] * 280), "x" * 700, "sin(1, 2)", "log()", "5!", "x % 2", "x // 2", "@x", "#", "$"]
        for text in bad:
            with self.subTest(text=text[:40]):
                with self.assertRaises(mp.MathParseError):
                    mp.parse(text)

    def test_fuzz(self):
        """Tasodifiy matn: faqat MathParseError yoki to'g'ri Node bo'lishi kerak."""
        rng = random.Random(42)
        atoms = ["x", "y", "1", "2.5", "pi", "e", "sin", "log", "factorial", "gamma", "(", ")", "+", "-", "*", "/", "^",
                 "**", ",", ".", " ", "_", "[", "]", "'", "=", "!", "0", "9", "E", "__", "if", "or", "abs", "oo"]
        ok = 0
        for _ in range(6000):
            text = "".join(rng.choice(atoms) for _ in range(rng.randint(1, 25)))
            try:
                node = mp.parse(text)
                mp.evaluate(node, {v: 1.3 for v in mp.variables(node)})
                ok += 1
            except mp.MathParseError:
                pass
        self.assertGreater(ok, 50)

    def test_eval_ishlatilmagan(self):
        import ast
        import inspect
        for module in (mp, mv, m, sym, viz):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                    self.assertNotIn(name, ("eval", "exec", "sympify", "parse_expr", "lambdify"), module.__name__)

    def test_aniq_kasr(self):
        self.assertEqual(mp.exact_value(mp.parse("1/3 + 1/6")), Fraction(1, 2))
        self.assertEqual(mp.exact_value(mp.parse("0.1 + 0.2")), Fraction(3, 10))
        self.assertIsNone(mp.exact_value(mp.parse("sqrt(2)")))
        for text in ("1/(2-2)", "0^-1"):
            with self.assertRaises(mp.MathParseError):
                mp.exact_value(mp.parse(text))

    def test_matritsa_va_malumot(self):
        for text in ("1 2; 3 4", "1,2;3,4", "[[1,2],[3,4]]", "1 2\n3 4"):
            self.assertEqual([[float(x.value) for x in r] for r in mp.parse_matrix(text)], [[1, 2], [3, 4]])
        for text in ("1 2; 3", "x 1; 1 1", "[[1,2],[3,4]] + 1", ""):
            with self.assertRaises(mp.MathParseError):
                mp.parse_matrix(text)
        self.assertEqual(mp.parse_data("1, 2; 3\n4 1/2"), [1, 2, 3, 4, 0.5])

    def test_to_string_qayta_oqiladi(self):
        for text in ("-(x+1)^2/(x-3)", "x-(x-1)", "(x^2)^3", "2^-x", "(-x)^2", "x/(2x)", "log(x, 2)*sin(x)"):
            node = mp.parse(text)
            again = mp.parse(mp.to_string(node))
            self.assertAlmostEqual(float(mp.evaluate(node, {"x": 1.7})), float(mp.evaluate(again, {"x": 1.7})))


class VerifierTest(unittest.TestCase):
    """Tekshiruvchi to'g'ri javobni o'tkazishi va NOTO'G'RI javobni rad etishi kerak."""

    def root(self, eq, x):
        lhs, _, rhs = mp.parse_relation(eq)
        return mv.check_root(lhs, rhs, {"x": x}).status

    def test_ildiz(self):
        self.assertEqual(self.root("x^2 = 4", 2), PASS)
        self.assertEqual(self.root("x^2 = 4", 2.001), FAIL)
        self.assertEqual(self.root("x^2 + 1 = 0", 1j), PASS)
        self.assertEqual(self.root("x^2 - 1000000x + 1 = 0", (1e6 + math.sqrt(1e12 - 4)) / 2), PASS)

    def test_soxta_ildizlar(self):
        self.assertEqual(self.root("1/x = 0", 1e20), FAIL)
        self.assertEqual(self.root("1/x = 0", 1e-20), FAIL)
        self.assertNotEqual(self.root("exp(x) = 0", -40), PASS)
        self.assertNotEqual(self.root("exp(x) = 0", -1000), PASS)
        self.assertEqual(self.root("sqrt(x) = -1", 1), FAIL)
        self.assertEqual(self.root("log(x) = 0", -1), FAIL)

    def test_hosila_va_boshlangich(self):
        f = mp.parse("x^3*sin(x)")
        good, bad = mp.parse("3x^2*sin(x) + x^3*cos(x)"), mp.parse("3x^2*sin(x) - x^3*cos(x)")
        self.assertEqual(mv.check_derivative(f, good, "x").status, PASS)
        self.assertEqual(mv.check_derivative(f, bad, "x").status, FAIL)
        self.assertEqual(mv.check_derivative(f, mp.parse("6x*sin(x) + 6x^2*cos(x) - x^3*sin(x)"), "x", 2).status, PASS)
        self.assertEqual(mv.check_antiderivative(mp.parse("x*exp(x)"), mp.parse("(x-1)*exp(x)"), "x").status, PASS)
        self.assertEqual(mv.check_antiderivative(mp.parse("x*exp(x)"), mp.parse("x*exp(x)"), "x").status, FAIL)

    def test_aniq_integral(self):
        f = mp.parse("x^2")
        self.assertEqual(mv.check_definite_integral(f, "x", 0, 3, 9).status, PASS)
        self.assertEqual(mv.check_definite_integral(f, "x", 0, 3, 9.01).status, FAIL)
        self.assertEqual(mv.check_definite_integral(mp.parse("1/x"), "x", 0, 1, 5).status, UNKNOWN)

    def test_limit(self):
        f = mp.parse("sin(x)/x")
        self.assertEqual(mv.check_limit(f, "x", 0, 1).status, PASS)
        self.assertEqual(mv.check_limit(f, "x", 0, 0).status, FAIL)
        self.assertNotEqual(mv.check_limit(f, "x", 0, 1.001).status, PASS)
        self.assertNotEqual(mv.check_limit(mp.parse("sin(1/x)"), "x", 0, 0).status, PASS)
        self.assertEqual(mv.check_limit(mp.parse("(1+1/x)^x"), "x", math.inf, math.e).status, PASS)
        self.assertEqual(mv.check_limit(mp.parse("1/x^2"), "x", 0, math.inf).status, PASS)
        self.assertNotEqual(mv.check_limit(mp.parse("1/x"), "x", 0, math.inf).status, PASS)

    def test_matritsa(self):
        A = np.array([[2.0, 1], [1, 3]])
        self.assertEqual(mv.check_inverse(A, np.array([[0.6, -0.2], [-0.2, 0.4]])).status, PASS)
        self.assertEqual(mv.check_inverse(A, np.eye(2)).status, FAIL)
        self.assertEqual(mv.check_determinant(A, 5.0).status, PASS)
        self.assertEqual(mv.check_determinant(A, 6.0).status, FAIL)
        self.assertEqual(mv.check_linear_solution(A, np.array([1.0, 1]), np.array([3.0, 4])).status, PASS)
        self.assertEqual(mv.check_linear_solution(A, np.array([1.0, 2]), np.array([3.0, 4])).status, FAIL)

    def test_tengsizlik_va_minimum(self):
        lhs, op, rhs = mp.parse_relation("x^2 > 4")
        good = [(-10, -2, True, False), (2, 10, False, True)]
        self.assertEqual(mv.check_inequality(lhs, op, rhs, "x", good, -10, 10).status, PASS)
        self.assertEqual(mv.check_inequality(lhs, op, rhs, "x", good[:1], -10, 10).status, FAIL)
        f = lambda v: (v[0] - 1) ** 2
        self.assertEqual(mv.check_minimum(f, [1.0], [(-5, 5)]).status, PASS)
        self.assertEqual(mv.check_minimum(f, [3.0], [(-5, 5)]).status, FAIL)


class OperationTest(unittest.TestCase):
    def run_ok(self, name, status=PASS, **params):
        r = m.run_operation(name, params)
        self.assertNotIn("error", r, r)
        if status:
            self.assertEqual(mv.overall(r["checks"]), status, [c.as_dict() for c in r["checks"]])
        return r

    def err(self, name, **params):
        r = m.run_operation(name, params)
        self.assertIn("error", r, r)
        return r["error"]

    def test_arifmetika(self):
        self.assertEqual(self.run_ok("evaluate", expression="(3/4 + 1/6)^2")["exact"], "121/144")
        self.assertEqual(float(self.run_ok("evaluate", None, expression="x^2+y", values="x=3; y=1")["approx"]), 10)
        self.err("evaluate", expression="1/0")
        self.err("evaluate", expression="sqrt(-1)")
        self.err("evaluate", expression="x+1")

    def roots(self, **p):
        return self.run_ok("solve_equation", **p)["data"]["roots"]

    def test_tenglamalar(self):
        self.assertEqual(self.roots(equation="x^2 - 5x + 6 = 0"), ["2", "3"])
        self.assertEqual(self.roots(equation="(x-1)^3 = 0"), ["1"])
        self.assertEqual(self.roots(equation="(x-2)^2*(x+3) = 0"), ["-3", "2"])
        self.assertEqual(self.roots(equation="(x-1)*(x-1.001) = 0"), ["1", "1.001"])
        self.assertEqual(len(self.roots(equation="x^2 + 4 = 0")), 2)
        got = [float(x) for x in self.roots(equation="sin(x) = 1/2", lower="0", upper="pi")]
        np.testing.assert_allclose(got, [math.pi / 6, 5 * math.pi / 6])
        np.testing.assert_allclose([float(x) for x in self.roots(equation="2^x = 32")], [5])
        np.testing.assert_allclose([float(x) for x in self.roots(equation="log(x, 3) = 4", lower="0", upper="100")], [81])

    def test_yechimsiz_tenglama_soxta_ildiz_bermaydi(self):
        for eq in ("1/x = 0", "exp(x) = 0", "sqrt(x) = -2"):
            r = m.run_operation("solve_equation", {"equation": eq})
            self.assertEqual(r["data"]["roots"], [], eq)
            self.assertNotEqual(mv.overall(r["checks"]), PASS)

    def test_tengsizliklar(self):
        r = self.run_ok("inequality", inequality="x^2 - 4 <= 0")
        self.assertEqual(r["approx"], "[-2; 2]")
        r = self.run_ok("inequality", inequality="1/x > 0", lower="-10", upper="10")
        self.assertEqual(r["approx"], "(0; 10]")
        r = self.run_ok("inequality", inequality="(x-1)/(x+2) >= 0", lower="-10", upper="10")
        self.assertEqual(r["approx"], "[-10; -2) ∪ [1; 10]")
        r = self.run_ok("inequality", inequality="sqrt(x) < 2", lower="-10", upper="10")
        self.assertEqual(r["data"]["intervals"][0][1], 4)
        self.assertAlmostEqual(r["data"]["intervals"][0][0], 0, places=6)
        self.assertEqual(self.run_ok("inequality", inequality="x^2 <= 0")["data"]["points"], [0.0])

    def test_chegaraviy_holatlar(self):
        r = self.run_ok("inequality", inequality="tan(x) > 1", lower="0", upper="6")
        self.assertNotIn("]", r["approx"])  # qutb nuqtalari ochiq qavs bilan
        r = m.run_operation("solve_equation", {"equation": "sin(1000x) = 0"})
        self.assertEqual(mv.overall(r["checks"]), UNKNOWN)
        self.assertEqual(mv.overall(m.run_operation("inequality", {"inequality": "sin(1000x) > 0"})["checks"]), FAIL)
        r = m.run_operation("optimize", {"expression": "1/x", "bounds": "x: -1, 1"})
        self.assertNotEqual(mv.overall(r["checks"]), PASS)
        self.err("derivative", expression="abs(x)", at="0")
        r = m.run_operation("evaluate", {"expression": "(1+10^-20-1)*10^20"})
        self.assertEqual((r["exact"], r["approx"]), ("1", "1"))

    def test_tizim(self):
        r = self.run_ok("system", equations="2x + y = 5; x - y = 1")
        np.testing.assert_allclose(r["data"]["solutions"][0], [2, 1])
        r = self.run_ok("system", equations="x^2 + y^2 = 25; x - y = 1")
        self.assertEqual(sorted(tuple(np.round(s, 6)) for s in r["data"]["solutions"]), [(-3, -4), (4, 3)])
        self.assertIn("yechim yo'q", self.run_ok("system", equations="x+y=1; x+y=2")["approx"])
        self.assertIn("cheksiz", self.run_ok("system", equations="x+y=1; 2x+2y=2")["approx"])
        self.err("system", equations="x+y=1")

    def test_matritsalar(self):
        self.assertEqual(self.run_ok("matrix", matrix="1 2; 3 4", operation="det")["exact"], "-2")
        self.assertEqual(self.run_ok("matrix", matrix="2 1; 1 3", operation="inverse")["exact"], "[3/5 -1/5; -1/5 2/5]")
        self.assertEqual(self.run_ok("matrix", matrix="1 2; 2 4", operation="rank")["exact"], "1")
        r = self.run_ok("matrix", matrix="2 0; 0 3", operation="eigen")
        self.assertIn("λ1 = 2", r["approx"])
        r = self.run_ok("matrix", matrix="2 1; 1 3", operation="solve", other="3 4")
        self.assertEqual(r["exact"], "x1 = 1, x2 = 1")
        self.assertEqual(self.run_ok("matrix", matrix="1 2; 3 4", operation="multiply", other="0 1; 1 0")["approx"], "[2 1; 4 3]")
        self.assertEqual(self.run_ok("matrix", matrix="sqrt(2) 0; 0 sqrt(2)", operation="det")["approx"], "2")
        self.assertIn("xos", self.err("matrix", matrix="1 2; 2 4", operation="inverse"))
        self.err("matrix", matrix="1 2 3; 4 5 6", operation="det")
        self.assertEqual(self.run_ok("matrix", matrix="1 1; 1 1", operation="solve", other="1 2")["exact"], "yechim yo'q")

    def test_integral_sonli(self):
        r = m.run_operation("integral", {"expression": "x^2", "lower": "0", "upper": "3"})
        self.assertAlmostEqual(r["data"]["value"], 9)
        r = m.run_operation("integral", {"expression": "exp(-x^2)", "lower": "-inf", "upper": "inf"})
        self.assertAlmostEqual(r["data"]["value"], math.sqrt(math.pi))
        r = m.run_operation("integral", {"expression": "1/x", "lower": "0", "upper": "1"})
        self.assertNotEqual(mv.overall(r["checks"]), PASS)
        self.assertIsNone(r["approx"])

    def test_limit_va_hosila_nuqtada(self):
        r = m.run_operation("limit", {"expression": "(1 - cos(x))/x^2", "point": "0"})
        self.assertAlmostEqual(float(r["approx"] or r["exact"].replace("1/2", "0.5")), 0.5, places=4)
        self.err("limit", expression="abs(x)/x", point="0")
        r = m.run_operation("derivative", {"expression": "x^3", "at": "2"})
        self.assertAlmostEqual(float(r["approx"].split("≈")[1].split()[0]), 12, places=5)

    def test_gradient(self):
        r = m.run_operation("gradient", {"expression": "x^2*y", "point": "x=3; y=2"})
        np.testing.assert_allclose(r["data"]["gradient"], [12, 9], rtol=1e-7)

    def test_ode(self):
        r = self.run_ok("ode", rhs="-2y", t0="0", y0="1", t_end="2")
        self.assertAlmostEqual(r["data"]["y_end"], math.exp(-4), places=8)
        r = self.run_ok("ode", rhs="-y", order="2", y0="0", v0="1", t_end="pi/2")
        self.assertAlmostEqual(r["data"]["y_end"], 1, places=7)
        self.err("ode", rhs="y^2", y0="1", t_end="2")  # yechim t=1 da portlaydi

    def test_pde_analitik_yechim_bilan(self):
        r = self.run_ok("heat", initial="sin(pi*x)", alpha="1", length="1", t_end="0.1")
        x, u = np.array(r["data"]["x"]), np.array(r["data"]["u"])
        np.testing.assert_allclose(u, np.exp(-math.pi ** 2 * 0.1) * np.sin(math.pi * x), atol=1e-5)
        r = self.run_ok("heat", initial="x", left="0", right="1", t_end="0.2")
        np.testing.assert_allclose(r["data"]["u"], r["data"]["x"], atol=1e-8)

    def test_statistika(self):
        r = self.run_ok("statistics", data="2, 4, 4, 4, 5, 5, 7, 9")
        self.assertEqual(r["data"]["o'rtacha"], 5)
        self.assertAlmostEqual(r["data"]["dispersiya (bosh to'plam, n)"], 4)
        r = self.run_ok("statistics", data="1 2 3 4", data_y="3 5 7 9")
        self.assertAlmostEqual(r["data"]["regressiya: qiyalik b"], 2)
        self.assertAlmostEqual(r["data"]["regressiya: ozod had a"], 1)

    def test_ehtimollik(self):
        v = lambda **p: self.run_ok("probability", **p)["data"]["value"]
        self.assertAlmostEqual(v(distribution="binomial", n="10", p="0.5", k="3", kind="le"), 176 / 1024)
        self.assertAlmostEqual(v(distribution="binomial", n="10", p="0.5", k="3", kind="ge"), 1 - 56 / 1024)
        self.assertAlmostEqual(v(distribution="poisson", lam="2", k="0", kind="eq"), math.exp(-2))
        self.assertAlmostEqual(v(distribution="normal", k="1.96", kind="le"), 0.9750021, places=6)
        self.err("probability", distribution="binomial", n="10", p="2", k="3")

    def test_optimallashtirish(self):
        r = self.run_ok("optimize", expression="(x-1)^2 + (y+2)^2", bounds="x: -5, 5; y: -5, 5")
        np.testing.assert_allclose(r["data"]["x"], [1, -2], atol=1e-5)
        r = self.run_ok("optimize", expression="x*(10-x)", bounds="x: 0, 10", goal="max")
        self.assertAlmostEqual(r["data"]["value"], 25)

    def test_fourier(self):
        r = self.run_ok("fourier_series", expression="x", half_period="pi", terms="4")
        np.testing.assert_allclose(r["data"]["b"], [0, 2, -1, 2 / 3, -0.5], atol=1e-8)
        r = m.run_operation("fourier_transform", {"expression": "exp(-pi*x^2)", "frequency": "1"})
        self.assertIn(f"{math.exp(-math.pi):.6f}"[:7], r["approx"])

    def test_laplas_sonli(self):
        r = m.run_operation("laplace", {"expression": "t*exp(-t)", "s_value": "2"})
        self.assertAlmostEqual(float(r["approx"].split("≈")[1]), 1 / 9, places=8)

    def test_xatolar_istisno_tashlamaydi(self):
        self.err("nomalum")
        self.err("evaluate")
        self.err("evaluate", expression="1", hack="2")
        self.err("evaluate", expression=["1"])
        for name, (_, _, _, fields) in m.OPERATIONS.items():
            for junk in ("__import__('os')", "((", "x" * 30000, "-1", "inf", "0"):
                with self.subTest(op=name, junk=junk[:10]):
                    r = m.run_operation(name, {f["name"]: junk for f in fields})
                    self.assertIsInstance(r, dict)
                    m.format_result(r)


class VisualizerTest(unittest.TestCase):
    def test_png(self):
        for png in (viz.plot_function(["tan(x)", "1/x"]), viz.plot_contour("x^2+y^2"), viz.plot_histogram([1, 2, 2, 3]),
                    viz.plot_series([0, 1], {"y": [0, 1]})):
            self.assertTrue(png.startswith(b"\x89PNG"))

    def test_har_bir_amal_grafigi(self):
        for name, (_, _, _, fields) in m.OPERATIONS.items():
            r = m.run_operation(name, {f["name"]: f["default"] for f in fields})
            if r.get("plot"):
                with self.subTest(op=name):
                    self.assertTrue(viz.render(r["plot"]).startswith(b"\x89PNG"))

    def test_xato(self):
        with self.assertRaises(mp.MathParseError):
            viz.plot_function(["sqrt(-1-x^2)"])
        with self.assertRaises(mp.MathParseError):
            viz.plot_function(["x+y"])


def reply(content=None, calls=()):
    tool_calls = [types.SimpleNamespace(id=f"c{i}", function=types.SimpleNamespace(name="math_calculate", arguments=json.dumps(a)))
                  for i, a in enumerate(calls)]
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content, tool_calls=tool_calls or None))])


class FakeClient:
    """Groq o'rniga soxta mijoz: haqiqiy API chaqirilmaydi."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw)))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class AiLayerTest(unittest.TestCase):
    def test_asbob_orqali_javob_va_tekshiruv_bloki(self):
        c = FakeClient(reply(calls=[{"operation": "solve_equation", "params": {"equation": "x^2=9"}}]), reply("x = ±3"))
        text, plot = m.answer(c, "x^2=9", "School", "steps", model="t")
        self.assertIn("Dasturiy tekshiruv", text)
        self.assertIn("x ≈ -3", text)
        self.assertIsNotNone(plot)
        tool_msg = c.requests[1]["messages"][-1]
        self.assertEqual(json.loads(tool_msg["content"])["tekshiruv"], "PASS")

    def test_asbobsiz_javob_tekshirilmagan_deb_belgilanadi(self):
        text, _ = m.answer(FakeClient(reply("Javob 5")), "?", "School", "quick", model="t")
        self.assertIn("Dasturiy tekshiruv bajarilmadi", text)

    def test_api_xatosida_zaxira_va_limit(self):
        text, _ = m.answer(FakeClient(RuntimeError("boom"), reply("matn")), "?", "School", "quick", model="t")
        self.assertIn("asbobsiz", text)
        text, _ = m.answer(FakeClient(RuntimeError("429"), RuntimeError("429 rate")), "?", "School", "quick", model="t")
        self.assertIn("limiti", text)

    def test_buzuq_asbob_sorovi(self):
        bad = reply(calls=[{"operation": "evaluate", "params": {"expression": "__import__('os')"}}, {"operation": "zzz"}])
        text, _ = m.answer(FakeClient(bad, reply("bo'lmadi")), "?", "School", "quick", model="t")
        self.assertIn("bajarilmadi", text)

    def test_noaniq_rasm_yechilmaydi(self):
        c = FakeClient(reply('{"clear": false, "problem": "x^? = 4", "unclear": ["daraja xira"]}'))
        text, _ = m.answer(c, "yech", "School", "steps", image_urls=["data:x"], model="t", vision_model="v")
        self.assertIn("daraja xira", text)
        self.assertEqual(len(c.requests), 1)  # yechish bosqichiga o'tilmagan

    def test_buzuq_rasm_javobi_aniqlashtiradi(self):
        for raw in ("bilmadim", "{}", '{"clear": "false", "problem": "1+1"}', '{"clear": true, "problem": ""}'):
            c = FakeClient(reply(raw))
            text, _ = m.answer(c, "yech", "School", "steps", image_urls=["data:x"], model="t", vision_model="v")
            self.assertIn("Aniqlashtiring", text)
            self.assertEqual(len(c.requests), 1)

    def test_aniq_rasm(self):
        c = FakeClient(reply('<think>{x}</think>{"clear": true, "problem": "2+2", "unclear": []}'), reply("4"))
        text, _ = m.answer(c, "yech", "School", "quick", image_urls=["data:x"], model="t", vision_model="v")
        self.assertIn("Rasmdan o'qilgan masala", text)
        self.assertIn("2+2", c.requests[1]["messages"][-1]["content"])

    def test_rasm_cheklovlari(self):
        self.assertIn("topilmadi", m.answer(FakeClient(), "?", "School", "quick", image_urls=["a"], model="t")[0])
        self.assertIn("3 ta", m.answer(FakeClient(), "?", "School", "quick", image_urls=["a"] * 4, model="t", vision_model="v")[0])

    def test_rejim_va_daraja_promptda(self):
        self.assertIn("tezkor", m.system_prompt("School", "quick"))
        self.assertIn("bosqichma-bosqich", m.system_prompt("Advanced", "steps"))
        self.assertIn("Fourier", m.system_prompt("Advanced", "steps"))

    def test_model_tanlash(self):
        ids = ["whisper-x", "openai/gpt-oss-safeguard-20b", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        self.assertEqual(m.pick_model(ids), "openai/gpt-oss-120b")
        self.assertEqual(m.pick_model(ids, vision=True), "qwen/qwen3.8-27b")
        self.assertIsNone(m.pick_model(["openai/gpt-oss-120b"], vision=True))
        self.assertEqual(m.pick_model(ids, override="a"), "a")


class SymbolicFallbackTest(unittest.TestCase):
    def test_sympy_yoq_bolsa_tushunarli_xato(self):
        with mock.patch.object(sym, "available", return_value=False):
            self.assertIn("SymPy", sym.run("evaluate", expression="1")["error"])
            self.assertIn("SymPy", m.run_operation("simplify", {"expression": "x+x"})["error"])
            self.assertIn("SymPy", m.run_operation("integral", {"expression": "x"})["error"])


@unittest.skipUnless(SYMPY, "SymPy o'rnatilmagan")
class SymPyTest(unittest.TestCase):
    def ok(self, name, **p):
        r = m.run_operation(name, p)
        self.assertNotIn("error", r, r)
        self.assertIsNotNone(r["exact"], r)
        self.assertNotEqual(mv.overall(r["checks"]), FAIL, [c.as_dict() for c in r["checks"]])
        return r

    def verified(self, name, **p):
        r = self.ok(name, **p)
        self.assertEqual(mv.overall(r["checks"]), PASS, [c.as_dict() for c in r["checks"]])
        return r["exact"]

    def test_aniq_qiymat_va_soddalashtirish(self):
        self.assertEqual(self.verified("evaluate", expression="sqrt(8)"), "2*sqrt(2)")
        self.assertEqual(self.verified("evaluate", expression="cbrt(-27)"), "-3")
        self.verified("simplify", expression="(x^2 - 1)/(x - 1)")

    def test_tenglama(self):
        self.assertIn("sqrt(2)", self.verified("solve_equation", equation="x^2 = 2"))
        self.ok("system", equations="x + y = 5; x - y = 1")
        self.ok("inequality", inequality="x^2 - 4 > 0")

    def test_analiz(self):
        self.verified("derivative", expression="x^3*sin(x)")
        self.verified("derivative", expression="log(x, 2) + atan(x)", order="2")
        self.assertIn("+ C", self.verified("integral", expression="x*exp(x)"))
        self.assertEqual(self.verified("integral", expression="x*exp(-x)", lower="0", upper="inf"), "1")
        self.assertEqual(self.verified("limit", expression="sin(x)/x", point="0"), "1")
        self.assertEqual(self.verified("limit", expression="(1+1/x)^x", point="inf"), "E")
        self.ok("gradient", expression="x^2*y", point="x=1; y=2")

    def test_ode_va_transformatsiyalar(self):
        self.verified("ode", rhs="-2y", y0="1", t_end="2")
        self.verified("ode", rhs="-y", order="2", y0="0", v0="1", t_end="3")
        self.verified("laplace", expression="t*exp(-t)")
        self.verified("laplace", expression="1/(s+1)^2", direction="inverse")
        self.ok("fourier_transform", expression="exp(-pi*x^2)", frequency="1")


if __name__ == "__main__":
    unittest.main()
