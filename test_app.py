"""app.py integratsiya testlari.

1) StubAppTest: Streamlit va Groq o'rniga soxta modullar. Ulanish mantiqini tekshiradi
   (haqiqiy interfeys va haqiqiy API EMAS).
2) RealStreamlitTest: Streamlit o'rnatilgan bo'lsa, rasmiy AppTest bilan ilovani ishga tushiradi; aks holda SKIP.
Internetga chiqilmaydi, GROQ_API_KEY kerak emas.
"""
import importlib.util
import os
import runpy
import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas  # noqa: F401  (soxta modullar bilan ishga tushirishdan oldin yuklansin, qayta import qilinmasin)
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app.py")
sys.path.insert(0, HERE)

import hisob  # noqa: E402,F401
import matematika  # noqa: E402,F401
import pypdf  # noqa: E402,F401


class Rerun(Exception):
    pass


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class FakeSt:
    """Streamlit'ning app.py ishlatadigan qismi. clicks: bosiladigan tugmalar yozuvi."""

    def __init__(self, state, clicks=(), chat=None, submit=False):
        self.session_state, self.clicks, self.chat, self.submit = state, set(clicks), chat, submit
        self.secrets, self.query_params, self.log = {"GROQ_API_KEY": "test"}, {"u": "tester"}, []
        self.sidebar = self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cache_data(self, **kw):
        return lambda f: f

    def columns(self, n):
        return [self] * n

    def button(self, label, **kw):
        self.log.append(("button", label))
        return label in self.clicks

    def selectbox(self, label, options, index=0, **kw):
        return list(options)[index]

    radio = selectbox

    def text_input(self, label, value="", **kw):
        return value

    def number_input(self, label, **kw):
        return kw["value"]

    def checkbox(self, *a, **kw):
        return False

    def file_uploader(self, *a, **kw):
        return None

    def form_submit_button(self, *a, **kw):
        return self.submit

    def chat_input(self, *a, **kw):
        return self.chat

    def rerun(self):
        raise Rerun

    def stop(self):
        raise AssertionError("st.stop chaqirildi")

    def __getattr__(self, name):
        if name not in ("set_page_config", "markdown", "error", "metric", "expander", "spinner", "line_chart", "bar_chart",
                        "caption", "form", "chat_message", "download_button", "dataframe", "image"):
            raise AttributeError(name)

        def method(*a, **kw):
            self.log.append((name, a[0] if a else None))
            return self

        return method


class FakeGroq:
    requests = []

    def __init__(self, api_key):
        ids = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        self.models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(data=[types.SimpleNamespace(id=i) for i in ids]))
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        FakeGroq.requests.append(kw)
        message = types.SimpleNamespace(content="soxta javob", tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class StubAppTest(unittest.TestCase):
    def setUp(self):
        FakeGroq.requests = []
        self.state = State()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old = os.getcwd()
        os.chdir(tmp.name)  # suhbat fayllari vaqtinchalik papkaga yoziladi
        self.addCleanup(os.chdir, old)

    def run_app(self, modules=None, **kw):
        st = FakeSt(self.state, **kw)
        fake = {"streamlit": st, "groq": types.SimpleNamespace(Groq=FakeGroq), "ddgs": types.SimpleNamespace(DDGS=None)}
        fake.update(modules or {})
        with mock.patch.dict(sys.modules, fake), mock.patch.object(requests, "get", side_effect=OSError("tarmoqsiz")):
            try:
                runpy.run_path(APP)
            except Rerun:
                st.log.append(("rerun", None))
        return st

    def test_tort_rejim_tugmalari(self):
        labels = [v for k, v in self.run_app().log if k == "button"]
        for label in ("🌐 Umumiy", "📒 Buxgalteriya", "🔮 Prognoz", "📐 Matematika"):
            self.assertIn(label, labels)

    def test_eski_rejimlar_oz_asboblari_bilan_ishlaydi(self):
        import hisob
        for mode, marker in (("gen", "Iqtisodchi AI'san"), ("acc", "Bosh Buxgalter"), ("fc", "Prognoz AI")):
            with self.subTest(mode=mode):
                self.state.clear()
                self.state["mode"] = mode
                FakeGroq.requests = []
                self.run_app(chat="salom")
                req = FakeGroq.requests[0]
                self.assertIn(marker, req["messages"][0]["content"])
                self.assertIs(req["tools"].__class__, list)
                self.assertEqual([t["function"]["name"] for t in req["tools"]], [t["function"]["name"] for t in hisob.TOOLS])
                chat = self.state["chats"][self.state["cur"]]
                self.assertEqual(chat["mode"], mode)
                self.assertEqual([x["content"] for x in chat["messages"]], ["salom", "soxta javob"])

    def test_matematika_rejimi_va_tarix(self):
        self.state["mode"] = "gen"
        self.run_app(chat="salom")
        first = self.state["cur"]
        self.run_app(clicks=["📐 Matematika"])
        self.assertEqual((self.state["mode"], self.state["cur"]), ("math", None))
        self.run_app(chat="2+2")
        req = FakeGroq.requests[-1]
        self.assertEqual(req["tools"][0]["function"]["name"], "math_calculate")
        self.assertEqual(req["model"], "openai/gpt-oss-120b")
        self.assertEqual(len(self.state["chats"]), 2)
        self.assertIn(first, self.state["chats"])
        self.assertIn("Dasturiy tekshiruv bajarilmadi", self.state["chats"][self.state["cur"]]["messages"][1]["content"])
        self.run_app(clicks=["💬 salom"])  # eski suhbatga qaytish
        self.assertEqual((self.state["mode"], self.state["cur"]), ("gen", first))

    def test_matematika_kalkulyatori(self):
        self.state["mode"] = "math"
        st = self.run_app(submit=True)
        self.assertTrue(any(k == "markdown" and "Ifoda qiymati" in str(v) for k, v in st.log))

    def test_tezkor_hisoblagichlar(self):
        for form, text in (("kredit", "Annuitet bo'yicha oylik to'lov"), ("maosh", "Qo'lga tegadigan maosh")):
            with self.subTest(form=form):
                self.state.clear()
                self.state["form"] = form
                self.run_app(submit=True)
                self.assertIn(text, self.state["chats"][self.state["cur"]]["messages"][1]["content"])

    def test_matematika_yuklanmasa_boshqa_rejimlar_ishlaydi(self):
        broken = {"matematika": None}  # import qilinganda ImportError
        self.run_app(modules=broken, chat="salom")
        self.assertEqual(self.state["chats"][self.state["cur"]]["messages"][1]["content"], "soxta javob")
        self.state.clear()
        self.state["mode"] = "math"
        st = self.run_app(modules=broken)
        self.assertTrue(any(k == "error" and "Mathematics AI yuklanmadi" in str(v) for k, v in st.log))


REAL = all(importlib.util.find_spec(n) for n in ("streamlit", "groq", "pypdf"))


@unittest.skipUnless(REAL, "Streamlit/Groq o'rnatilmagan")
class RealStreamlitTest(unittest.TestCase):
    def test_ilova_ochiladi_va_matematika_ishlaydi(self):
        from streamlit.testing.v1 import AppTest
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, old)
        with mock.patch.object(requests, "get", side_effect=OSError("tarmoqsiz")), \
                mock.patch("groq.Groq", FakeGroq), mock.patch.dict(sys.modules, {"ddgs": types.SimpleNamespace(DDGS=None)}):
            at = AppTest.from_file(APP, default_timeout=60)
            at.secrets["GROQ_API_KEY"] = "test"
            at.run()
            self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
            at.button(key="m_math").click().run()
            self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
            next(b for b in at.button if b.label == "Hisoblash").click().run()
            self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
            self.assertTrue(any("Ifoda qiymati" in x.value for x in at.markdown))


if __name__ == "__main__":
    unittest.main()
