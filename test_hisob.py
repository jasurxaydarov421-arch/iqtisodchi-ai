"""hisob.py uchun avtomatik testlar.

Testlar unittest uslubida yozilgan: ham `pytest -v`, ham `python -m unittest discover -s tests -v` bilan ishlaydi.
Kutilgan qiymatlar kod formulasidan emas, qo'lda yoki mustaqil usulda (masalan bisektsiya) hisoblangan.
"""
import inspect
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hisob  # noqa: E402
from hisob import HisobXatosi  # noqa: E402

RATES = {"USD": "12000", "EUR": "13000", "RUB": "135.2"}
STAVKALAR = {"JSHDS": 12.0, "INPS": 4.0, "Ijtimoiy soliq": 12.0, "QQS": 12.0}  # testlarni stavka o'zgarishidan ajratish uchun


def tolov_bisektsiya(summa, yillik_foiz, oy):
    """Annuitet to'lovini formulasiz topadi: qoldiq oxirida 0 bo'ladigan to'lovni bisektsiya bilan qidiradi."""
    r = yillik_foiz / 1200
    lo, hi = 0.0, summa * 2
    for _ in range(200):
        t = (lo + hi) / 2
        q = summa
        for _ in range(oy):
            q = q * (1 + r) - t
        lo, hi = (t, hi) if q > 0 else (lo, t)
    return t


class KreditTest(unittest.TestCase):
    def test_annuitet_mustaqil_usul_bilan_mos(self):
        for s, p, n in [(100_000_000, 24, 36), (5_000_000, 18.5, 12), (250_000_000, 30, 120), (1_000_000, 1, 6)]:
            with self.subTest(s=s, p=p, n=n):
                r = hisob.annuitet_kredit(s, p, n)
                self.assertAlmostEqual(r["oylik_tolov"], tolov_bisektsiya(s, p, n), delta=1)
                self.assertAlmostEqual(r["jami_tolov"], tolov_bisektsiya(s, p, n) * n, delta=n)

    def test_annuitet_nol_foiz(self):
        r = hisob.annuitet_kredit(1_200_000, 0, 12)
        self.assertEqual((r["oylik_tolov"], r["jami_tolov"], r["jami_foiz"]), (100_000, 1_200_000, 0))

    def test_bir_oylik_kredit_ikkala_usulda_bir_xil(self):
        # 1 oyda to'lov = S·(1 + r): 12 000 000 · 1,01 = 12 120 000
        self.assertEqual(hisob.annuitet_kredit(12_000_000, 12, 1)["oylik_tolov"], 12_120_000)
        self.assertEqual(hisob.differensial_kredit(12_000_000, 12, 1)["birinchi_oy_tolov"], 12_120_000)

    def test_differensial_qolda_hisoblangan(self):
        # S=12 mln, 12% yillik (1% oylik), 12 oy. 1-oy: 1 000 000 + 120 000; oxirgi oy: 1 000 000 + 10 000.
        # Jami foiz = S·r·(n+1)/2 = 12 000 000 · 0,01 · 6,5 = 780 000
        r = hisob.differensial_kredit(12_000_000, 12, 12)
        self.assertEqual(r["birinchi_oy_tolov"], 1_120_000)
        self.assertEqual(r["oxirgi_oy_tolov"], 1_010_000)
        self.assertEqual(r["jami_foiz"], 780_000)
        self.assertEqual(r["jami_tolov"], 12_780_000)

    def test_jadval_oxirida_qoldiq_nol_va_asosiy_qarz_yigindisi(self):
        for fn in (hisob._ann, hisob._diff):
            for s, p, n in [(100_000_000, 24, 36), (7_777_777, 13.3, 17), (3_000_000, 0, 5)]:
                with self.subTest(fn=fn.__name__, s=s, p=p, n=n):
                    _, rows = fn(s, p, n)
                    self.assertEqual(len(rows), n)
                    self.assertLessEqual(rows[-1]["Qoldiq"], 1)
                    self.assertAlmostEqual(sum(x["Asosiy qarz"] for x in rows), s, delta=n)

    def test_differensialda_foiz_annuitetdan_kam(self):
        for s, p, n in [(100_000_000, 24, 36), (10_000_000, 15, 6)]:
            self.assertLess(hisob.differensial_kredit(s, p, n)["jami_foiz"], hisob.annuitet_kredit(s, p, n)["jami_foiz"])

    def test_qabul_qilinadigan_korinishlar(self):
        base = hisob.annuitet_kredit(100_000_000, 24, 36)["oylik_tolov"]
        for oy in ("36", 36.0, " 36 "):
            self.assertEqual(hisob.annuitet_kredit("100 000 000", "24", oy)["oylik_tolov"], base)
        self.assertEqual(hisob.annuitet_kredit(1e8, "24,0", 36)["oylik_tolov"], base)

    def test_notogri_qiymatlar_rad_etiladi(self):
        yaxshi = dict(summa=100_000_000, yillik_foiz=24, oy=36)
        yomon = [("summa", v) for v in (0, -1, "abc", None, float("nan"), float("inf"), True)] \
            + [("yillik_foiz", v) for v in (-1, 1001, "x", None)] \
            + [("oy", v) for v in (0, -3, 12.5, 601, "x", None)]
        for fn in (hisob.annuitet_kredit, hisob.differensial_kredit):
            for nom, qiymat in yomon:
                with self.subTest(fn=fn.__name__, nom=nom, qiymat=qiymat):
                    with self.assertRaises(HisobXatosi):
                        fn(**{**yaxshi, nom: qiymat})


class IshHaqiTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(hisob.SOLIQ, STAVKALAR)
        p.start()
        self.addCleanup(p.stop)

    def test_qolda_hisoblangan(self):
        # 10 000 000: JSHDS 12% = 1 200 000; INPS 4% = 400 000; qo'lga = 8 400 000; ijtimoiy soliq 12% = 1 200 000
        r = hisob.ish_haqi(10_000_000)
        self.assertEqual(r["JSHDS"], 1_200_000)
        self.assertEqual(r["INPS"], 400_000)
        self.assertEqual(r["qolga_tegadi"], 8_400_000)
        self.assertEqual(r["ijtimoiy_soliq_ish_beruvchi"], 1_200_000)
        self.assertEqual(r["ish_beruvchi_jami_xarajat"], 11_200_000)

    def test_nol_maosh(self):
        r = hisob.ish_haqi(0)
        self.assertEqual((r["brutto"], r["JSHDS"], r["INPS"], r["qolga_tegadi"], r["ish_beruvchi_jami_xarajat"]), (0, 0, 0, 0, 0))

    def test_stavka_ozgarsa_natija_ozgaradi(self):
        hisob.SOLIQ["JSHDS"] = 10.0
        self.assertEqual(hisob.ish_haqi(10_000_000)["JSHDS"], 1_000_000)

    def test_notogri(self):
        for v in (-1, -5_000_000, "abc", None, float("nan")):
            with self.subTest(v=v):
                with self.assertRaises(HisobXatosi):
                    hisob.ish_haqi(v)


class QQSTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(hisob.SOLIQ, STAVKALAR)
        p.start()
        self.addCleanup(p.stop)

    def test_qqs_qoshish(self):
        r = hisob.qqs(1_000_000)
        self.assertEqual((r["stavka"], r["qqs"], r["qqs_bilan"], r["qqssiz"]), (12.0, 120_000, 1_120_000, 1_000_000))

    def test_qqs_ajratib_olish(self):
        r = hisob.qqs(1_120_000, qqs_bilan=True)
        self.assertEqual((r["qqssiz"], r["qqs"], r["qqs_bilan"]), (1_000_000, 120_000, 1_120_000))

    def test_nol_stavka_haqiqiy_nol_foiz(self):
        # Eski xato: stavka=0 berilsa 12% hisoblanardi
        for stavka in (0, 0.0, "0"):
            with self.subTest(stavka=stavka):
                r = hisob.qqs(1_000_000, stavka=stavka)
                self.assertEqual((r["stavka"], r["qqs"], r["qqssiz"], r["qqs_bilan"]), (0.0, 0, 1_000_000, 1_000_000))
                r = hisob.qqs(1_000_000, qqs_bilan=True, stavka=stavka)
                self.assertEqual((r["qqs"], r["qqssiz"]), (0, 1_000_000))

    def test_stavka_berilmasa_standart(self):
        for stavka in (None, ""):
            self.assertEqual(hisob.qqs(1_000_000, stavka=stavka)["stavka"], 12.0)

    def test_boshqa_stavka(self):
        r = hisob.qqs(2_000_000, stavka=15)
        self.assertEqual((r["qqs"], r["qqs_bilan"]), (300_000, 2_300_000))

    def test_qqs_bilan_qiymatlari(self):
        for v in (True, "true", "True", "ha", "yes", 1, "1"):
            self.assertEqual(hisob.qqs(1_120_000, qqs_bilan=v)["qqs"], 120_000, v)
        for v in (False, "false", "no", 0, "0", "yo'q"):
            self.assertEqual(hisob.qqs(1_000_000, qqs_bilan=v)["qqs"], 120_000, v)

    def test_aylanma_tekshiruvi(self):
        for s in (1000, 987_654.32, 12_345_678):
            r = hisob.qqs(s, qqs_bilan=True)
            self.assertAlmostEqual(r["qqssiz"] + r["qqs"], s, delta=1)

    def test_notogri(self):
        for kw in (dict(summa=-1), dict(summa="abc"), dict(summa=1, stavka=-1), dict(summa=1, stavka=101),
                   dict(summa=1, stavka="x"), dict(summa=1, qqs_bilan="balki")):
            with self.subTest(kw=kw):
                with self.assertRaises(HisobXatosi):
                    hisob.qqs(**kw)


class MurakkabFoizTest(unittest.TestCase):
    def test_qolda_hisoblangan(self):
        self.assertEqual(hisob.murakkab_foiz(1_000_000, 12, 1, 12)["yakuniy_summa"], 1_126_825)  # 1,01^12
        self.assertEqual(hisob.murakkab_foiz(1_000_000, 12, 1, 1)["yakuniy_summa"], 1_120_000)
        self.assertEqual(hisob.murakkab_foiz(1_000_000, 12, 2, 1)["yakuniy_summa"], 1_254_400)  # 1,12^2
        r = hisob.murakkab_foiz(1_000_000, 0, 5)
        self.assertEqual((r["yakuniy_summa"], r["foyda"]), (1_000_000, 0))

    def test_notogri(self):
        yaxshi = dict(summa=1_000_000, yillik_foiz=18, yil=3, yiliga_marta=12)
        for nom, v in [("summa", 0), ("summa", -1), ("yillik_foiz", -1), ("yil", 0), ("yil", -3), ("yil", 101),
                       ("yiliga_marta", 0), ("yiliga_marta", 2.5), ("yiliga_marta", 366), ("summa", "x")]:
            with self.subTest(nom=nom, v=v):
                with self.assertRaises(HisobXatosi):
                    hisob.murakkab_foiz(**{**yaxshi, nom: v})

    def test_juda_katta_natija_xatosi(self):
        with self.assertRaises(HisobXatosi):
            hisob.murakkab_foiz(1_000_000, 1000, 100, 365)


class BiznesTahlilTest(unittest.TestCase):
    def test_qolda_hisoblangan(self):
        # marja = 100−60 = 40 (40%); zararsizlik = 4000/40 = 100 dona; daromad 20 000; foyda = 40·200−4000 = 4000 (20%)
        r = hisob.biznes_tahlil(100, 60, 4000, 200)
        self.assertEqual((r["birlik_marjasi"], r["marja_foizi"], r["zararsizlik_soni"]), (40, 40.0, 100))
        self.assertEqual((r["umumiy_daromad"], r["sof_foyda"], r["rentabellik_foizi"]), (20_000, 4000, 20.0))

    def test_zararsizlik_yuqoriga_yaxlitlanadi(self):
        self.assertEqual(hisob.biznes_tahlil(100, 60, 4001, 10)["zararsizlik_soni"], 101)

    def test_nol_yoki_manfiy_marja(self):
        for var in (100, 120):
            r = hisob.biznes_tahlil(100, var, 1000, 10)
            self.assertIsNone(r["zararsizlik_soni"])
            self.assertIn("zararsizlik nuqtasiga yetib bo'lmaydi", r["izoh"])

    def test_sotuv_nol(self):
        r = hisob.biznes_tahlil(100, 60, 4000, 0)
        self.assertEqual((r["sof_foyda"], r["umumiy_daromad"]), (-4000, 0))
        self.assertIsNone(r["rentabellik_foizi"])

    def test_notogri(self):
        yaxshi = dict(narx=100, ozgaruvchan_xarajat=60, doimiy_xarajat=4000, sotuv_soni=200)
        for nom, v in [("narx", 0), ("narx", -5), ("ozgaruvchan_xarajat", -1), ("doimiy_xarajat", -1), ("sotuv_soni", -1), ("narx", "x")]:
            with self.subTest(nom=nom, v=v):
                with self.assertRaises(HisobXatosi):
                    hisob.biznes_tahlil(**{**yaxshi, nom: v})


class ValyutaTest(unittest.TestCase):
    def setUp(self):
        hisob.set_rates(dict(RATES))
        self.addCleanup(hisob.set_rates, {})

    def test_qolda_hisoblangan(self):
        self.assertEqual(hisob.valyuta_aylantirish(100, "USD", "UZS")["natija"], 1_200_000.0)
        self.assertEqual(hisob.valyuta_aylantirish(1_200_000, "UZS", "USD")["natija"], 100.0)
        self.assertEqual(hisob.valyuta_aylantirish(100, "usd", "eur")["natija"], round(100 * 12000 / 13000, 2))
        self.assertEqual(hisob.valyuta_aylantirish(100, "USD", "USD")["natija"], 100.0)
        self.assertEqual(hisob.valyuta_aylantirish(0, "USD", "UZS")["natija"], 0.0)

    def test_kurs_mavjud_emas(self):
        hisob.set_rates({"USD": "-", "EUR": "-", "RUB": "-"})
        with self.assertRaisesRegex(HisobXatosi, "CBU kursi hozir mavjud emas"):
            hisob.valyuta_aylantirish(100, "USD", "UZS")
        hisob.set_rates({})
        with self.assertRaises(HisobXatosi):
            hisob.valyuta_aylantirish(100, "EUR", "UZS")
        self.assertEqual(hisob.valyuta_aylantirish(100, "UZS", "UZS")["natija"], 100.0)  # UZS kursiga ehtiyoj yo'q

    def test_notogri(self):
        for kw in (dict(summa=100, kimdan="GBP", kimga="UZS"), dict(summa=100, kimdan="USD", kimga=""),
                   dict(summa=-1, kimdan="USD", kimga="UZS"), dict(summa="x", kimdan="USD", kimga="UZS")):
            with self.subTest(kw=kw):
                with self.assertRaises(HisobXatosi):
                    hisob.valyuta_aylantirish(**kw)


class PrognozTest(unittest.TestCase):
    def test_toza_chiziqli_qator(self):
        r = hisob.prognoz([100, 110, 120, 130], 3)
        self.assertEqual(r["chiziqli_trend"], [140.0, 150.0, 160.0])
        self.assertEqual(r["taxminiy_past_chegara"], r["taxminiy_yuqori_chegara"])  # qoldiq 0 → oraliq nol kenglikda
        g = (130 / 100) ** (1 / 3) - 1
        self.assertEqual(r["davrlik_ortacha_osish_foizi"], round(g * 100, 2))
        self.assertEqual(r["osish_bo'yicha_prognoz"][0], round(130 * (1 + g), 2))

    def test_shovqinli_qatorda_oraliq(self):
        r = hisob.prognoz([12000, 12100, 12250, 12300, 12400, 12390, 12480], 4)
        self.assertEqual(len(r["chiziqli_trend"]), 4)
        for past, trend, yuqori in zip(r["taxminiy_past_chegara"], r["chiziqli_trend"], r["taxminiy_yuqori_chegara"]):
            self.assertLess(past, trend)
            self.assertLess(trend, yuqori)

    def test_doimiy_va_nolli_qator(self):
        self.assertEqual(hisob.prognoz([5, 5, 5], 2)["chiziqli_trend"], [5.0, 5.0])
        self.assertIsNone(hisob.prognoz([0, 1, 2, 3])["davrlik_ortacha_osish_foizi"])  # 0 dan CAGR hisoblanmaydi

    def test_davr_soni_chegaralari(self):
        self.assertEqual(len(hisob.prognoz([1, 2, 3], 20)["chiziqli_trend"]), 12)
        self.assertEqual(len(hisob.prognoz([1, 2, 3], 0)["chiziqli_trend"]), 1)

    def test_notogri(self):
        for q in ([1, 2], [], "123", None, 5, [1, "x", 3], [1, float("nan"), 3], [1, None, 3]):
            with self.subTest(q=q):
                with self.assertRaises(HisobXatosi):
                    hisob.prognoz(q)


class AsboblarTest(unittest.TestCase):
    """call_tool va TOOLS (AI ishlatadigan qatlam)."""

    def setUp(self):
        hisob.set_rates(dict(RATES))
        self.addCleanup(hisob.set_rates, {})

    def test_xato_json_korinishida_qaytadi(self):
        for nom, args in [("annuitet_kredit", {"summa": -1, "yillik_foiz": 24, "oy": 12}), ("nomalum_asbob", {}),
                          ("ish_haqi", {"brutto": 1, "ortiqcha": 2}), ("qqs", {})]:
            with self.subTest(nom=nom):
                out = json.loads(hisob.call_tool(nom, json.dumps(args)))
                self.assertIn("xato", out)
        self.assertIn("xato", json.loads(hisob.call_tool("ish_haqi", "{buzuq json")))

    def test_xato_xabari_ozbekcha_harflar_bilan(self):
        raw = hisob.call_tool("annuitet_kredit", json.dumps({"summa": 1, "yillik_foiz": 24, "oy": 0}))
        self.assertNotIn("\\u", raw)

    def test_tools_va_funksiyalar_mos(self):
        nomlar = {t["function"]["name"] for t in hisob.TOOLS}
        self.assertEqual(nomlar, set(hisob.FUNCS))
        for t in hisob.TOOLS:
            f = t["function"]
            params = inspect.signature(hisob.FUNCS[f["name"]]).parameters
            for nom in f["parameters"]["properties"]:
                self.assertIn(nom, params, f"{f['name']}: sxemadagi '{nom}' funksiyada yo'q")
            for nom in f["parameters"]["required"]:
                self.assertIn(nom, f["parameters"]["properties"])
            majburiy = {n for n, p in params.items() if p.default is inspect.Parameter.empty}
            self.assertLessEqual(majburiy, set(f["parameters"]["required"]), f"{f['name']}: majburiy parametr sxemada ko'rsatilmagan")

    def test_har_bir_natijada_formula_va_izoh(self):
        chaqiruvlar = [("annuitet_kredit", dict(summa=1e8, yillik_foiz=24, oy=36)), ("differensial_kredit", dict(summa=1e8, yillik_foiz=24, oy=36)),
                       ("ish_haqi", dict(brutto=8e6)), ("qqs", dict(summa=1e6)), ("qqs", dict(summa=1e6, qqs_bilan=True)),
                       ("murakkab_foiz", dict(summa=1e6, yillik_foiz=18, yil=3)),
                       ("biznes_tahlil", dict(narx=100, ozgaruvchan_xarajat=60, doimiy_xarajat=4000, sotuv_soni=200)),
                       ("valyuta_aylantirish", dict(summa=100, kimdan="USD", kimga="UZS")), ("prognoz", dict(qiymatlar=[1, 2, 3, 4]))]
        for nom, args in chaqiruvlar:
            with self.subTest(nom=nom):
                out = json.loads(hisob.call_tool(nom, json.dumps(args)))
                self.assertTrue(out.get("formula"), "formula yo'q")
                self.assertTrue(out.get("izoh"), "izoh yo'q")
        self.assertTrue(json.loads(hisob.call_tool("soliq_stavkalari", "{}")).get("izoh"))

    def test_soliq_stavkalari_asl_lugatni_buzmaydi(self):
        before = dict(hisob.SOLIQ)
        r = hisob.soliq_stavkalari()
        r["JSHDS"] = 99
        self.assertEqual(hisob.SOLIQ, before)

    def test_set_rates_valyutaga_tasir_qiladi(self):
        hisob.set_rates({"USD": "10000"})
        self.assertEqual(hisob.valyuta_aylantirish(1, "USD", "UZS")["natija"], 10000.0)


class SoliqManbaTest(unittest.TestCase):
    """Soliq stavkalarining manbasi va tekshiruv holati (D bandi)."""

    def test_har_bir_stavkada_manba_yozuvi_bor(self):
        self.assertEqual(set(hisob.SOLIQ_MANBA), set(hisob.SOLIQ))
        for k, m in hisob.SOLIQ_MANBA.items():
            with self.subTest(stavka=k):
                self.assertIn(m["holat"], ("yangiliklar bilan mos", "tasdiqlanmagan"))
                self.assertRegex(m["tekshirilgan"], r"^\d{4}-\d{2}-\d{2}$")
                if m["holat"] != "tasdiqlanmagan":
                    self.assertTrue(m["manba"])
                    self.assertTrue(m["havola"])

    def test_inps_rasmiy_tasdiqsiz_ozgartirilmagan(self):
        # INPS (xodim) stavkasi rasmiy manbada tasdiqlanmaguncha o'zgartirilmaydi va "tasdiqlanmagan" deb belgilanadi.
        # Tasdiqlanganda SOLIQ["INPS"], SOLIQ_MANBA["INPS"] va shu testni birga yangilang.
        self.assertEqual(hisob.SOLIQ["INPS"], 4.0)
        self.assertEqual(hisob.tasdiqlanmagan(), ["INPS"])

    def test_soliq_stavkalari_manba_va_holatni_qaytaradi(self):
        r = hisob.soliq_stavkalari()
        self.assertEqual(r["tasdiqlanmagan"], ["INPS"])
        self.assertEqual(r["manba"]["QQS"]["tekshirilgan"], hisob.SOLIQ_MANBA["QQS"]["tekshirilgan"])
        r["manba"]["QQS"]["holat"] = "buzildi"
        self.assertNotEqual(hisob.SOLIQ_MANBA["QQS"]["holat"], "buzildi")  # nusxa qaytariladi

    def test_ish_haqi_izohida_inps_ogohlantirishi(self):
        self.assertIn("INPS", hisob.ish_haqi(8_000_000)["izoh"].split("Diqqat:")[1])
        with mock.patch.dict(hisob.SOLIQ_MANBA["INPS"], {"holat": "yangiliklar bilan mos"}):
            self.assertNotIn("Diqqat", hisob.ish_haqi(8_000_000)["izoh"])


class EskiNatijalarTest(unittest.TestCase):
    """Eski (v0) kod bergan natijalar bilan solishtirish: matematik mantiq o'zgarmaganini isbotlaydi."""

    @classmethod
    def setUpClass(cls):
        yol = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_v0.json")
        with open(yol, encoding="utf-8") as f:
            cls.golden = json.load(f)

    def test_barcha_eski_kalitlar_bir_xil(self):
        hisob.set_rates(dict(self.golden["rates"]))
        self.addCleanup(hisob.set_rates, {})
        self.assertGreater(len(self.golden["cases"]), 400)
        for c in self.golden["cases"]:
            with self.subTest(fn=c["fn"], args=c["args"]):
                yangi = json.loads(hisob.call_tool(c["fn"], json.dumps(c["args"])))
                for kalit, qiymat in c["out"].items():
                    self.assertEqual(yangi.get(kalit), qiymat, kalit)

    def test_kredit_jadvallari_bir_xil(self):
        self.assertEqual(hisob._ann(1e8, 24, 36)[1], self.golden["rows"]["ann"])
        self.assertEqual(hisob._diff(1e8, 24, 36)[1], self.golden["rows"]["diff"])


if __name__ == "__main__":
    unittest.main()
