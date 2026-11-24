"""Hisob-kitob qatlami: soliq stavkalari, kredit/QQS/ish haqi/prognoz funksiyalari va AI asboblari (TOOLS).

Bu modul Streamlit'ga bog'liq emas, shuning uchun uni alohida test qilish mumkin.
"""
import copy
import json
import math

# ====================== SOZLAMALAR ======================
# Soliq stavkalari (%). Qonun o'zgarsa, faqat shu yerni yangilang.
# DIQQAT: stavkani o'zgartirsangiz, quyidagi SOLIQ_MANBA ni ham yangilang (manba, sana, holat).
SOLIQ = {"JSHDS": 12.0, "INPS": 4.0, "Ijtimoiy soliq": 12.0, "QQS": 12.0,
         "Foyda solig'i": 15.0, "Aylanma solig'i": 4.0}

# Har bir stavkaning manbasi va tekshiruv holati.
#   holat "yangiliklar bilan mos": rasmiy yangiliklar (gov.uz, kun.uz, anhor.uz) bilan solishtirilgan, lex.uz bilan hali emas;
#   holat "tasdiqlanmagan":        rasmiy manba topilmagan, stavka o'zgartirilmagan va ishonchli deb hisoblanmaydi.
_HAVOLA_2026 = ["https://gov.uz/oz/urganch/news/view/138410",
                "https://kun.uz/news/2025/12/18/2026-yilda-asosiy-soliqlar-boyicha-stavkalar-ozgarmaydi",
                "https://anhor.uz/uzl/news/asosiy-soliq-stavkalari/"]
SOLIQ_MANBA = {k: {"holat": "yangiliklar bilan mos", "yil": 2026, "tekshirilgan": "2026-11-24",
                   "manba": "2026-yil uchun asosiy soliq stavkalari o'zgarishsiz qolgani haqidagi xabarlar (gov.uz, kun.uz, anhor.uz)",
                   "havola": list(_HAVOLA_2026)} for k in ("JSHDS", "QQS", "Foyda solig'i", "Ijtimoiy soliq", "Aylanma solig'i")}
SOLIQ_MANBA["Ijtimoiy soliq"]["eslatma"] = "Budjet tashkilotlari uchun stavka 25%; ayrim sohalar uchun imtiyozli stavkalar bor."
SOLIQ_MANBA["Aylanma solig'i"]["eslatma"] = "Ayrim toifalar (masalan marketpleys, YaTT va o'zini o'zi band qilganlar) uchun alohida tartib bo'lishi mumkin."
SOLIQ_MANBA["INPS"] = {"holat": "tasdiqlanmagan", "yil": 2026, "tekshirilgan": "2026-11-24", "manba": None, "havola": [],
                       "eslatma": "Rasmiy manba topilmadi: lex.uz yoki soliq.uz dan tekshiring. Stavka o'zgartirilmagan."}


def tasdiqlanmagan(*kalitlar):
    """Rasmiy manba bilan tasdiqlanmagan stavkalar ro'yxati (kalit berilmasa, hammasi ichidan)."""
    return [k for k in (kalitlar or SOLIQ) if SOLIQ_MANBA.get(k, {}).get("holat") == "tasdiqlanmagan"]


# ====================== HISOB-KITOB FUNKSIYALARI ======================
class HisobXatosi(ValueError):
    """Noto'g'ri kirish qiymati. Xabar foydalanuvchiga ko'rsatish uchun o'zbekcha yoziladi."""


MAX_OY = 600           # kredit muddati chegarasi (oy)
MAX_FOIZ = 1000.0      # yillik foiz chegarasi (%): yozuv xatolarini ushlash uchun
VALYUTALAR = ("UZS", "USD", "EUR", "RUB")


def _f(x):
    return float(str(x).replace(" ", "").replace(",", "."))


def _son(x, nom, dan=None, dan_katta=None, gacha=None, butun=False):
    """_f() ning tekshiruvli varianti: son bo'lishi, cheksiz/NaN emasligi va chegaralarni tekshiradi.

    dan: v >= dan, dan_katta: v > dan_katta, gacha: v <= gacha, butun: faqat butun son (12.0 ham mos).
    """
    if x is None or isinstance(x, bool):
        raise HisobXatosi(f"«{nom}» son bo'lishi kerak.")
    try:
        v = _f(x)
    except (TypeError, ValueError):
        raise HisobXatosi(f"«{nom}» son bo'lishi kerak (berilgan: {x!r}).") from None
    if not math.isfinite(v):
        raise HisobXatosi(f"«{nom}» oddiy son bo'lishi kerak (cheksiz yoki NaN bo'lmasin).")
    if butun:
        if v != int(v):
            raise HisobXatosi(f"«{nom}» butun son bo'lishi kerak (berilgan: {x}).")
        v = int(v)
    if dan is not None and v < dan:
        raise HisobXatosi(f"«{nom}» {dan:g} dan kichik bo'lmasligi kerak (berilgan: {x}).")
    if dan_katta is not None and v <= dan_katta:
        raise HisobXatosi(f"«{nom}» {dan_katta:g} dan katta bo'lishi kerak (berilgan: {x}).")
    if gacha is not None and v > gacha:
        raise HisobXatosi(f"«{nom}» {gacha:g} dan oshmasligi kerak (berilgan: {x}).")
    return v


def _mantiq(x, nom):
    """True/False qiymatini tekshirib o'qiydi (true/false, 1/0, ha/yo'q)."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    t = str(x).strip().lower()
    if t in ("true", "1", "ha", "yes"):
        return True
    if t in ("false", "0", "yo'q", "yoq", "no", "", "none", "null"):
        return False
    raise HisobXatosi(f"«{nom}» true yoki false bo'lishi kerak (berilgan: {x!r}).")


def _fm(x):
    """Raqamni probel bilan ajratib yozadi: 12500000 -> '12 500 000'."""
    return f"{x:,.0f}".replace(",", " ")


def _kredit_tekshir(s, p, n):
    return (_son(s, "summa", dan_katta=0),
            _son(p, "yillik_foiz", dan=0, gacha=MAX_FOIZ),
            _son(n, "oy", butun=True, dan=1, gacha=MAX_OY))

def _ann(s, p, n):
    s, p, n = _kredit_tekshir(s, p, n)
    r = p / 1200
    t = s / n if r == 0 else s * r * (1 + r) ** n / ((1 + r) ** n - 1)
    q, rows = s, []
    for i in range(1, n + 1):
        f = q * r
        q -= t - f
        rows.append({"Oy": i, "To'lov": round(t), "Asosiy qarz": round(t - f), "Foiz": round(f), "Qoldiq": round(max(q, 0))})
    res = {"oylik_tolov": round(t), "jami_tolov": round(t * n), "jami_foiz": round(t * n - s)}
    res["formula"] = ("A = S / n (foiz 0% bo'lgani uchun)" if r == 0 else "A = S·r·(1+r)^n / ((1+r)^n − 1)") + \
        f"; S = {_fm(s)} so'm, r = {p:g}%/12 = {r * 100:.4f}% (oylik), n = {n} oy"
    res["izoh"] = (f"Annuitet kreditda {n} oy davomida har oy bir xil {_fm(res['oylik_tolov'])} so'm to'lanadi. "
                   f"Jami to'lov {_fm(res['jami_tolov'])} so'm, shundan foiz {_fm(res['jami_foiz'])} so'm.")
    return res, rows

def _diff(s, p, n):
    s, p, n = _kredit_tekshir(s, p, n)
    r, a, q, rows = p / 1200, s / n, s, []
    for i in range(1, n + 1):
        f = q * r
        rows.append({"Oy": i, "To'lov": round(a + f), "Asosiy qarz": round(a), "Foiz": round(f), "Qoldiq": round(max(q - a, 0))})
        q -= a
    jf = sum(x["Foiz"] for x in rows)
    res = {"birinchi_oy_tolov": rows[0]["To'lov"], "oxirgi_oy_tolov": rows[-1]["To'lov"], "jami_tolov": round(s + jf), "jami_foiz": jf}
    res["formula"] = (f"To'lov(k) = S/n + Qoldiq(k)·r; S/n = {_fm(s)}/{n}, r = {p:g}%/12 = {r * 100:.4f}% (oylik); "
                      "Qoldiq(k) — k-oy boshidagi qarz")
    res["izoh"] = (f"Differensial kreditda asosiy qarz har oy teng ulushda to'lanadi, foiz esa qoldiqdan hisoblanadi, shuning uchun to'lov "
                   f"kamayib boradi: {_fm(res['birinchi_oy_tolov'])} so'mdan {_fm(res['oxirgi_oy_tolov'])} so'mgacha. "
                   f"Jami to'lov {_fm(res['jami_tolov'])} so'm, shundan foiz {_fm(res['jami_foiz'])} so'm.")
    return res, rows

def annuitet_kredit(summa, yillik_foiz, oy):
    return _ann(summa, yillik_foiz, oy)[0]

def differensial_kredit(summa, yillik_foiz, oy):
    return _diff(summa, yillik_foiz, oy)[0]

def ish_haqi(brutto):
    b = _son(brutto, "brutto", dan=0)
    j, i, ij = (b * SOLIQ[k] / 100 for k in ("JSHDS", "INPS", "Ijtimoiy soliq"))
    eslatma = "Ijtimoiy soliq stavkasi budjet tashkilotlari uchun 25%."
    if tasdiqlanmagan("JSHDS", "INPS", "Ijtimoiy soliq"):
        eslatma += f" Diqqat: {', '.join(tasdiqlanmagan('JSHDS', 'INPS', 'Ijtimoiy soliq'))} stavkasi rasmiy manba bilan hali tasdiqlanmagan."
    return {"brutto": round(b), "JSHDS": round(j), "INPS": round(i), "qolga_tegadi": round(b - j - i),
            "ijtimoiy_soliq_ish_beruvchi": round(ij), "ish_beruvchi_jami_xarajat": round(b + ij),
            "formula": (f"JSHDS = brutto·{SOLIQ['JSHDS']:g}%; INPS = brutto·{SOLIQ['INPS']:g}%; qo'lga tegadi = brutto − JSHDS − INPS; "
                        f"ijtimoiy soliq (ish beruvchi) = brutto·{SOLIQ['Ijtimoiy soliq']:g}%; ish beruvchi xarajati = brutto + ijtimoiy soliq"),
            "izoh": (f"{_fm(b)} so'm brutto maoshdan {_fm(j)} so'm JSHDS va {_fm(i)} so'm INPS ushlanadi, qo'lga {_fm(b - j - i)} so'm tegadi. "
                     f"Ish beruvchi qo'shimcha {_fm(ij)} so'm ijtimoiy soliq to'laydi. "
                     "Soddalashtirilgan hisob: JSHDS va INPS bruttodan olindi, imtiyoz va chegirmalar hisobga olinmagan. " + eslatma)}

def qqs(summa, qqs_bilan=False, stavka=None):
    s = _son(summa, "summa", dan=0)
    # stavka=0 haqiqiy 0% stavka hisoblanadi; stavka berilmasa (None yoki bo'sh) QQS stavkasi olinadi
    p = SOLIQ["QQS"] if stavka is None or str(stavka).strip() == "" else _son(stavka, "stavka", dan=0, gacha=100)
    if _mantiq(qqs_bilan, "qqs_bilan"):
        net = s / (1 + p / 100)
        return {"stavka": p, "qqssiz": round(net), "qqs": round(s - net), "qqs_bilan": round(s),
                "formula": f"QQSsiz = summa / (1 + {p:g}/100); QQS = summa − QQSsiz",
                "izoh": f"{_fm(s)} so'm QQS bilan berilgan ({p:g}%): QQSsiz summa {_fm(net)} so'm, QQS {_fm(s - net)} so'm."}
    return {"stavka": p, "qqssiz": round(s), "qqs": round(s * p / 100), "qqs_bilan": round(s * (1 + p / 100)),
            "formula": f"QQS = summa·{p:g}/100; QQS bilan = summa·(1 + {p:g}/100)",
            "izoh": f"{_fm(s)} so'm QQSsiz summaga {p:g}% QQS qo'shilsa: QQS {_fm(s * p / 100)} so'm, jami {_fm(s * (1 + p / 100))} so'm."}

def murakkab_foiz(summa, yillik_foiz, yil, yiliga_marta=12):
    s = _son(summa, "summa", dan_katta=0)
    p = _son(yillik_foiz, "yillik_foiz", dan=0, gacha=MAX_FOIZ)
    y = _son(yil, "yil", dan_katta=0, gacha=100)
    m = _son(yiliga_marta, "yiliga_marta", butun=True, dan=1, gacha=365)
    try:
        fv = s * (1 + p / 100 / m) ** (m * y)
    except OverflowError:
        raise HisobXatosi("Natija juda katta chiqdi: foiz yoki muddatni kamaytiring.") from None
    return {"yakuniy_summa": round(fv), "foyda": round(fv - s),
            "formula": f"S_n = S·(1 + r/m)^(m·t); S = {_fm(s)}, r = {p:g}%, m = {m} (yiliga marta), t = {y:g} yil",
            "izoh": f"{_fm(s)} so'm yillik {p:g}% bilan, yiliga {m} marta foiz qo'shilganda {y:g} yildan keyin {_fm(fv)} so'm bo'ladi; foyda {_fm(fv - s)} so'm."}

def biznes_tahlil(narx, ozgaruvchan_xarajat, doimiy_xarajat, sotuv_soni):
    n = _son(narx, "narx", dan_katta=0)
    v = _son(ozgaruvchan_xarajat, "ozgaruvchan_xarajat", dan=0)
    d = _son(doimiy_xarajat, "doimiy_xarajat", dan=0)
    q = _son(sotuv_soni, "sotuv_soni", dan=0)
    marja = n - v
    daromad = n * q
    foyda = marja * q - d
    be = math.ceil(d / marja) if marja > 0 else None
    if be is None:
        izoh = "Birlik marjasi noldan katta emas: har bir sotuv xarajatni qoplamaydi, zararsizlik nuqtasiga yetib bo'lmaydi."
    else:
        holat = "zararsizlik nuqtasidan yuqori, foyda bor" if q > be else ("zararsizlik nuqtasida" if q == be else "zararsizlik nuqtasidan past, zarar bor")
        izoh = f"Zararsizlik nuqtasi {be} dona; hozirgi sotuv {q:g} dona ({holat}). Sof foyda {_fm(foyda)} so'm."
    return {"birlik_marjasi": round(marja), "marja_foizi": round(marja / n * 100, 1) if n else None,
            "zararsizlik_soni": be,
            "umumiy_daromad": round(daromad), "sof_foyda": round(foyda),
            "rentabellik_foizi": round(foyda / daromad * 100, 1) if daromad else None,
            "formula": ("Marja = narx − o'zgaruvchan xarajat; zararsizlik soni = doimiy xarajat / marja (yuqoriga yaxlitlanadi); "
                        "sof foyda = marja·sotuv − doimiy xarajat; rentabellik = sof foyda / daromad·100"),
            "izoh": izoh}

def valyuta_aylantirish(summa, kimdan, kimga):
    def k(c):
        c = str(c).strip().upper()
        if c not in VALYUTALAR:
            nom = c or "bo'sh"
            raise HisobXatosi(f"Noma'lum valyuta kodi: {nom}. Qo'llab-quvvatlanadi: {', '.join(VALYUTALAR)}.")
        if c == "UZS":
            return 1.0
        try:
            return _son(cbu[c], "kurs", dan_katta=0)
        except (KeyError, HisobXatosi):
            raise HisobXatosi("CBU kursi hozir mavjud emas: " + c) from None
    sm = _son(summa, "summa", dan=0)
    natija = round(sm * k(kimdan) / k(kimga), 2)
    a, b = str(kimdan).strip().upper(), str(kimga).strip().upper()
    return {"natija": natija, "manba": "CBU kursi",
            "formula": "natija = summa · kurs(kimdan) / kurs(kimga); UZS kursi = 1",
            "izoh": f"{sm:,.2f} {a} = {natija:,.2f} {b} (CBU rasmiy kursi bo'yicha)"}

def soliq_stavkalari():
    return {**SOLIQ, "izoh": "Barcha stavkalar foizda (%). Har bir stavkaning manbasi va holati 'manba' maydonida.",
            "manba": copy.deepcopy(SOLIQ_MANBA), "tasdiqlanmagan": tasdiqlanmagan()}

def prognoz(qiymatlar, davr_soni=3):
    """Tarixiy qator bo'yicha chiziqli trend va CAGR prognozi (oddiy statistik usul)."""
    if not isinstance(qiymatlar, (list, tuple)):
        raise HisobXatosi("«qiymatlar» sonlar ro'yxati bo'lishi kerak.")
    v = [_son(x, f"qiymatlar[{i + 1}]") for i, x in enumerate(qiymatlar)]
    n, k = len(v), max(1, min(int(_son(davr_soni, "davr_soni")), 12))
    if n < 3:
        raise HisobXatosi("Kamida 3 ta tarixiy qiymat kerak.")
    mx, my = (n - 1) / 2, sum(v) / n
    sxx = sum((i - mx) ** 2 for i in range(n))
    b = sum((i - mx) * (v[i] - my) for i in range(n)) / sxx
    a = my - b * mx
    std = math.sqrt(sum((v[i] - (a + b * i)) ** 2 for i in range(n)) / (n - 2)) if n > 2 else 0.0
    lin = [a + b * (n - 1 + j) for j in range(1, k + 1)]
    g = (v[-1] / v[0]) ** (1 / (n - 1)) - 1 if v[0] > 0 and v[-1] > 0 else None
    return {"oxirgi_qiymat": round(v[-1], 2),
            "chiziqli_trend": [round(x, 2) for x in lin],
            "taxminiy_past_chegara": [round(x - 1.96 * std, 2) for x in lin],
            "taxminiy_yuqori_chegara": [round(x + 1.96 * std, 2) for x in lin],
            "davrlik_ortacha_osish_foizi": round(g * 100, 2) if g is not None else None,
            "osish_bo'yicha_prognoz": [round(v[-1] * (1 + g) ** j, 2) for j in range(1, k + 1)] if g is not None else None,
            "formula": ("Chiziqli trend: y = a + b·t (eng kichik kvadratlar); chegaralar: trend ± 1,96·σ (σ — qoldiqlarning standart og'ishi); "
                        "o'sish: CAGR = (oxirgi/birinchi)^(1/(n−1)) − 1"),
            "izoh": "Bu faqat statistik ekstrapolyatsiya; tashqi omillar hisobga olinmagan."}

FUNCS = {f.__name__: f for f in (annuitet_kredit, differensial_kredit, ish_haqi, qqs, murakkab_foiz,
                                 biznes_tahlil, valyuta_aylantirish, soliq_stavkalari, prognoz)}

def _tool(name, desc, props=None, req=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": req or []}}}

N, I = {"type": "number"}, {"type": "integer"}
TOOLS = [
    _tool("annuitet_kredit", "Annuitet kredit: oylik to'lov, jami to'lov, jami foiz (so'm)",
          {"summa": N, "yillik_foiz": N, "oy": I}, ["summa", "yillik_foiz", "oy"]),
    _tool("differensial_kredit", "Differensial kredit: birinchi/oxirgi oy to'lovi, jami to'lov, jami foiz",
          {"summa": N, "yillik_foiz": N, "oy": I}, ["summa", "yillik_foiz", "oy"]),
    _tool("ish_haqi", "Brutto maoshdan JSHDS, INPS, qo'lga tegadigan summa va ijtimoiy soliq", {"brutto": N}, ["brutto"]),
    _tool("qqs", "QQS hisobi. qqs_bilan=true bo'lsa summa QQS bilan berilgan (ajratib olinadi). stavka ixtiyoriy: foydalanuvchi aytmasa umuman bermang (joriy QQS stavkasi olinadi); 0 — haqiqiy 0% stavka",
          {"summa": N, "qqs_bilan": {"type": "boolean"}, "stavka": N}, ["summa"]),
    _tool("murakkab_foiz", "Depozit/investitsiya murakkab foiz bilan yakuniy summa",
          {"summa": N, "yillik_foiz": N, "yil": N, "yiliga_marta": I}, ["summa", "yillik_foiz", "yil"]),
    _tool("biznes_tahlil", "Unit-ekonomika: marja, zararsizlik nuqtasi, sof foyda, rentabellik",
          {"narx": N, "ozgaruvchan_xarajat": N, "doimiy_xarajat": N, "sotuv_soni": N},
          ["narx", "ozgaruvchan_xarajat", "doimiy_xarajat", "sotuv_soni"]),
    _tool("valyuta_aylantirish", "CBU kursi bo'yicha valyuta aylantirish (UZS, USD, EUR, RUB)",
          {"summa": N, "kimdan": {"type": "string"}, "kimga": {"type": "string"}}, ["summa", "kimdan", "kimga"]),
    _tool("soliq_stavkalari", "Joriy soliq stavkalari (JSHDS, INPS, ijtimoiy soliq, QQS, foyda solig'i)"),
    _tool("prognoz", "Tarixiy raqamlar qatori (eskidan yangiga) bo'yicha chiziqli trend va o'sish sur'ati prognozi, ishonch oralig'i bilan",
          {"qiymatlar": {"type": "array", "items": {"type": "number"}}, "davr_soni": I}, ["qiymatlar"]),
]

def call_tool(name, args):
    try:
        if name not in FUNCS:
            raise HisobXatosi("Noma'lum asbob: " + str(name))
        return json.dumps(FUNCS[name](**json.loads(args or "{}")), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"xato": str(e)}, ensure_ascii=False)


# ====================== CBU KURSLARI ======================
# valyuta_aylantirish() shu lug'atdan foydalanadi. app.py kurslarni yuklagach set_rates() ni chaqiradi.
cbu = {}


def set_rates(rates):
    """CBU kurslarini o'rnatadi: {"USD": "12100.5", ...}"""
    global cbu
    cbu = rates
