import base64, io, json, math, os, re, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import pandas as pd
import requests
import streamlit as st
from groq import Groq
from pypdf import PdfReader

DDGS = None
for _m in ("ddgs", "duckduckgo_search"):
    try:
        DDGS = __import__(_m, fromlist=["DDGS"]).DDGS
        break
    except ImportError:
        pass

st.set_page_config(page_title="Iqtisodchi AI", page_icon="🏛️", layout="centered")
st.markdown("<style>.stApp{background:#F9F8F6;color:#1F1F1F}section[data-testid='stSidebar']{background:#F2F0EC!important}.hero{text-align:center;margin:1rem 0}.logo{display:inline-flex;align-items:center;gap:12px;text-align:left;margin-bottom:10px}.ltxt{font-weight:700;font-size:15px;line-height:1.15;color:#16354D}div[data-testid='stChatMessage']:has(div[data-testid='stChatMessageAvatarUser']){flex-direction:row-reverse;background:#EFECE6!important;border-radius:18px!important;padding:10px 16px!important;margin-left:auto!important;max-width:82%}div[data-testid='stChatMessage']:has(div[data-testid='stChatMessageAvatarUser']) div[data-testid='stChatMessageAvatarUser']{display:none}div.stButton>button{border-radius:14px!important;background:#FFF!important}</style>", unsafe_allow_html=True)

NAVY, RED = "#16354D", "#FF3B22"
# (top, left, width, height, rang) px
_BLOCKS = ((0, 0, 60, 10, NAVY), (0, 0, 10, 60, NAVY), (14, 14, 9, 24, RED), (14, 26, 9, 30, RED), (14, 38, 9, 17, RED),
           (14, 50, 10, 9, RED), (47, 26, 9, 13, NAVY), (34, 38, 9, 26, NAVY), (26, 50, 10, 34, NAVY))
LOGO = '<div style="width:60px;height:60px;position:relative;flex-shrink:0">' + "".join(
    f'<div style="position:absolute;top:{t}px;left:{l}px;width:{w}px;height:{h}px;background:{c}"></div>'
    for t, l, w, h, c in _BLOCKS) + "</div>"

# ====================== SOZLAMALAR ======================
# Soliq stavkalari (%). Qonun o'zgarsa, faqat shu yerni yangilang.
# DIQQAT: INPS (xodim) stavkasini lex.uz dan tekshirib qo'ying.
SOLIQ = {"JSHDS": 12.0, "INPS": 4.0, "Ijtimoiy soliq": 12.0, "QQS": 12.0, "Foyda solig'i": 15.0, "Aylanma solig'i": 4.0}

# ====================== TIZIM KO'RSATMALARI ======================
UMUMIY = (
    "Ravon o'zbek tilida (lotin) yoz. Raqamli hisob-kitobni HECH QACHON o'zing qilma: "
    "tegishli asbobdan (tool) foydalan va natijani faqat shundan ol. "
    "Javob tuzilishi: 1) birinchi jumlada natijani **qalin** qilib ayt; 2) qisqa tushuntirish; "
    "3) oxirida ishlatilgan formula (agar hisob bo'lsa). Raqamlarni probel bilan ajrat (12 500 000 so'm). "
    "Aniq bilmagan narsani taxmin qilma, 'tekshirib ko'ring' de. "
    "Kod so'ralsa, to'liq ishlaydigan Python kodini ```python blokida yoz va qisqa izoh ber. "
    "Foydalanuvchi rasm yuborsa (hujjat, chek, hisob-faktura, jadval, grafik), uni diqqat bilan o'qib tahlil qil; "
    "o'qib bo'lmagan yoki ko'rinmagan joyni taxmin qilma, ochiq ayt."
)
SYSTEM = {
    "gen": "Sen TMII (Toshkent menejment va iqtisodiyot instituti) Iqtisodchi AI'san: iqtisodiyot, moliya, "
           "biznes tahlil, kredit va investitsiya masalalari bo'yicha yordamchi. Biznes tahlilda unit-ekonomika, "
           "zararsizlik nuqtasi, rentabellik, SWOT kabi usullardan foydalan. " + UMUMIY,
    "acc": "Sen TMII Bosh Buxgalter AI'san: O'zbekiston buxgalteriya hisobi milliy standartlari (BHMS) va "
           "Soliq kodeksi bo'yicha mutaxassis. Provodkalarni DOIM jadval shaklida yoz: "
           "| № | Debet | Kredit | Summa | Izoh |. Hisob raqamlarini O'zbekiston schyotlar rejasi bo'yicha ko'rsat; "
           "ishonching komil bo'lmagan raqamni '(tekshiring)' deb belgila. Soliq stavkalarini faqat "
           "soliq_stavkalari asbobidan ol. " + UMUMIY,
    "fc": "Sen TMII Iqtisodiy Prognoz AI'san: The Economist uslubidagi iqtisodiy tahlilchi va dasturchi. "
          "Inflyatsiya, so'm kursi, foiz stavkalari, iqtisodiy o'sish, tashqi savdo, tovar bozorlari bo'yicha prognoz berasan. "
          "Qoidalar: "
          "1) Prognozni DOIM uch ssenariyda ber: asosiy, optimistik, pessimistik, har biriga taxminiy ehtimol (%) bilan. "
          "2) Farazlarni va asosiy omillarni (drivers) ayt. "
          "3) Xavflarni va kuzatiladigan signal ko'rsatkichlarni ayt. "
          "4) Internet ma'lumotini manbasi bilan ishlat; ma'lumot eskirgan yoki yetarli bo'lmasa, ochiq ayt. "
          "5) Tarixiy raqamlar qatori bo'lsa (masalan, tizim bergan kurs tarixi), prognoz asbobidan foydalan. "
          "6) Oxirida: 'Bu prognoz, kafolat emas.' "
          "Uslub: sarlavha hukm bilan boshlansin (masalan 'So'm bosim ostida qoladi'), keyin aniq, qisqa tezislar. "
          "Foydalanuvchi savolida davr ko'rsatilmasa, eng yaqin yil/chorakni ol. " + UMUMIY,
}

# ====================== HISOB-KITOB FUNKSIYALARI ======================
def _f(x):
    return float(str(x).replace(" ", "").replace(",", "."))

def _plan(s, p, n, diff=False):
    """Kredit jadvali. diff=False: annuitet, True: differensial. -> (xulosa, qatorlar)"""
    r, q, rows = p / 1200, s, []
    t = s / n if r == 0 else s * r * (1 + r) ** n / ((1 + r) ** n - 1)
    for i in range(1, n + 1):
        f = q * r
        a = s / n if diff else t - f
        rows.append({"Oy": i, "To'lov": round(a + f if diff else t), "Asosiy qarz": round(a), "Foiz": round(f), "Qoldiq": round(max(q - a, 0))})
        q -= a
    if not diff:
        return {"oylik_tolov": round(t), "jami_tolov": round(t * n), "jami_foiz": round(t * n - s)}, rows
    jf = sum(x["Foiz"] for x in rows)
    return {"birinchi_oy_tolov": rows[0]["To'lov"], "oxirgi_oy_tolov": rows[-1]["To'lov"], "jami_tolov": round(s + jf), "jami_foiz": jf}, rows

def annuitet_kredit(summa, yillik_foiz, oy): return _plan(_f(summa), _f(yillik_foiz), int(_f(oy)))[0]
def differensial_kredit(summa, yillik_foiz, oy): return _plan(_f(summa), _f(yillik_foiz), int(_f(oy)), True)[0]

def ish_haqi(brutto):
    b = _f(brutto)
    j, i, ij = (b * SOLIQ[k] / 100 for k in ("JSHDS", "INPS", "Ijtimoiy soliq"))
    return {"brutto": round(b), "JSHDS": round(j), "INPS": round(i), "qolga_tegadi": round(b - j - i),
            "ijtimoiy_soliq_ish_beruvchi": round(ij), "ish_beruvchi_jami_xarajat": round(b + ij)}

def qqs(summa, qqs_bilan=False, stavka=None):
    s = _f(summa)
    p = _f(stavka) if stavka else SOLIQ["QQS"]
    if qqs_bilan in (True, 1, "true", "True", "ha"):
        net = s / (1 + p / 100)
        return {"stavka": p, "qqssiz": round(net), "qqs": round(s - net), "qqs_bilan": round(s)}
    return {"stavka": p, "qqssiz": round(s), "qqs": round(s * p / 100), "qqs_bilan": round(s * (1 + p / 100))}

def murakkab_foiz(summa, yillik_foiz, yil, yiliga_marta=12):
    s, p, y, m = _f(summa), _f(yillik_foiz), _f(yil), _f(yiliga_marta)
    fv = s * (1 + p / 100 / m) ** (m * y)
    return {"yakuniy_summa": round(fv), "foyda": round(fv - s)}

def biznes_tahlil(narx, ozgaruvchan_xarajat, doimiy_xarajat, sotuv_soni):
    n, v, d, q = _f(narx), _f(ozgaruvchan_xarajat), _f(doimiy_xarajat), _f(sotuv_soni)
    marja, daromad = n - v, n * q
    foyda = marja * q - d
    return {"birlik_marjasi": round(marja), "marja_foizi": round(marja / n * 100, 1) if n else None,
            "zararsizlik_soni": math.ceil(d / marja) if marja > 0 else None,
            "umumiy_daromad": round(daromad), "sof_foyda": round(foyda),
            "rentabellik_foizi": round(foyda / daromad * 100, 1) if daromad else None}

def valyuta_aylantirish(summa, kimdan, kimga):
    def k(c):
        c = c.upper()
        try:
            return 1.0 if c == "UZS" else float(cbu[c])
        except Exception:
            raise ValueError("CBU kursi hozir mavjud emas: " + c)
    return {"natija": round(_f(summa) * k(kimdan) / k(kimga), 2), "manba": "CBU kursi"}

def soliq_stavkalari():
    return SOLIQ

def prognoz(qiymatlar, davr_soni=3):
    """Tarixiy qator bo'yicha chiziqli trend va CAGR prognozi (oddiy statistik usul)."""
    v = [_f(x) for x in qiymatlar]
    n, k = len(v), max(1, min(int(_f(davr_soni)), 12))
    if n < 3:
        raise ValueError("Kamida 3 ta tarixiy qiymat kerak")
    mx, my = (n - 1) / 2, sum(v) / n
    b = sum((i - mx) * (v[i] - my) for i in range(n)) / sum((i - mx) ** 2 for i in range(n))
    a = my - b * mx
    std = math.sqrt(sum((v[i] - (a + b * i)) ** 2 for i in range(n)) / (n - 2))
    lin = [a + b * (n - 1 + j) for j in range(1, k + 1)]
    g = (v[-1] / v[0]) ** (1 / (n - 1)) - 1 if v[0] > 0 and v[-1] > 0 else None
    return {"oxirgi_qiymat": round(v[-1], 2),
            "chiziqli_trend": [round(x, 2) for x in lin],
            "taxminiy_past_chegara": [round(x - 1.96 * std, 2) for x in lin],
            "taxminiy_yuqori_chegara": [round(x + 1.96 * std, 2) for x in lin],
            "davrlik_ortacha_osish_foizi": round(g * 100, 2) if g is not None else None,
            "osish_bo'yicha_prognoz": [round(v[-1] * (1 + g) ** j, 2) for j in range(1, k + 1)] if g is not None else None,
            "izoh": "Bu faqat statistik ekstrapolyatsiya; tashqi omillar hisobga olinmagan."}

FUNCS = {f.__name__: f for f in (annuitet_kredit, differensial_kredit, ish_haqi, qqs, murakkab_foiz,
                                 biznes_tahlil, valyuta_aylantirish, soliq_stavkalari, prognoz)}

N, I = {"type": "number"}, {"type": "integer"}
_KREDIT = {"summa": N, "yillik_foiz": N, "oy": I}
# (nom, tavsif, parametrlar, majburiy parametrlar)
TOOLS = [{"type": "function", "function": {"name": n, "description": d,
          "parameters": {"type": "object", "properties": p, "required": r.split()}}} for n, d, p, r in (
    ("annuitet_kredit", "Annuitet kredit: oylik to'lov, jami to'lov, jami foiz (so'm)", _KREDIT, "summa yillik_foiz oy"),
    ("differensial_kredit", "Differensial kredit: birinchi/oxirgi oy to'lovi, jami to'lov, jami foiz", _KREDIT, "summa yillik_foiz oy"),
    ("ish_haqi", "Brutto maoshdan JSHDS, INPS, qo'lga tegadigan summa va ijtimoiy soliq", {"brutto": N}, "brutto"),
    ("qqs", "QQS hisobi. qqs_bilan=true bo'lsa summa QQS bilan berilgan (ajratib olinadi)",
     {"summa": N, "qqs_bilan": {"type": "boolean"}, "stavka": N}, "summa"),
    ("murakkab_foiz", "Depozit/investitsiya murakkab foiz bilan yakuniy summa",
     {"summa": N, "yillik_foiz": N, "yil": N, "yiliga_marta": I}, "summa yillik_foiz yil"),
    ("biznes_tahlil", "Unit-ekonomika: marja, zararsizlik nuqtasi, sof foyda, rentabellik",
     {"narx": N, "ozgaruvchan_xarajat": N, "doimiy_xarajat": N, "sotuv_soni": N}, "narx ozgaruvchan_xarajat doimiy_xarajat sotuv_soni"),
    ("valyuta_aylantirish", "CBU kursi bo'yicha valyuta aylantirish (UZS, USD, EUR, RUB)",
     {"summa": N, "kimdan": {"type": "string"}, "kimga": {"type": "string"}}, "summa kimdan kimga"),
    ("soliq_stavkalari", "Joriy soliq stavkalari (JSHDS, INPS, ijtimoiy soliq, QQS, foyda solig'i)", {}, ""),
    ("prognoz", "Tarixiy raqamlar qatori (eskidan yangiga) bo'yicha chiziqli trend va o'sish sur'ati prognozi, ishonch oralig'i bilan",
     {"qiymatlar": {"type": "array", "items": {"type": "number"}}, "davr_soni": I}, "qiymatlar"),
)]

def call_tool(name, args):
    try:
        return json.dumps(FUNCS[name](**json.loads(args or "{}")), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"xato": str(e)})

# ====================== GROQ, KURS, TARIX ======================
try:
    KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    KEY = os.getenv("GROQ_API_KEY")
if not KEY:
    st.error("GROQ_API_KEY topilmadi. Secrets bo'limiga qo'shing.")
    st.stop()
client = Groq(api_key=KEY)

def _pick(prefs, skip=()):
    """Groq model ro'yxatidan birinchi mos keladigan modelni qaytaradi."""
    ids = [m.id for m in client.models.list().data if not any(x in m.id for x in skip)]
    return next((m for p in prefs for m in ids if p in m), None), ids

@st.cache_data(ttl=3600)
def get_model():
    try:
        m, ids = _pick(("llama-3.3-70b", "gpt-oss", "llama-4", "qwen", "llama-3.1-70b", "llama-3.1-8b"), ("whisper", "guard", "tts"))
        if m or ids:
            return m or ids[0]
    except Exception:
        pass
    return "llama-3.3-70b-versatile"

MODEL = get_model()

@st.cache_data(ttl=3600)
def get_vision_model():
    """Rasm tushunadigan modelni Groq ro'yxatidan topadi."""
    try:
        m = _pick(("llama-4-scout", "llama-4-maverick", "vision"))[0]
        if m:
            return m
    except Exception:
        pass
    return "meta-llama/llama-4-scout-17b-16e-instruct"

def img_to_url(f):
    """Yuklangan rasmni kichraytirib, base64 data URL ga aylantiradi (Groq limitiga sig'sin)."""
    from PIL import Image
    im = Image.open(io.BytesIO(f.getvalue())).convert("RGB")
    im.thumbnail((1280, 1280))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

@st.cache_data(ttl=600)
def get_rates():
    try:
        r = requests.get("https://cbu.uz/oz/arkhiv-kursov-valyut/json/", timeout=5).json()
        return {i["Ccy"]: i["Rate"] for i in r if i.get("Ccy") in ("USD", "EUR", "RUB")}
    except Exception:
        return {"USD": "-", "EUR": "-", "RUB": "-"}

cbu = get_rates()

@st.cache_data(ttl=3600)
def get_history(ccy, days=30):
    """CBU arxividan so'nggi `days` kunlik kurs: [(sana, kurs), ...] eskidan yangiga."""
    def one(d):
        try:
            r = requests.get(f"https://cbu.uz/oz/arkhiv-kursov-valyut/json/{ccy}/{d.isoformat()}/", timeout=5).json()
            return d, float(str(r[0]["Rate"]).replace(",", "."))
        except Exception:
            return d, None
    today = date.today()
    with ThreadPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(one, [today - timedelta(days=i) for i in range(days)]))
    return sorted((d, v) for d, v in res if v is not None)

def fx_context():
    """Prognoz rejimi uchun: USD/UZS so'nggi 30 kunlik kurs matni."""
    h = get_history("USD")
    if len(h) < 3:
        return ""
    vals = [v for _, v in h]
    return (f"\nUSD/UZS CBU kursi, {h[0][0]} dan {h[-1][0]} gacha ({len(vals)} kun, eskidan yangiga): {vals}. "
            f"O'zgarish: {(vals[-1] / vals[0] - 1) * 100:+.2f}%, min {min(vals):.0f}, max {max(vals):.0f}. "
            "Bu qatorni prognoz asbobiga berib, trendni hisoblat.")

uid = re.sub(r"\W", "", str(st.query_params.get("u", "")))[:16]
if not uid:
    uid = uuid.uuid4().hex[:12]
    st.query_params["u"] = uid
CHAT_FILE = "chats_" + uid + ".json"

def load_chats():
    try:
        with open(CHAT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_chats():
    try:
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(S.chats, f, ensure_ascii=False)
    except Exception:
        pass

S = st.session_state
for _k, _v in (("cur", None), ("mode", "gen"), ("form", None), ("jadval", None), ("xlsx", None)):
    S.setdefault(_k, _v)
S.setdefault("chats", load_chats())

def reset(cur=None):
    """Joriy suhbat tanlovini va natija jadvallarini tozalaydi."""
    S.cur, S.form, S.jadval, S.xlsx = cur, None, None, None

def add(u, a):
    if not S.cur or S.cur not in S.chats:
        S.cur = uuid.uuid4().hex[:8]
        S.chats[S.cur] = {"title": u[:38], "mode": S.mode, "messages": []}
    S.chats[S.cur]["messages"] += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    save_chats()

def fm(x):
    return f"{x:,.0f}".replace(",", " ")

def tbl(head, rows):
    """Markdown jadval."""
    line = lambda r: "| " + " | ".join(map(str, r)) + " |"
    return "\n".join([line(head), "|" + "---|" * len(head)] + [line(r) for r in rows])

def to_excel(sheets):
    """{varaq nomi: [qatorlar]} -> xlsx baytlari"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()

# ====================== AI ======================
def ask_ai(prompt, mode, doc, hist, web_on, imgs=None):
    web, fc = "", mode == "fc"
    if (web_on or fc) and DDGS:  # prognoz rejimida internet doim yoqiladi
        try:
            with DDGS() as d:
                res = d.text(prompt, max_results=4 if fc else 2)
                web = "\n".join(f"{x.get('title', '')}: {x.get('body', '')} ({x.get('href', '')})" for x in res)
        except Exception:
            pass
    sys = SYSTEM[mode] + "\nCBU kurslari (so'm): " + str(cbu) + (fx_context() if fc else "")
    model, user_content = MODEL, prompt
    if imgs:  # rasm bo'lsa vision model ishlatiladi
        model = get_vision_model()
        try:
            user_content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": img_to_url(f)}} for f in imgs]
        except Exception as e:
            return "Rasmni o'qib bo'lmadi: " + str(e)
    if doc:
        sys += "\nYuklangan fayl:\n" + doc[:8000]
    if web:
        sys += "\nInternet (manba sifatida ko'rsat):\n" + web[:3500 if fc else 2000]
    msgs = ([{"role": "system", "content": sys}] + [{"role": x["role"], "content": x["content"]} for x in hist[-10:]]
            + [{"role": "user", "content": user_content}])
    temp = 0.4 if fc else 0.2
    try:
        for _ in range(4):
            r = client.chat.completions.create(model=model, messages=msgs, tools=TOOLS, tool_choice="auto", temperature=temp).choices[0].message
            if not r.tool_calls:
                return r.content
            msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in r.tool_calls]})
            msgs += [{"role": "tool", "tool_call_id": t.id, "content": call_tool(t.function.name, t.function.arguments)} for t in r.tool_calls]
        return "Hisob juda cho'zilib ketdi. Savolni soddaroq yozib ko'ring."
    except Exception:
        try:  # asbobsiz zaxira urinish: faqat oddiy xabarlar
            plain = [m for m in msgs if m["role"] in ("system", "user", "assistant") and not m.get("tool_calls")]
            return client.chat.completions.create(model=model, messages=plain, temperature=temp).choices[0].message.content
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                return "So'rovlar limiti to'ldi. Bir daqiqadan keyin qayta urinib ko'ring."
            return "Xato: " + str(e)

# ====================== TEZKOR HISOBLAGICHLAR ======================
def kredit_form():
    with st.form("kredit"):
        st.markdown("**🧮 Kredit hisobi**")
        c1, c2, c3 = st.columns(3)
        s = c1.number_input("Summa (so'm)", min_value=1_000_000, value=100_000_000, step=1_000_000)
        p = c2.number_input("Yillik foiz, %", min_value=0.0, max_value=100.0, value=24.0, step=0.5)
        n = c3.number_input("Muddat (oy)", min_value=1, max_value=360, value=36)
        ok = st.form_submit_button("Hisoblash", use_container_width=True)
    if ok:
        a, ar = _plan(float(s), float(p), int(n))
        d, dr = _plan(float(s), float(p), int(n), True)
        md = (f"**Annuitet bo'yicha oylik to'lov: {fm(a['oylik_tolov'])} so'm.**\n\n"
              + tbl(["", "Annuitet", "Differensial"], [
                  ("Oylik to'lov", fm(a["oylik_tolov"]), f"{fm(d['birinchi_oy_tolov'])} → {fm(d['oxirgi_oy_tolov'])}"),
                  ("Jami to'lov", fm(a["jami_tolov"]), fm(d["jami_tolov"])),
                  ("Jami foiz", fm(a["jami_foiz"]), fm(d["jami_foiz"]))])
              + f"\n\nDifferensial usulda foiz **{fm(a['jami_foiz'] - d['jami_foiz'])} so'm** kam, lekin dastlabki oylarda to'lov yuqori.\n\n"
              "Formula: A = S·r·(1+r)ⁿ / ((1+r)ⁿ − 1), r = yillik foiz / 12.")
        add(f"Kredit: {fm(s)} so'm, yillik {p}%, {n} oy", md)
        S.jadval = {"Annuitet jadvali": ar, "Differensial jadvali": dr}
        S.xlsx = to_excel({"Annuitet": ar, "Differensial": dr})
        S.form = None
        st.rerun()

def maosh_form():
    with st.form("maosh"):
        st.markdown("**🧾 Ish haqi solig'i**")
        b = st.number_input("Brutto maosh (so'm)", min_value=0, value=8_000_000, step=100_000)
        ok = st.form_submit_button("Hisoblash", use_container_width=True)
    if ok:
        r = ish_haqi(b)
        md = (f"**Qo'lga tegadigan maosh: {fm(r['qolga_tegadi'])} so'm.**\n\n"
              + tbl(["Ko'rsatkich", "Summa (so'm)"], [
                  ("Brutto", fm(r["brutto"])),
                  (f"JSHDS ({SOLIQ['JSHDS']:g}%)", fm(r["JSHDS"])),
                  (f"INPS ({SOLIQ['INPS']:g}%)", fm(r["INPS"])),
                  (f"Ijtimoiy soliq, ish beruvchi ({SOLIQ['Ijtimoiy soliq']:g}%)", fm(r["ijtimoiy_soliq_ish_beruvchi"])),
                  ("Ish beruvchi jami xarajati", fm(r["ish_beruvchi_jami_xarajat"]))])
              + "\n\nSoddalashtirilgan hisob: JSHDS va INPS bruttodan olindi.")
        add(f"Ish haqi solig'i: {fm(b)} so'm brutto", md)
        S.form = None
        st.rerun()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### 🏛️ Iqtisodchi AI")
    if st.button("➕ Yangi suhbat", use_container_width=True):
        reset()
        st.rerun()
    st.markdown("**Tezkor hisoblagichlar**")
    for label, key in (("🧮 Kredit hisobi", "kredit"), ("🧾 Ish haqi solig'i", "maosh")):
        if st.button(label, use_container_width=True):
            S.form = key
            st.rerun()
    web_on = st.checkbox("🌐 Internetdan qidirish", value=False)
    file = st.file_uploader("📎 PDF/Excel/CSV", type=["pdf", "xlsx", "csv"])
    for flag, code in (("🇺🇸", "USD"), ("🇪🇺", "EUR"), ("🇷🇺", "RUB")):
        st.metric(f"{flag} {code}", str(cbu.get(code)) + " so'm")
    with st.expander("📉 Kurs dinamikasi (30 kun)"):
        ccy = st.selectbox("Valyuta", ["USD", "EUR", "RUB"], label_visibility="collapsed")
        with st.spinner("CBU arxivi yuklanmoqda..."):
            h = get_history(ccy)
        if len(h) >= 2:
            st.line_chart(pd.DataFrame({ccy: [v for _, v in h]}, index=[d for d, _ in h]))
            st.caption(f"{h[0][0]:%d.%m} → {h[-1][0]:%d.%m}: {fm(h[0][1])} → {fm(h[-1][1])} so'm ({(h[-1][1] / h[0][1] - 1) * 100:+.2f}%)")
        else:
            st.caption("Kurs tarixi hozir yuklanmadi. Keyinroq urinib ko'ring.")
    st.markdown("**🕘 Suhbatlar**")
    for cid in reversed(list(S.chats)[-15:]):
        if st.button("💬 " + S.chats[cid]["title"], key="c_" + cid, use_container_width=True):
            S.mode = S.chats[cid].get("mode", "gen")
            reset(cid)
            st.rerun()
    if S.cur in S.chats and st.button("🗑️ Suhbatni o'chirish", use_container_width=True):
        del S.chats[S.cur]
        save_chats()
        S.cur, S.jadval, S.xlsx = None, None, None
        st.rerun()

doc = ""
if file:
    try:
        if file.name.endswith(".pdf"):
            doc = "".join(p.extract_text() or "" for p in PdfReader(file).pages)
        else:
            doc = (pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)).to_string()
    except Exception as e:
        st.sidebar.error("Faylni o'qib bo'lmadi: " + str(e))

# ====================== ASOSIY EKRAN ======================
MODES = (("gen", "🌐 Umumiy"), ("acc", "📒 Buxgalteriya"), ("fc", "🔮 Prognoz"))
for col, (key, label) in zip(st.columns(3), MODES):
    if col.button(label, use_container_width=True, key="m_" + key) and S.mode != key:
        S.mode = key
        reset()
        st.rerun()

# Bosh ekran: rejim -> (sarlavha, (tugma, amal), (tugma, amal)); amal "kredit"/"maosh" bo'lsa forma, aks holda savol matni
HOME = {
    "gen": ("Bugun qanday iqtisodiy tahlil qilamiz?",
            ("📊 Biznes tahlil", "Kichik biznes unit-ekonomikasini misol bilan hisobla."), ("🧮 Kredit masalasi", "kredit")),
    "acc": ("Buxgalteriya hisobi va soliq tahlili",
            ("📒 Provodkalar", "11.2 mln so'mga QQS bilan xomashyo olindi. BHMS provodkasini yoz."), ("🧾 Ish haqi solig'i", "maosh")),
    "fc": ("Iqtisodiy prognoz va kod",
           ("🔮 Yil prognozi", "Keyingi yil uchun O'zbekiston inflyatsiyasi va so'm kursi prognozini uch ssenariyda ber."),
           ("💻 Kod yozdir", "Streamlit'da oddiy kredit kalkulyatori uchun qisqa Python kodi yoz.")),
}
msgs = S.chats.get(S.cur, {}).get("messages", [])
sel = None

if not msgs:
    title, *btns = HOME[S.mode]
    st.markdown('<div class="hero"><div class="logo">' + LOGO + f'<div class="ltxt"><span style="color:{RED}">TOSHKENT</span><br>MENEJMENT VA<br>IQTISODIYOT<br>INSTITUTI</div></div><h3>' + title + '</h3></div>', unsafe_allow_html=True)
    for col, (label, act) in zip(st.columns(2), btns):
        if col.button(label, use_container_width=True):
            if act in ("kredit", "maosh"):
                S.form = act
                st.rerun()
            sel = act

if S.form == "kredit":
    kredit_form()
elif S.form == "maosh":
    maosh_form()

for m in msgs:
    with st.chat_message(m["role"], avatar=None if m["role"] == "user" else "🏛️"):
        st.markdown(m["content"])

if S.jadval:
    if S.xlsx:
        st.download_button("⬇️ Excelga yuklab olish", S.xlsx, file_name="kredit_jadvali.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    ar, dr = S.jadval.get("Annuitet jadvali"), S.jadval.get("Differensial jadvali")
    if ar and dr:
        with st.expander("📈 To'lov dinamikasi", expanded=True):
            idx = [x["Oy"] for x in ar]
            col = lambda rows, k: [x[k] for x in rows]
            st.markdown("**Qoldiq qarz (so'm)**")
            st.line_chart(pd.DataFrame({"Annuitet": col(ar, "Qoldiq"), "Differensial": col(dr, "Qoldiq")}, index=idx))
            st.markdown("**Annuitet: oylik to'lov tarkibi (asosiy qarz va foiz)**")
            st.bar_chart(pd.DataFrame({"Asosiy qarz": col(ar, "Asosiy qarz"), "Foiz": col(ar, "Foiz")}, index=idx))
            st.markdown("**Oylik to'lov taqqoslash (so'm)**")
            st.line_chart(pd.DataFrame({"Annuitet": col(ar, "To'lov"), "Differensial": col(dr, "To'lov")}, index=idx))
    for nom, rows in S.jadval.items():
        with st.expander("📋 " + nom):
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

try:  # 📎 tugmali kirish maydoni (Streamlit 1.43+)
    inp = st.chat_input("Savol yozing yoki rasm yuboring...", accept_file=True, file_type=["png", "jpg", "jpeg", "webp"])
except TypeError:
    inp = st.chat_input("Savol yoki masala yozing...")
imgs = []
if inp is not None and not isinstance(inp, str):
    imgs = list(getattr(inp, "files", None) or [])[:4]
    inp = (getattr(inp, "text", "") or "").strip() or ("Rasmni tahlil qil." if imgs else None)
prompt = sel or inp

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        for f in imgs:
            st.image(f, width=240)
    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Hisoblanmoqda..."):
            ans = ask_ai(prompt, S.mode, doc, msgs, web_on, imgs)
    add(prompt + (f"\n\n🖼 Rasm yuborildi: {len(imgs)} ta" if imgs else ""), ans)
    S.jadval, S.xlsx = None, None
    st.rerun()
    
