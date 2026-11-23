import base64, io, json, math, os, re, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import pandas as pd
import requests
import streamlit as st
from groq import Groq
from pypdf import PdfReader

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

st.set_page_config(page_title="Iqtisodchi AI", page_icon="🏛️", layout="centered")
st.markdown("<style>.stApp{background:#F9F8F6;color:#1F1F1F}section[data-testid='stSidebar']{background:#F2F0EC!important}.hero{text-align:center;margin:1rem 0}.logo{display:inline-flex;align-items:center;gap:12px;text-align:left;margin-bottom:10px}.ltxt{font-weight:700;font-size:15px;line-height:1.15;color:#16354D}div[data-testid='stChatMessage']:has(div[data-testid='stChatMessageAvatarUser']){flex-direction:row-reverse;background:#EFECE6!important;border-radius:18px!important;padding:10px 16px!important;margin-left:auto!important;max-width:82%}div[data-testid='stChatMessage']:has(div[data-testid='stChatMessageAvatarUser']) div[data-testid='stChatMessageAvatarUser']{display:none}div.stButton>button{border-radius:14px!important;background:#FFF!important}</style>", unsafe_allow_html=True)

LOGO = '<div style="width:60px;height:60px;position:relative;flex-shrink:0"><div style="position:absolute;top:0;left:0;width:60px;height:10px;background:#16354D"></div><div style="position:absolute;top:0;left:0;width:10px;height:60px;background:#16354D"></div><div style="position:absolute;top:14px;left:14px;width:9px;height:24px;background:#FF3B22"></div><div style="position:absolute;top:14px;left:26px;width:9px;height:30px;background:#FF3B22"></div><div style="position:absolute;top:14px;left:38px;width:9px;height:17px;background:#FF3B22"></div><div style="position:absolute;top:14px;left:50px;width:10px;height:9px;background:#FF3B22"></div><div style="position:absolute;top:47px;left:26px;width:9px;height:13px;background:#16354D"></div><div style="position:absolute;top:34px;left:38px;width:9px;height:26px;background:#16354D"></div><div style="position:absolute;top:26px;left:50px;width:10px;height:34px;background:#16354D"></div></div>'

# ====================== SOZLAMALAR ======================
# Soliq stavkalari (%). Qonun o'zgarsa, faqat shu yerni yangilang.
# DIQQAT: INPS (xodim) stavkasini lex.uz dan tekshirib qo'ying.
SOLIQ = {"JSHDS": 12.0, "INPS": 4.0, "Ijtimoiy soliq": 12.0, "QQS": 12.0,
         "Foyda solig'i": 15.0, "Aylanma solig'i": 4.0}

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

def _ann(s, p, n):
    r = p / 1200
    t = s / n if r == 0 else s * r * (1 + r) ** n / ((1 + r) ** n - 1)
    q, rows = s, []
    for i in range(1, n + 1):
        f = q * r
        q -= t - f
        rows.append({"Oy": i, "To'lov": round(t), "Asosiy qarz": round(t - f), "Foiz": round(f), "Qoldiq": round(max(q, 0))})
    return {"oylik_tolov": round(t), "jami_tolov": round(t * n), "jami_foiz": round(t * n - s)}, rows

def _diff(s, p, n):
    r, a, q, rows = p / 1200, s / n, s, []
    for i in range(1, n + 1):
        f = q * r
        rows.append({"Oy": i, "To'lov": round(a + f), "Asosiy qarz": round(a), "Foiz": round(f), "Qoldiq": round(max(q - a, 0))})
        q -= a
    jf = sum(x["Foiz"] for x in rows)
    return {"birinchi_oy_tolov": rows[0]["To'lov"], "oxirgi_oy_tolov": rows[-1]["To'lov"], "jami_tolov": round(s + jf), "jami_foiz": jf}, rows

def annuitet_kredit(summa, yillik_foiz, oy):
    return _ann(_f(summa), _f(yillik_foiz), int(_f(oy)))[0]

def differensial_kredit(summa, yillik_foiz, oy):
    return _diff(_f(summa), _f(yillik_foiz), int(_f(oy)))[0]

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
    marja = n - v
    daromad = n * q
    foyda = marja * q - d
    return {"birlik_marjasi": round(marja), "marja_foizi": round(marja / n * 100, 1) if n else None,
            "zararsizlik_soni": math.ceil(d / marja) if marja > 0 else None,
            "umumiy_daromad": round(daromad), "sof_foyda": round(foyda),
            "rentabellik_foizi": round(foyda / daromad * 100, 1) if daromad else None}

def valyuta_aylantirish(summa, kimdan, kimga):
    def k(c):
        c = c.upper()
        if c == "UZS":
            return 1.0
        try:
            return float(cbu[c])
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
    _tool("qqs", "QQS hisobi. qqs_bilan=true bo'lsa summa QQS bilan berilgan (ajratib olinadi)",
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

@st.cache_data(ttl=3600)
def get_model():
    try:
        models = [m.id for m in client.models.list().data if not any(x in m.id for x in ("whisper", "guard", "tts"))]
        for pref in ("llama-3.3-70b", "gpt-oss", "llama-4", "qwen", "llama-3.1-70b", "llama-3.1-8b"):
            for m in models:
                if pref in m:
                    return m
        if models:
            return models[0]
    except Exception:
        pass
    return "llama-3.3-70b-versatile"

MODEL = get_model()

@st.cache_data(ttl=3600)
def get_vision_model():
    """Rasm tushunadigan modelni Groq ro'yxatidan topadi."""
    try:
        ids = [m.id for m in client.models.list().data]
        for pref in ("llama-4-scout", "llama-4-maverick", "vision"):
            for m in ids:
                if pref in m:
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
S.setdefault("chats", load_chats())
S.setdefault("cur", None)
S.setdefault("mode", "gen")
S.setdefault("form", None)
S.setdefault("jadval", None)
S.setdefault("xlsx", None)

def add(u, a):
    if not S.cur or S.cur not in S.chats:
        S.cur = uuid.uuid4().hex[:8]
        S.chats[S.cur] = {"title": u[:38], "mode": S.mode, "messages": []}
    S.chats[S.cur]["messages"] += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    save_chats()

def fm(x):
    return f"{x:,.0f}".replace(",", " ")

def to_excel(sheets):
    """{varaq nomi: [qatorlar]} -> xlsx baytlari"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()

# ====================== AI ======================
def ask_ai(prompt, mode, doc, hist, web_on, imgs=None):
    web = ""
    fc = mode == "fc"
    if (web_on or fc) and DDGS:  # prognoz rejimida internet doim yoqiladi
        try:
            with DDGS() as d:
                res = d.text(prompt, max_results=4 if fc else 2)
                web = "\n".join(f"{x.get('title', '')}: {x.get('body', '')} ({x.get('href', '')})" for x in res)
        except Exception:
            pass
    sys = SYSTEM[mode] + "\nCBU kurslari (so'm): " + str(cbu)
    if fc:
        sys += fx_context()
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
    msgs = [{"role": "system", "content": sys}] + [{"role": x["role"], "content": x["content"]} for x in hist[-10:]] + [{"role": "user", "content": user_content}]
    temp = 0.4 if fc else 0.2
    try:
        for _ in range(4):
            r = client.chat.completions.create(model=model, messages=msgs, tools=TOOLS, tool_choice="auto", temperature=temp).choices[0].message
            if not r.tool_calls:
                return r.content
            msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in r.tool_calls]})
            for t in r.tool_calls:
                msgs.append({"role": "tool", "tool_call_id": t.id, "content": call_tool(t.function.name, t.function.arguments)})
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
        a, ar = _ann(float(s), float(p), int(n))
        d, dr = _diff(float(s), float(p), int(n))
        md = (f"**Annuitet bo'yicha oylik to'lov: {fm(a['oylik_tolov'])} so'm.**\n\n"
              "| | Annuitet | Differensial |\n|---|---|---|\n"
              f"| Oylik to'lov | {fm(a['oylik_tolov'])} | {fm(d['birinchi_oy_tolov'])} → {fm(d['oxirgi_oy_tolov'])} |\n"
              f"| Jami to'lov | {fm(a['jami_tolov'])} | {fm(d['jami_tolov'])} |\n"
              f"| Jami foiz | {fm(a['jami_foiz'])} | {fm(d['jami_foiz'])} |\n\n"
              f"Differensial usulda foiz **{fm(a['jami_foiz'] -
