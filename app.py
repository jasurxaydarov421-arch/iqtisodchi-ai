import json, math, os, re, uuid
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
    "3) oxirida ishlatilgan formula. Raqamlarni probel bilan ajrat (12 500 000 so'm). "
    "Aniq bilmagan narsani taxmin qilma, 'tekshirib ko'ring' de."
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
        return 1.0 if c == "UZS" else float(cbu[c])
    return {"natija": round(_f(summa) * k(kimdan) / k(kimga), 2), "manba": "CBU kursi"}

def soliq_stavkalari():
    return SOLIQ

FUNCS = {f.__name__: f for f in (annuitet_kredit, differensial_kredit, ish_haqi, qqs, murakkab_foiz,
                                 biznes_tahlil, valyuta_aylantirish, soliq_stavkalari)}

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

@st.cache_data(ttl=600)
def get_rates():
    try:
        r = requests.get("https://cbu.uz/oz/arkhiv-kursov-valyut/json/", timeout=5).json()
        return {i["Ccy"]: i["Rate"] for i in r if i.get("Ccy") in ("USD", "EUR", "RUB")}
    except Exception:
        return {"USD": "-", "EUR": "-", "RUB": "-"}

cbu = get_rates()

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

def add(u, a):
    if not S.cur or S.cur not in S.chats:
        S.cur = uuid.uuid4().hex[:8]
        S.chats[S.cur] = {"title": u[:38], "mode": S.mode, "messages": []}
    S.chats[S.cur]["messages"] += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    save_chats()

def fm(x):
    return f"{x:,.0f}".replace(",", " ")

# ====================== AI ======================
def ask_ai(prompt, mode, doc, hist, web_on):
    web = ""
    if web_on and DDGS:
        try:
            with DDGS() as d:
                web = "\n".join(x.get("body", "") for x in d.text(prompt, max_results=2))
        except Exception:
            pass
    sys = SYSTEM[mode] + "\nCBU kurslari (so'm): " + str(cbu)
    if doc:
        sys += "\nYuklangan fayl:\n" + doc[:8000]
    if web:
        sys += "\nInternet:\n" + web[:2000]
    msgs = [{"role": "system", "content": sys}] + [{"role": x["role"], "content": x["content"]} for x in hist[-10:]] + [{"role": "user", "content": prompt}]
    try:
        for _ in range(4):
            r = client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS, tool_choice="auto", temperature=0.2).choices[0].message
            if not r.tool_calls:
                return r.content
            msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in r.tool_calls]})
            for t in r.tool_calls:
                msgs.append({"role": "tool", "tool_call_id": t.id, "content": call_tool(t.function.name, t.function.arguments)})
        return "Hisob juda cho'zilib ketdi. Savolni soddaroq yozib ko'ring."
    except Exception:
        try:  # asbobsiz zaxira urinish
            return client.chat.completions.create(model=MODEL, messages=msgs[:1] + msgs[1:], temperature=0.2).choices[0].message.content
        except Exception as e:
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
              f"Differensial usulda foiz **{fm(a['jami_foiz'] - d['jami_foiz'])} so'm** kam, lekin dastlabki oylarda to'lov yuqori.\n\n"
              "Formula: A = S·r·(1+r)ⁿ / ((1+r)ⁿ − 1), r = yillik foiz / 12.")
        add(f"Kredit: {fm(s)} so'm, yillik {p}%, {n} oy", md)
        S.jadval, S.form = {"Annuitet jadvali": ar, "Differensial jadvali": dr}, None
        st.rerun()

def maosh_form():
    with st.form("maosh"):
        st.markdown("**🧾 Ish haqi solig'i**")
        b = st.number_input("Brutto maosh (so'm)", min_value=0, value=8_000_000, step=100_000)
        ok = st.form_submit_button("Hisoblash", use_container_width=True)
    if ok:
        r = ish_haqi(b)
        md = (f"**Qo'lga tegadigan maosh: {fm(r['qolga_tegadi'])} so'm.**\n\n"
              "| Ko'rsatkich | Summa (so'm) |\n|---|---|\n"
              f"| Brutto | {fm(r['brutto'])} |\n"
              f"| JSHDS ({SOLIQ['JSHDS']:g}%) | {fm(r['JSHDS'])} |\n"
              f"| INPS ({SOLIQ['INPS']:g}%) | {fm(r['INPS'])} |\n"
              f"| Ijtimoiy soliq, ish beruvchi ({SOLIQ['Ijtimoiy soliq']:g}%) | {fm(r['ijtimoiy_soliq_ish_beruvchi'])} |\n"
              f"| Ish beruvchi jami xarajati | {fm(r['ish_beruvchi_jami_xarajat'])} |\n\n"
              "Soddalashtirilgan hisob: JSHDS va INPS bruttodan olindi.")
        add(f"Ish haqi solig'i: {fm(b)} so'm brutto", md)
        S.form = None
        st.rerun()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### 🏛️ Iqtisodchi AI")
    if st.button("➕ Yangi suhbat", use_container_width=True):
        S.cur, S.form, S.jadval = None, None, None
        st.rerun()
    st.markdown("**Tezkor hisoblagichlar**")
    if st.button("🧮 Kredit hisobi", use_container_width=True):
        S.form = "kredit"
        st.rerun()
    if st.button("🧾 Ish haqi solig'i", use_container_width=True):
        S.form = "maosh"
        st.rerun()
    web_on = st.checkbox("🌐 Internetdan qidirish", value=False)
    file = st.file_uploader("📎 PDF/Excel/CSV", type=["pdf", "xlsx", "csv"])
    st.metric("🇺🇸 USD", str(cbu.get("USD")) + " so'm")
    st.metric("🇪🇺 EUR", str(cbu.get("EUR")) + " so'm")
    st.metric("🇷🇺 RUB", str(cbu.get("RUB")) + " so'm")
    st.markdown("**🕘 Suhbatlar**")
    for cid in reversed(list(S.chats)[-15:]):
        if st.button("💬 " + S.chats[cid]["title"], key="c_" + cid, use_container_width=True):
            S.cur, S.mode, S.form, S.jadval = cid, S.chats[cid].get("mode", "gen"), None, None
            st.rerun()
    if S.cur in S.chats and st.button("🗑️ Suhbatni o'chirish", use_container_width=True):
        del S.chats[S.cur]
        save_chats()
        S.cur, S.jadval = None, None
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
m1, m2 = st.columns(2)
if m1.button("🌐 Umumiy (Iqtisodiyot, Moliya, Biznes)", use_container_width=True) and S.mode != "gen":
    S.mode, S.cur, S.form, S.jadval = "gen", None, None, None
    st.rerun()
if m2.button("📒 Buxgalteriya hisobi (BHMS)", use_container_width=True) and S.mode != "acc":
    S.mode, S.cur, S.form, S.jadval = "acc", None, None, None
    st.rerun()

is_acc = S.mode == "acc"
msgs = S.chats.get(S.cur, {}).get("messages", [])
sel = None

if not msgs:
    t = "Buxgalteriya hisobi va soliq tahlili" if is_acc else "Bugun qanday iqtisodiy tahlil qilamiz?"
    st.markdown('<div class="hero"><div class="logo">' + LOGO + '<div class="ltxt"><span style="color:#FF3B22">TOSHKENT</span><br>MENEJMENT VA<br>IQTISODIYOT<br>INSTITUTI</div></div><h3>' + t + '</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("📒 Provodkalar" if is_acc else "📊 Biznes tahlil", use_container_width=True):
        sel = "11.2 mln so'mga QQS bilan xomashyo olindi. BHMS provodkasini yoz." if is_acc else "Kichik biznes unit-ekonomikasini misol bilan hisobla."
    if c2.button("🧾 Ish haqi solig'i" if is_acc else "🧮 Kredit masalasi", use_container_width=True):
        S.form = "maosh" if is_acc else "kredit"
        st.rerun()

if S.form == "kredit":
    kredit_form()
elif S.form == "maosh":
    maosh_form()

for m in msgs:
    with st.chat_message(m["role"], avatar=None if m["role"] == "user" else "🏛️"):
        st.markdown(m["content"])

if S.jadval:
    for nom, rows in S.jadval.items():
        with st.expander("📋 " + nom):
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

inp = st.chat_input("Savol yoki masala yozing...")
prompt = sel or inp

if prompt:
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Hisoblanmoqda..."):
            ans = ask_ai(prompt, S.mode, doc, msgs, web_on)
    add(prompt, ans)
    S.jadval = None
    st.rerun()
    
