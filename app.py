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

from hisob import SOLIQ, TOOLS, HisobXatosi, _ann, _diff, call_tool, ish_haqi, set_rates, tasdiqlanmagan
try:  # Mathematics AI: kutubxona yetishmasa ham qolgan rejimlar ishlayveradi
    import matematika
    MATH_XATO = ""
except ImportError as e:
    matematika, MATH_XATO = None, str(e)

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
set_rates(cbu)

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
        try:
            a, ar = _ann(float(s), float(p), int(n))
            d, dr = _diff(float(s), float(p), int(n))
        except HisobXatosi as e:
            st.error(str(e))
            return
        md = (f"**Annuitet bo'yicha oylik to'lov: {fm(a['oylik_tolov'])} so'm.**\n\n"
              "| | Annuitet | Differensial |\n|---|---|---|\n"
              f"| Oylik to'lov | {fm(a['oylik_tolov'])} | {fm(d['birinchi_oy_tolov'])} → {fm(d['oxirgi_oy_tolov'])} |\n"
              f"| Jami to'lov | {fm(a['jami_tolov'])} | {fm(d['jami_tolov'])} |\n"
              f"| Jami foiz | {fm(a['jami_foiz'])} | {fm(d['jami_foiz'])} |\n\n"
              f"Differensial usulda foiz **{fm(a['jami_foiz'] - d['jami_foiz'])} so'm** kam, lekin dastlabki oylarda to'lov yuqori.\n\n"
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
        try:
            r = ish_haqi(b)
        except HisobXatosi as e:
            st.error(str(e))
            return
        md = (f"**Qo'lga tegadigan maosh: {fm(r['qolga_tegadi'])} so'm.**\n\n"
              "| Ko'rsatkich | Summa (so'm) |\n|---|---|\n"
              f"| Brutto | {fm(r['brutto'])} |\n"
              f"| JSHDS ({SOLIQ['JSHDS']:g}%) | {fm(r['JSHDS'])} |\n"
              f"| INPS ({SOLIQ['INPS']:g}%) | {fm(r['INPS'])} |\n"
              f"| Ijtimoiy soliq, ish beruvchi ({SOLIQ['Ijtimoiy soliq']:g}%) | {fm(r['ijtimoiy_soliq_ish_beruvchi'])} |\n"
              f"| Ish beruvchi jami xarajati | {fm(r['ish_beruvchi_jami_xarajat'])} |\n\n"
              "Soddalashtirilgan hisob: JSHDS va INPS bruttodan olindi."
              + (" Diqqat: INPS stavkasi rasmiy manba bilan hali tasdiqlanmagan." if "INPS" in tasdiqlanmagan() else ""))
        add(f"Ish haqi solig'i: {fm(b)} so'm brutto", md)
        S.form = None
        st.rerun()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### 🏛️ Iqtisodchi AI")
    if st.button("➕ Yangi suhbat", use_container_width=True):
        S.cur, S.form, S.jadval, S.xlsx = None, None, None, None
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
            S.cur, S.mode, S.form, S.jadval, S.xlsx = cid, S.chats[cid].get("mode", "gen"), None, None, None
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
MODES = (("gen", "🌐 Umumiy"), ("acc", "📒 Buxgalteriya"), ("fc", "🔮 Prognoz"), ("math", "📐 Matematika"))
for col, (key, label) in zip(st.columns(len(MODES)), MODES):
    if col.button(label, use_container_width=True, key="m_" + key) and S.mode != key:
        S.mode, S.cur, S.form, S.jadval, S.xlsx = key, None, None, None, None
        st.rerun()

msgs = S.chats.get(S.cur, {}).get("messages", [])
sel = None

if not msgs:
    t = {"gen": "Bugun qanday iqtisodiy tahlil qilamiz?",
         "acc": "Buxgalteriya hisobi va soliq tahlili",
         "fc": "Iqtisodiy prognoz va kod",
         "math": "Mathematics AI"}[S.mode]
    st.markdown('<div class="hero"><div class="logo">' + LOGO + '<div class="ltxt"><span style="color:#FF3B22">TOSHKENT</span><br>MENEJMENT VA<br>IQTISODIYOT<br>INSTITUTI</div></div><h3>' + t + '</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if S.mode == "acc":
        if c1.button("📒 Provodkalar", use_container_width=True):
            sel = "11.2 mln so'mga QQS bilan xomashyo olindi. BHMS provodkasini yoz."
        if c2.button("🧾 Ish haqi solig'i", use_container_width=True):
            S.form = "maosh"
            st.rerun()
    elif S.mode == "fc":
        if c1.button("🔮 Yil prognozi", use_container_width=True):
            sel = "Keyingi yil uchun O'zbekiston inflyatsiyasi va so'm kursi prognozini uch ssenariyda ber."
        if c2.button("💻 Kod yozdir", use_container_width=True):
            sel = "Streamlit'da oddiy kredit kalkulyatori uchun qisqa Python kodi yoz."
    elif S.mode == "math":
        if c1.button("📐 Tenglama yechish", use_container_width=True):
            sel = "x^2 - 5x + 6 = 0 tenglamani yech."
        if c2.button("∫ Integral", use_container_width=True):
            sel = "x*exp(-x) funksiyasining 0 dan cheksizgacha integralini hisobla."
    else:
        if c1.button("📊 Biznes tahlil", use_container_width=True):
            sel = "Kichik biznes unit-ekonomikasini misol bilan hisobla."
        if c2.button("🧮 Kredit masalasi", use_container_width=True):
            S.form = "kredit"
            st.rerun()

if S.mode == "math":
    if matematika:
        math_level, math_answer_mode = matematika.panel(st)
    else:
        st.error("Mathematics AI yuklanmadi: " + MATH_XATO + ". `pip install -r requirements.txt` ni bajaring.")

if S.form == "kredit":
    kredit_form()
elif S.form == "maosh":
    maosh_form()

for m in msgs:
    with st.chat_message(m["role"], avatar=None if m["role"] == "user" else "🏛️"):
        st.markdown(m["content"])
if S.mode == "math" and matematika:
    matematika.show_chat_plot(st, msgs)

if S.jadval:
    if S.xlsx:
        st.download_button("⬇️ Excelga yuklab olish", S.xlsx, file_name="kredit_jadvali.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    ar, dr = S.jadval.get("Annuitet jadvali"), S.jadval.get("Differensial jadvali")
    if ar and dr:
        with st.expander("📈 To'lov dinamikasi", expanded=True):
            idx = [x["Oy"] for x in ar]
            st.markdown("**Qoldiq qarz (so'm)**")
            st.line_chart(pd.DataFrame({"Annuitet": [x["Qoldiq"] for x in ar],
                                        "Differensial": [x["Qoldiq"] for x in dr]}, index=idx))
            st.markdown("**Annuitet: oylik to'lov tarkibi (asosiy qarz va foiz)**")
            st.bar_chart(pd.DataFrame({"Asosiy qarz": [x["Asosiy qarz"] for x in ar],
                                       "Foiz": [x["Foiz"] for x in ar]}, index=idx))
            st.markdown("**Oylik to'lov taqqoslash (so'm)**")
            st.line_chart(pd.DataFrame({"Annuitet": [x["To'lov"] for x in ar],
                                        "Differensial": [x["To'lov"] for x in dr]}, index=idx))
    for nom, rows in S.jadval.items():
        with st.expander("📋 " + nom):
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

try:  # 📎 tugmali kirish maydoni (Streamlit 1.43+)
    inp = st.chat_input("Savol yozing yoki rasm yuboring...", accept_file=True,
                        file_type=["png", "jpg", "jpeg", "webp"])
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
            if S.mode != "math":
                ans = ask_ai(prompt, S.mode, doc, msgs, web_on, imgs)
            elif matematika:
                ans = matematika.chat_reply(st, client, prompt, math_level, math_answer_mode, msgs, imgs, doc, img_to_url)
            else:
                ans = "Mathematics AI yuklanmadi: " + MATH_XATO
    add(prompt + (f"\n\n🖼 Rasm yuborildi: {len(imgs)} ta" if imgs else ""), ans)
    S.jadval, S.xlsx = None, None
    st.rerun()
