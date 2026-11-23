import json, math, re, requests
from duckduckgo_search import DDGS
from groq import Groq
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
import streamlit as st

st.set_page_config(page_title="Iqtisodchi AI", page_icon="🏛️", layout="centered")
st.markdown("<style>.stApp{background:#F9F8F6;color:#1F1F1F}section[data-testid='stSidebar']{background:#F2F0EC!important}.hero{text-align:center;margin:1rem 0}.logo{display:inline-flex;align-items:center;gap:12px;text-align:left;margin-bottom:10px}.ltxt{font-weight:700;font-size:15px;line-height:1.15;color:#16354D}div[data-testid='stChatMessage']:has(div[data-testid='stChatMessageAvatarUser']){flex-direction:row-reverse;background:#EFECE6!important;border-radius:18px!important;padding:10px 16px!important;margin-left:auto!important;max-width:82%}div[data-testid='stChatMessage']:has(div[data-testid='stChatMessageAvatarUser']) div[data-testid='stChatMessageAvatarUser']{display:none}div.stButton>button{border-radius:14px!important;background:#FFF!important}</style>", unsafe_allow_html=True)

LOGO = '<div style="width:60px;height:60px;position:relative;flex-shrink:0"><div style="position:absolute;top:0;left:0;width:60px;height:10px;background:#16354D"></div><div style="position:absolute;top:0;left:0;width:10px;height:60px;background:#16354D"></div><div style="position:absolute;top:14px;left:14px;width:9px;height:24px;background:#FF3B22"></div><div style="position:absolute;top:14px;left:26px;width:9px;height:30px;background:#FF3B22"></div><div style="position:absolute;top:14px;left:38px;width:9px;height:17px;background:#FF3B22"></div><div style="position:absolute;top:14px;left:50px;width:10px;height:9px;background:#FF3B22"></div><div style="position:absolute;top:47px;left:26px;width:9px;height:13px;background:#16354D"></div><div style="position:absolute;top:34px;left:38px;width:9px;height:26px;background:#16354D"></div><div style="position:absolute;top:26px;left:50px;width:10px;height:34px;background:#16354D"></div></div>'

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

@st.cache_data(ttl=600)
def get_rates():
    try:
        r = requests.get("https://cbu.uz", timeout=5).json()
        return {i["Ccy"]: i["Rate"] for i in r if i.get("Ccy") in ("USD", "EUR", "RUB")}
    except Exception:
        return {"USD": "-", "EUR": "-", "RUB": "-"}

cbu = get_rates()

def run_calc(text):
    out = []
    for c in re.findall(r"```calc\s*(.*?)\s*```", text, re.DOTALL):
        try:
            loc = {}
            exec(c, {"__builtins__": {}}, {"math": math, "round": round, "sum": sum, "pow": pow, **loc})
            out.append(str(loc.get("natija", "")))
        except Exception:
            pass
    return re.sub(r"```calc\s*.*?\s*```", "", text, flags=re.DOTALL).strip(), ", ".join(out)

def ask_ai(prompt, is_acc, doc, hist):
    try:
        with DDGS() as d:
            web = "\n".join([x.get("body", "") for x in d.text(prompt, max_results=2)])
    except Exception:
        web = ""
    role = "TMII Bosh Buxgalter AI (O'zbekiston 21-son BHMS, Debet/Kredit jadvali, Soliq kodeksi)." if is_acc else "TMII Iqtisodchi, Biznes tahlilchi va Moliyachi AI."
    sys = role + " Ravon o'zbek tilida yoz. Masala berilsa DARHOL avval tushuntirib ishla, birinchi jumlada natijani **bold** qil, oxirida formulani ko'rsat. Murakkab hisob uchun ```calc blokida Python yoz (natija = ...). CBU: " + str(cbu) + ". Fayl: " + doc + ". Web: " + web
    msgs = [{"role": "system", "content": sys}] + [{"role": x["role"], "content": x["content"]} for x in hist] + [{"role": "user", "content": prompt}]
    try:
        ans = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.2).choices[0].message.content
        clean, calc = run_calc(ans)
        if calc:
            msgs += [{"role": "assistant", "content": ans}, {"role": "user", "content": "Aniq hisob: " + calc + ". Tushuntirilgan yechim va formulani yoz."}]
            clean = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, temperature=0.2).choices[0].message.content
        return clean
    except Exception as e:
        return "Xato: " + str(e)

if "messages" not in st.session_state: st.session_state.messages = []
if "mode" not in st.session_state: st.session_state.mode = "gen"

with st.sidebar:
    st.markdown("### 🏛️ Iqtisodchi AI")
    if st.button("➕ Yangi suhbat", use_container_width=True): st.session_state.messages = []; st.rerun()
    st.metric("🇺🇸 USD", str(cbu.get("USD")) + " so'm")
    st.metric("🇪🇺 EUR", str(cbu.get("EUR")) + " so'm")
    st.metric("🇷🇺 RUB", str(cbu.get("RUB")) + " so'm")
    file = st.file_uploader("📎 PDF/Excel/CSV", type=["pdf", "xlsx", "csv"])

doc = ""
if file:
    doc = "".join([p.extract_text() or "" for p in PdfReader(file).pages]) if file.name.endswith(".pdf") else (pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)).to_string()

m1, m2 = st.columns(2)
if m1.button("🌐 Umumiy (Iqtisodiyot, Moliya, Biznes)", use_container_width=True): st.session_state.mode = "gen"; st.rerun()
if m2.button("📒 Buxgalteriya hisobi (BHMS)", use_container_width=True): st.session_state.mode = "acc"; st.rerun()

is_acc = st.session_state.mode == "acc"
sel = None

if not st.session_state.messages:
    t = "Buxgalteriya hisobi va soliq tahlili" if is_acc else "Bugun qanday iqtisodiy tahlil qilamiz?"
    st.markdown('<div class="hero"><div class="logo">' + LOGO + '<div class="ltxt"><span style="color:#FF3B22">TOSHKENT</span><br>MENEJMENT VA<br>IQTISODIYOT<br>INSTITUTI</div></div><h3>' + t + '</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("📒 Provodkalar" if is_acc else "📊 Biznes tahlil", use_container_width=True): sel = "11.2 mln so'mga QQS bilan xomashyo olindi. BHMS provodkasini yoz." if is_acc else "Kichik biznes unit-ekonomikasini misol bilan hisobla."
    if c2.button("🧾 Ish haqi solig'i" if is_acc else "🧮 Kredit masalasi", use_container_width=True): sel = "8 mln so'm maoshdan JSHODS, INPS va ijtimoiy soliqni hisobla." if is_acc else "100 mln so'm kredit 24% stavkada 3 yilga annuitet oylik to'lovini hisobla."

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=None if m["role"] == "user" else "🏛️"): st.markdown(m["content"])

inp = st.chat_input("Savol yoki masala yozing...")
prompt = sel if sel else inp

if prompt:
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Hisoblanmoqda..."):
            ans = ask_ai(prompt, is_acc, doc, st.session_state.messages)
            st.markdown(ans)
            st.session_state.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}]
            if sel: st.rerun()
    
