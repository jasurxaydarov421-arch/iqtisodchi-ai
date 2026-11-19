import os, json, re, requests
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from duckduckgo_search import DDGS

st.set_page_config(page_title="Iqtisodchi AI — Shaxsiy Yordamchi", page_icon="✴️", layout="centered")

# --- CLAUDE USLUBIDAGI ILIQ VA ZAMONAVIY DIZAYN (CSS) ---
st.markdown("""
<style>
    @import url('https://googleapis.com');

    /* Asosiy fon (Claude uslubidagi iliq sut rang) */
    .stApp {
        background-color: #F9F8F6;
        color: #1A1918;
        font-family: 'Inter', sans-serif;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    /* Yon panel (Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: #F2F0EC !important;
        border-right: 1px solid #E5E2DC !important;
    }
    /* Claude uslubidagi bosh sarlavha (Tepadan yetarli joy qoldirilgan) */
    .claude-hero {
        text-align: center;
        margin-top: 4.5rem;
        margin-bottom: 1.5rem;
    }
    .claude-icon {
        font-size: 2.4rem;
        color: #D97757;
        margin-bottom: 0.3rem;
    }
    .claude-title {
        font-size: 1.85rem;
        font-weight: 600;
        color: #1A1918;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }
    .claude-sub {
        font-size: 0.95rem;
        color: #6B6863;
    }
    .tool-badge {
        display: inline-block;
        background-color: #EFECE6;
        color: #5C5852;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 8px;
        border: 1px solid #E2DDD5;
    }
    /* Chat xabarlari dizayni */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 1rem 0 !important;
    }
    /* Foydalanuvchi xabari foni */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #EFECE6 !important;
        border-radius: 18px !important;
        padding: 12px 18px !important;
        margin-bottom: 12px !important;
    }
    /* Pastki yozish paneli */
    div[data-testid="stBottom"] {
        background-color: #F9F8F6 !important;
        border-top: none !important;
        padding-bottom: 14px !important;
    }
    div[data-testid="stChatInput"] {
        border-radius: 20px !important;
        border: 1px solid #DCD8D0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
        background-color: #FFFFFF !important;
        padding: 6px 10px !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #1A1918 !important;
    }
    /* "+" (Popover) tugmasi */
    div[data-testid="stPopover"] > button {
        border-radius: 16px !important;
        width: 46px !important;
        height: 46px !important;
        border: 1px solid #DCD8D0 !important;
        background-color: #FFFFFF !important;
        color: #D97757 !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04) !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stPopover"] > button:hover {
        border-color: #D97757 !important;
        background-color: #FFFBF9 !important;
    }
    /* Tezkor savol tugmalari */
    div.stButton > button {
        border-radius: 14px !important;
        border: 1px solid #E2DDD5 !important;
        background-color: #FFFFFF !important;
        color: #2D2B28 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #D97757 !important;
        color: #D97757 !important;
        background-color: #FFFDFB !important;
    }
</style>
""", unsafe_allow_html=True)

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

@st.cache_data(ttl=3600)
def get_model():
    for m in ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]:
        try:
            client.chat.completions.create(model=m, messages=[{"role": "user", "content": "1"}], max_tokens=1)
            return m
        except Exception:
            continue
    return "llama-3.1-8b-instant"

MODEL = get_model()

@st.cache_data(ttl=600)
def get_rates():
    try:
        r = requests.get("https://cbu.uz", timeout=5).json()
        return {i['Ccy']: i['Rate'] for i in r if i['Ccy'] in ['USD', 'EUR', 'RUB']}
    except Exception:
        return {"USD": "-", "EUR": "-", "RUB": "-"}

cbu = get_rates()

def web_search(q):
    try:
        with DDGS() as d:
            res = list(d.text(f"{q} iqtisodiyot", max_results=4))
            return "\n".join([f"- {x['title']}: {x['body']}" for x in res])
    except Exception:
        return ""

def parse_chart(text):
    m = re.search(r"```chart\s*(.*?)\s*```", text, re.DOTALL)
    clean = re.sub(r"```chart\s*.*?\s*```", "", text, flags=re.DOTALL).strip()
    fig = None
    if m:
        try:
            d = json.loads(m.group(1))
            df = pd.DataFrame({d.get("x_label", "X"): d["x"], d.get("y_label", "Y"): d["y"]})
            t = d.get("title", "Grafik")
            if d.get("type") == "line":
                fig = px.line(df, x=df.columns[0], y=df.columns[1], title=t, markers=True, color_discrete_sequence=["#D97757"])
            else:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=t, text_auto=True, color_discrete_sequence=["#D97757"])
            fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        except Exception:
            pass
    return clean, fig

# --- HOLATLAR ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_tool" not in st.session_state:
    st.session_state.active_tool = "Avtomatik (Umumiy iqtisodchi)"

# --- YON PANEL (CLAUDE SIDEBAR USLUBIDA) ---
with st.sidebar:
    st.markdown("### ✴️ Iqtisodchi AI")
    if st.button("➕ Yangi suhbat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("**Markaziy bank kurslari**")
    st.metric("🇺🇸 USD", f"{cbu.get('USD')} so'm")
    st.metric("🇪🇺 EUR", f"{cbu.get('EUR')} so'm")
    st.metric("🇷🇺 RUB", f"{cbu.get('RUB')} so'm")
    
    st.divider()
    st.caption("🔒 Shaxsiy va maxfiy chat. Bank kartasi yoki shaxsiy ma'lumotlar so'ralmaydi.")

# --- BOSH EKRAN (CHAT BO'SH BO'LGANDA) ---
selected_prompt = None
if len(st.session_state.messages) == 0:
    st.markdown(f"""
    <div class="claude-hero">
        <div class="claude-icon">✴️</div>
        <div class="claude-title">Bugun qanday iqtisodiy tahlil qilamiz?</div>
        <div class="claude-sub">Shaxsiy AI iqtisodchi va biznes tahlilchi yordamchingiz</div>
        <div class="tool-badge">Rejim: {st.session_state.active_tool}</div>
    </div>
    """, unsafe_allow_html=True)

    # Tezkor savol tugmalari
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📈 Bugungi valyuta kurslari tahlili", use_container_width=True):
            selected_prompt = "O'zbekistondagi bugungi USD, EUR va RUB kurslari hamda valyuta bozori holatini qisqacha tahlil qilib ber."
        if st.button("🏛️ Iqtisodiy ta'limotlar tarixi", use_container_width=True):
            selected_prompt = "Klassik iqtisodiy maktab va Keynschilik nazariyasi o'rtasidagi asosiy farqlarni jadval ko'rinishida tushuntir."
    with c2:
        if st.button("💼 Biznes reja va unit-ekonomika", use_container_width=True):
            selected_prompt = "Kichik biznes boshlashda unit-ekonomika va xarajatlarni qanday hisoblash kerak? Misol bilan tushuntir."
        if st.button("📊 Inflyatsiya va asosiy stavka", use_container_width=True):
            selected_prompt = "Markaziy bank asosiy stavkasi inflyatsiya va kredit foizlariga qanday ta'sir qiladi?"

# --- CHAT TARIXI ---
for i, m in enumerate(st.session_state.messages):
    avatar = "👤" if m["role"] == "user" else "✴️"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])
        if m.get("fig"):
            st.plotly_chart(m["fig"], use_container_width=True, key=f"c_{i}")

# --- PASTKI PANEL ("+" MENYU VA INPUT) ---
col_plus, col_input = st.columns([1, 7])

doc_info = ""
with col_plus:
    with st.popover("➕", use_container_width=True):
        st.markdown("**📎 Hujjat yuklash**")
        file = st.file_uploader("PDF, Excel yoki CSV", type=["pdf", "xlsx", "csv"], label_visibility="collapsed")
        if file:
            if file.name.endswith(".pdf"):
                doc_info = "".join([p.extract_text() for p in PdfReader(file).pages[:15]])[:6000]
                st.success(f"Yuklandi: {file.name}")
            else:
                df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
                doc_info = df.describe().to_string()
                st.success(f"Jadval yuklandi: {file.name}")
        
        st.divider()
        st.markdown("**🛠️ Iqtisodiy rejimlar**")
        if st.button("📜 Iqtisodiy ta'limotlar tarixi", use_container_width=True):
            st.session_state.active_tool = "Iqtisodiy ta'limotlar tarixi"
            st.rerun()
        if st.button("📊 Biznes tahlil va qurish", use_container_width=True):
            st.session_state.active_tool = "Biznes tahlil va qurish"
            st.rerun()
        if st.button("🌐 Umumiy iqtisodchi", use_container_width=True):
            st.session_state.active_tool = "Avtomatik (Umumiy iqtisodchi)"
            st.rerun()

with col_input:
    user_input = st.chat_input("Iqtisodchi AI ga savol yozing...")

prompt = selected_prompt if selected_prompt else user_input

# --- JAVOB GENERATSIYASI ---
if prompt:
                  
