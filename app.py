import json, re, requests
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from duckduckgo_search import DDGS

st.set_page_config(page_title="Iqtisodchi AI", page_icon="✴️", layout="centered")

st.markdown("""
<style>
    @import url('https://googleapis.com');
    .stApp { background-color: #F9F8F6; color: #1A1918; font-family: 'Inter', sans-serif; }
    header[data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] { background-color: #F2F0EC !important; border-right: 1px solid #E5E2DC !important; }
    .claude-hero { text-align: center; margin-top: 4.5rem; margin-bottom: 1.5rem; }
    .claude-icon { font-size: 2.3rem; color: #D97757; }
    .claude-title { font-size: 1.8rem; font-weight: 600; color: #1A1918; margin: 0.3rem 0; }
    .claude-sub { font-size: 0.95rem; color: #6B6863; }
    .tool-badge { display: inline-block; background: #EFECE6; color: #5C5852; padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; margin-top: 8px; border: 1px solid #E2DDD5; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) { background-color: #EFECE6 !important; border-radius: 18px !important; padding: 12px 18px !important; }
    div[data-testid="stBottom"] { background-color: #F9F8F6 !important; border-top: none !important; }
    div[data-testid="stChatInput"] { border-radius: 20px !important; border: 1px solid #DCD8D0 !important; background-color: #FFFFFF !important; }
    div[data-testid="stPopover"] > button { border-radius: 16px !important; width: 46px !important; height: 46px !important; border: 1px solid #DCD8D0 !important; background-color: #FFFFFF !important; color: #D97757 !important; }
    div.stButton > button { border-radius: 14px !important; border: 1px solid #E2DDD5 !important; background-color: #FFFFFF !important; color: #2D2B28 !important; font-size: 0.88rem !important; }
    div.stButton > button:hover { border-color: #D97757 !important; color: #D97757 !important; }
</style>
""", unsafe_allow_html=True)

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

@st.cache_data(ttl=3600)
def get_model():
    for m in ("llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"):
        try:
            client.chat.completions.create(model=m, messages=[{"role": "user", "content": "1"}], max_tokens=1)
            return m
        except Exception:
            pass
    return "llama-3.1-8b-instant"

MODEL = get_model()

@st.cache_data(ttl=600)
def get_rates():
    try:
        r = requests.get("https://cbu.uz", timeout=5).json()
        return {i.get("Ccy"): i.get("Rate", "-") for i in r if i.get("Ccy") in ("USD", "EUR", "RUB")}
    except Exception:
        return {"USD": "-", "EUR": "-", "RUB": "-"}

cbu = get_rates()

def web_search(q):
    try:
        with DDGS() as d:
            return "\n".join([f"- {x.get('title', '')}: {x.get('body', '')}" for x in d.text(f"{q} iqtisodiyot", max_results=3)])
    except Exception:
        return ""

def parse_chart(text):
    m = re.search(r"```chart\s*(.*?)\s*```", text, re.DOTALL)
    clean = re.sub(r"```chart\s*.*?\s*```", "", text, flags=re.DOTALL).strip()
    fig = None
    if m:
        try:
            d = json.loads(m.group(1))
            xl, yl, t = d.get("x_label", "X"), d.get("y_label", "Y"), d.get("title", "Grafik")
            df = pd.DataFrame({xl: d.get("x"), yl: d.get("y")})
            fig = px.line(df, x=xl, y=yl, title=t, markers=True, color_discrete_sequence=["#D97757"]) if d.get("type") == "line" else px.bar(df, x=xl, y=yl, title=t, text_auto=True, color_discrete_sequence=["#D97757"])
            fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        except Exception:
            pass
    return clean, fig

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_tool" not in st.session_state:
    st.session_state.active_tool = "Avtomatik (Umumiy iqtisodchi)"

with st.sidebar:
    st.markdown("### ✴️ Iqtisodchi AI")
    if st.button("➕ Yangi suhbat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.metric("🇺🇸 USD", f"{cbu.get('USD', '-')} so'm")
    st.metric("🇪🇺 EUR", f"{cbu.get('EUR', '-')} so'm")
    st.metric("🇷🇺 RUB", f"{cbu.get('RUB', '-')} so'm")
    st.divider()
    st.caption("🔒 Shaxsiy va maxfiy chat. Bank kartasi yoki shaxsiy ma'lumotlar so'ralmaydi.")

selected_prompt = None
if not st.session_state.messages:
    st.markdown(f'<div class="claude-hero"><div class="claude-icon">✴️</div><div class="claude-title">Bugun qanday iqtisodiy tahlil qilamiz?</div><div class="claude-sub">Shaxsiy AI iqtisodchi va biznes tahlilchi yordamchingiz</div><div class="tool-badge">Rejim: {st.session_state.active_tool}</div></div>', unsafe_allow_html=True)
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

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"], avatar="👤" if m["role"] == "user" else "✴️"):
        st.markdown(m["content"])
        if m.get("fig"):
            st.plotly_chart(m["fig"], use_container_width=True, key=f"c_{i}")

col_plus, col_input = st.columns((1, 7))
doc_info = ""
with col_plus:
    with st.popover("➕", use_container_width=True):
        st.markdown("**📎 Hujjat yuklash**")
        file = st.file_uploader("PDF, Excel yoki CSV", type=["pdf", "xlsx", "csv"], label_visibility="collapsed")
        if file:
            if file.name.endswith(".pdf"):
                doc_info = "".join([p.extract_text() or "" for p in PdfReader(file).pages])
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

if prompt:
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="✴️"):
        with st.spinner("Tahlil qilinmoqda..."):
            try:
                sys_prompt = f"Sen Claude AI uslubidagi professional AI iqtisodchi-ekspertsan. Rejim: {st.session_state.active_tool}. Faqat ravon o'zbek tilida javob ber, asosiy xulosani **bold** qilib ber, sarlavha va jadvallardan foydalan. Grafik kerak bo'lsa oxirida ```chart blokida JSON yoz (type, title, x_label, y_label, x, y). CBU: {cbu}. Fayl: {doc_info}. Internet: {web_search(prompt)}"
                msgs = [{"role": "system", "content": sys_prompt}] + [{"role": x["role"], "content": x["content"]} for x in st.session_state.messages] + [{"role": "user", "content": prompt}]
                a_res = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.5)
                ans = next(iter(a_res.choices)).message.content
                clean, fig = parse_chart(ans)
                st.markdown(clean)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="new_c")
                st.session_state.messages.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": clean, "fig": fig}])
                if selected_prompt:
                    st.rerun()
            except Exception as e:
                st.error(f"Xatolik: {e}")
            
