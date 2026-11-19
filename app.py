import os, json, re, requests
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from duckduckgo_search import DDGS

st.set_page_config(page_title="SALOM MEN AI AGENT", page_icon="✨", layout="centered")

# --- 100% SKRINSHOTDAGI KABI OQ VA YUMALOQ MOBIL DIZAYN (CSS) ---
st.markdown("""
<style>
    /* Asosiy fon va Streamlit tepa panelini yashirish */
    .stApp {
        background-color: #ffffff;
        color: #1f1f1f;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    /* Bosh sarlavha */
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 3rem;
        margin-bottom: 0.4rem;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #5f6368;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Pastki suzuvchi panel (Savol bering + "+" tugmasi) */
    div[data-testid="stBottom"] {
        background-color: #ffffff !important;
        border-top: none !important;
        padding-bottom: 12px !important;
    }
    div[data-testid="stChatInput"] {
        border-radius: 28px !important;
        border: 1px solid #dfe1e5 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
        background-color: #ffffff !important;
        padding: 4px 8px !important;
    }
    /* "+" (Popover) tugmasini yumaloq qilish */
    div[data-testid="stPopover"] > button {
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        border: 1px solid #dfe1e5 !important;
        background-color: #f8f9fa !important;
        color: #1f1f1f !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    /* Menyu ichidagi tugmalar */
    div[data-testid="stPopoverBody"] button {
        width: 100%;
        border-radius: 14px !important;
        border: none !important;
        background-color: #f1f3f4 !important;
        color: #1f1f1f !important;
        text-align: left !important;
        padding: 12px 16px !important;
        margin-bottom: 6px !important;
        font-weight: 500 !important;
    }
    div[data-testid="stPopoverBody"] button:hover {
        background-color: #e8eaed !important;
    }
</style>
""", unsafe_allow_html=True)

# --- GOOGLE AKKAUNT ORQALI KIRISH (SOZLANGAN BO'LSA) ---
has_google_auth = "auth" in st.secrets and "google" in st.secrets["auth"]
if has_google_auth and not st.user.is_logged_in:
    st.markdown('<div class="hero-title">SALOM MEN AI AGENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Shaxsiy va maxfiy chatni boshlash uchun Google akkauntingiz bilan kiring</div>', unsafe_allow_html=True)
    if st.button("🔐 Google akkaunt orqali kirish"):
        st.login("google")
    st.stop()

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
        return {"USD": "-"}

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
            fig = px.line(df, x=df.columns, y=df.columns, title=t, markers=True) if d.get("type") == "line" else px.bar(df, x=df.columns, y=df.columns, title=t, text_auto=True)
            fig.update_layout(template="plotly_white")
        except Exception:
            pass
    return clean, fig

# --- HOLATLAR (HAR BIR FOYDALANUVCHI UCHUN ALOHIDA MAXFIY XOTIRA) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_tool" not in st.session_state:
    st.session_state.active_tool = "Avtomatik (Umumiy iqtisodchi)"

# --- YON PANEL (YANGI CHAT VA KURSLAR) ---
with st.sidebar:
    st.title("✨ AI AGENT")
    if has_google_auth and st.user.is_logged_in:
        st.caption(f"👤 {st.user.name}")
        if st.button("🚪 Chiqish"):
            st.logout()
    else:
        st.caption("🔒 Shaxsiy va maxfiy chat (boshqalarga ko'rinmaydi)")
    if st.button("➕ Yangi chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.metric("USD (Markaziy bank)", f"{cbu.get('USD')} so'm")

# --- BOSH EKRAN SARLAVHASI ---
if len(st.session_state.messages) == 0:
    st.markdown('<div class="hero-title">SALOM MEN AI AGENT</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">Faol vosita: <b>{st.session_state.active_tool}</b></div>', unsafe_allow_html=True)

# --- CHAT TARIXI ---
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("fig"):
            st.plotly_chart(m["fig"], use_container_width=True, key=f"c_{i}")

# --- PASTKI "+" MENYU VA "SAVOL BERING" PANELI ---
col_plus, col_input = st.columns([1, 7])

doc_info = ""
with col_plus:
    with st.popover("➕", use_container_width=True):
        st.markdown("**Fayllar va Hujjatlar**")
        file = st.file_uploader("📎 PDF, Excel yoki CSV yuklash", type=["pdf", "xlsx", "csv"], label_visibility="collapsed")
        if file:
            if file.name.endswith(".pdf"):
                doc_info = "".join([p.extract_text() for p in PdfReader(file).pages[:15]])[:6000]
                st.success(f"Yuklandi: {file.name}")
            else:
                df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
                doc_info = df.describe().to_string()
                st.success(f"Jadval yuklandi: {file.name}")
        
        st.divider()
        st.markdown("**Vositalar**")
        if st.button("📜 Iqtisodiy ta'limotlar tarixi"):
            st.session_state.active_tool = "Iqtisodiy ta'limotlar tarixi"
            st.rerun()
        if st.button("📊 Biznes tahlil va qurish"):
            st.session_state.active_tool = "Biznes tahlil va qurish"
            st.rerun()
        if st.button("🌐 Umumiy iqtisodchi (Barchasi)"):
            st.session_state.active_tool = "Avtomatik (Umumiy iqtisodchi)"
            st.rerun()

with col_input:
    prompt = st.chat_input("Savol bering")

# --- JAVOB GENERATSIYASI ---
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Tahlil qilinmoqda..."):
            try:
                tool = st.session_state.active_tool
                web = web_search(prompt)
                
                sys = f"""Sen Google Gemini uslubidagi professional AI iqtisodchi-ekspertsan. Hozirgi tanlangan vosita: {tool}.
                Qoidalar:
                1. Faqat ravon va adabiy o'zbek tilida javob ber.
                2. Birinchi gapdayoq savolning eng asosiy javobini yoki xulosasini **qalin shrift (bold)** bilan aniq ber.
                3. Ma'lumotlarni kichik sarlavhalar (###), punktlar (*) va taqqoslash jadvallari (Markdown Table) ko'rinishida tuzilmali ber.
                4. Agar vosita "Iqtisodiy ta'limotlar tarixi" bo'lsa: iqtisodiy maktablar (merkantilizm, klassik, keynschilik, monetarizm) va mutafakkirlar ta'limotini chuqur tahlil qil.
                5. Agar vosita "Biznes tahlil va qurish" bo'lsa: noldan biznes qurish, unit-ekonomika, xarajat-foyda tahlili va amaliy strategiya ber.
                6. Agar raqamlarni taqqoslash yoki grafik chizish kerak bo'lsa, javob oxirida aniq shu formatda JSON qo'sh:
                ```chart
                {{"type":"bar","title":"Sarlavha","x_label":"X","y_label":"Y","x":["A","B"],"y":}}
                ```
                Qo'shimcha ma'lumotlar: CBU kurslari: {cbu}, Fayl: {doc_info}, Internet ma'lumotlari: {web}"""

                msgs = [{"role": "system", "content": sys}] + [{"role": x["role"], "content": x["content"]} for x in st.session_state.messages[-8:]] + [{"role": "user", "content": prompt}]
                a_res = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.5)
                ans = a_res.choices.message.content
                clean, fig = parse_chart(ans)
                
                st.markdown(clean)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="new_c")
                    try:
                        st.download_button("📥 Grafikni rasm (PNG) yuklab olish", fig.to_image(format="png"), "grafik.png", "image/png")
                    except Exception:
                        pass
                st.session_state.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": clean, "fig": fig}]
            except Exception as e:
                st.error(f"Xatolik: {e}")
    
