import os, json, re, requests
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from duckduckgo_search import DDGS

st.set_page_config(page_title="AI Iqtisodchi Agent", page_icon="✨", layout="centered")

# --- MAXSUS GEMINI USLUBIDAGI DIZAYN (CSS) ---
st.markdown("""
<style>
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #9aa0a6;
        margin-bottom: 1.5rem;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 16px;
        padding: 14px 16px;
        text-align: left;
        border: 1px solid rgba(155, 114, 203, 0.3);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #4285F4;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

pwd_secret = st.secrets.get("APP_PASSWORD")
if pwd_secret and not st.session_state.get("auth"):
    st.title("🔒 Tizimga kirish")
    if st.button("Kirish") and st.text_input("Parol:", type="password") == pwd_secret:
        st.session_state.auth = True
        st.rerun()
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
        cbu = {i['Ccy']: i['Rate'] for i in r if i['Ccy'] in ['USD', 'EUR', 'RUB']}
    except Exception:
        cbu = {"USD": "-"}
    try:
        cr = requests.get("https://coingecko.com", timeout=5).json()
    except Exception:
        cr = {}
    return cbu, cr

cbu, crypto = get_rates()

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
            fig = px.line(df, x=df.columns[0], y=df.columns[1], title=t, markers=True) if d.get("type") == "line" else px.bar(df, x=df.columns[0], y=df.columns[1], title=t, text_auto=True)
            fig.update_layout(template="plotly_dark")
        except Exception:
            pass
    return clean, fig

# --- YON PANEL ---
with st.sidebar:
    st.title("✨ AI Iqtisodchi")
    mode = st.selectbox("Yo'nalish:", ["Avtomatik (Router)", "IQTISODIY TA'LIMOTLAR TARIXI", "BIZNES QURISH VA TAHLIL", "MAKROIQTISODIYOT", "KRIPTO VA BIRJA", "SOLIQ MASLAHATCHISI"])
    use_web = st.checkbox("🌐 Internetdan jonli qidiruv", value=True)
    st.divider()
    st.metric("USD (CBU)", f"{cbu.get('USD')} so'm")
    if crypto:
        st.metric("Bitcoin", f"${crypto['bitcoin']['usd']:,}")
    file = st.file_uploader("Hujjat yuklang (PDF, Excel, CSV)", type=["xlsx", "csv", "pdf"])
    if st.button("➕ Yangi suhbat boshlash"):
        st.session_state.messages = []
        st.rerun()

doc_info = ""
if file:
    if file.name.endswith(".pdf"):
        doc_info = "".join([p.extract_text() for p in PdfReader(file).pages[:15]])[:6000]
        st.success(f"📄 PDF yuklandi: {file.name}")
    else:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        with st.expander("📊 Yuklangan jadval", expanded=True):
            st.dataframe(df.head(5))
        doc_info = df.describe().to_string()

if "messages" not in st.session_state:
    st.session_state.messages = []

selected_prompt = None

# --- BOSH EKRAN (GEMINI USLUBIDAGI TEZKOR KARTOCHKALAR) ---
if len(st.session_state.messages) == 0:
    st.markdown('<div class="hero-title">Salom, Jasur 👋</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Bugun qaysi iqtisodiy mavzuni tahlil qilamiz?</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📜 Iqtisodiy ta'limotlar tarixi\n\nAdam Smit, Keyns va klassik maktablar tahlili"):
            selected_prompt = "Iqtisodiy ta'limotlar tarixidagi eng muhim maktablar (merkantilizm, klassik maktab, keynschilik, monetarizm) va ularning asoschilarini jadval ko'rinishida tushuntirib ber."
        if st.button("📊 Makroiqtisodiy tahlil va Grafik\n\nO'zbekiston YaIM, inflyatsiya va valyuta kursi"):
            selected_prompt = "O'zbekistonning oxirgi 4 yildagi asosiy makroiqtisodiy ko'rsatkichlari va inflyatsiya dinamikasini tahlil qilib, grafik chizib ber."
    with col2:
        if st.button("🚀 Biznes qurish va tahlil\n\nNoldan biznes-reja, unit-ekonomika va foyda"):
            selected_prompt = "O'zbekiston sharoitida noldan daromadli biznes qurish bosqichlari, xarajatlar va foyda tahlili (unit-ekonomika) bo'yicha amaliy qo'llanma ber."
        if st.button("⚖️ Soliqlar va Birja tahlili\n\nQQS, foyda solig'i va kripto bozori holati"):
            selected_prompt = "O'zbekistonda tadbirkorlar uchun asosiy soliq rejimlari (aylanma soliq va QQS) farqini jadvalda taqqoslab ber."

# --- CHAT TARIXI ---
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("fig"):
            st.plotly_chart(m["fig"], use_container_width=True, key=f"c_{i}")

user_input = st.chat_input("Iqtisodiyot, biznes qurish yoki ta'limotlar haqida savol bering...")
prompt = selected_prompt if selected_prompt else user_input

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Tahlil qilinmoqda..."):
            try:
                if mode == "Avtomatik (Router)":
                    r_prompt = f"Savol qaysi bo'limga tegishli? Faqat bitta ibora yoz (IQTISODIY TA'LIMOTLAR TARIXI, BIZNES QURISH VA TAHLIL, MAKROIQTISODIYOT, KRIPTO VA BIRJA, SOLIQ MASLAHATCHISI): {prompt}"
                    r_res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": r_prompt}], temperature=0)
                    cat = r_res.choices[0].message.content.strip().upper()
                else:
                    cat = mode

                web = web_search(prompt) if use_web else ""
                
                sys = f"""Sen Google Gemini darajasidagi mukammal, bilimdon AI iqtisodchi-ekspertsan (Yo'nalish: {cat}).
                Qoidalar:
                1. Faqat ravon va adabiy o'zbek tilida javob ber.
                2. Birinchi gapdayoq savolning eng asosiy javobini yoki xulosasini **qalin shrift (bold)** bilan ber.
                3. Ma'lumotlarni kichik sarlavhalar (###), punktlar (*) va taqqoslash jadvallari (Markdown Table) ko'rinishida tuzilmali ber.
                4. Javob oxirida mavzuni chuqurroq o'rganish uchun 1-2 ta keyingi savol yoki qadam taklif qil.
                5. Agar raqamlarni taqqoslash yoki grafik chizish so'ralsa, oxirida aniq shu formatda JSON qo'sh:
                ```chart
                {{"type":"bar","title":"Sarlavha","x_label":"X","y_label":"Y","x":["A","B"],"y":[10,20]}}
                ```
                Qo'shimcha ma'lumotlar: Kurslar: {cbu}, Kripto: {crypto}, Yuklangan fayl: {doc_info}, Internet ma'lumotlari: {web}"""

                msgs = [{"role": "system", "content": sys}] + [{"role": x["role"], "content": x["content"]} for x in st.session_state.messages[-8:]] + [{"role": "user", "content": prompt}]
                a_res = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.5)
                ans = a_res.choices[0].message.content
                clean, fig = parse_chart(ans)
                out = f"**📌 Bo'lim:** `{cat}`\n\n{clean}"
                st.markdown(out)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="new_c")
                    try:
                        st.download_button("📥 Grafikni rasm (PNG) yuklab olish", fig.to_image(format="png"), "grafik.png", "image/png")
                    except Exception:
                        pass
                st.session_state.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": out, "fig": fig}]
                if selected_prompt:
                    st.rerun()
            except Exception as e:
                st.error(f"Xatolik: {e}")
        
