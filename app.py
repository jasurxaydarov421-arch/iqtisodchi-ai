import os
import requests
import pandas as pd
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from duckduckgo_search import DDGS

st.set_page_config(page_title="AI Iqtisodchi Maslahatchi", page_icon="📊", layout="wide")

# --- 1. PAROL BILAN HIMOYA ---
app_password = st.secrets.get("APP_PASSWORD") or os.environ.get("APP_PASSWORD")
if app_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🔒 Tizimga kirish")
        pwd = st.text_input("Shaxsiy parolni kiriting:", type="password")
        if st.button("Kirish"):
            if pwd == app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Parol noto'g'ri!")
        st.stop()

api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

@st.cache_data(ttl=3600)
def get_working_model():
    candidates = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b"
    ]
    for m in candidates:
        try:
            client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1
            )
            return m
        except Exception:
            continue
    models = [m.id for m in client.models.list().data if "whisper" not in m.id and "guard" not in m.id and "tts" not in m.id]
    return models[0] if models else "llama-3.1-8b-instant"

ACTIVE_MODEL = get_working_model()

# --- 2. TASHQI MA'LUMOTLAR VA JONLI QIDIRUV ---
@st.cache_data(ttl=600)
def get_cbu_rates():
    try:
        res = requests.get("https://cbu.uz", timeout=5).json()
        return {item['Ccy']: item['Rate'] for item in res if item['Ccy'] in ['USD', 'EUR', 'RUB']}
    except Exception:
        return {"USD": "Ma'lumot yo'q", "EUR": "-", "RUB": "-"}

@st.cache_data(ttl=300)
def get_crypto_prices():
    try:
        url = "https://coingecko.com"
        return requests.get(url, timeout=5).json()
    except Exception:
        return {}

def search_economic_news(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{query} iqtisodiyot yangiliklar", max_results=4))
            if not results:
                return ""
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception:
        return ""

# --- 3. YON PANEL (SIDEBAR) ---
with st.sidebar:
    st.title("📊 AI Iqtisodchi Pro")
    mode = st.selectbox(
        "Yo'nalishni tanlang:",
        [
            "Avtomatik (Router)",
            "1. Makroiqtisodiyot va Yangi Munosabatlar",
            "2. Biznes tahlil va Strategiya",
            "3. Kripto va Birja",
            "4. Soliq maslahatchisi"
        ]
    )
    use_web = st.checkbox("🌐 Internetdan yangiliklarni qidirish", value=True)
    
    st.divider()
    st.subheader("💱 Jonli kurslar")
    rates = get_cbu_rates()
    st.metric("USD (Markaziy bank)", f"{rates.get('USD')} so'm")
    crypto = get_crypto_prices()
    if crypto:
        st.metric("Bitcoin (BTC)", f"${crypto['bitcoin']['usd']:,}")
        st.metric("Toncoin (TON)", f"${crypto['toncoin']['usd']}")
        
    st.divider()
    uploaded_file = st.file_uploader("Hujjat yuklang (Excel, CSV yoki PDF)", type=["xlsx", "csv", "pdf"])
    
    if st.button("🗑️ Suhbat xotirasini tozalash"):
        st.session_state.messages = []
        st.rerun()

# --- 4. ROUTER VA MASLAHATCHI AGENTLAR ---
def route_query(text):
    prompt = f"""Foydalanuvchi savolini tahlil qil va faqat bitta so'z bilan javob ber:
    MAKRO - agar iqtisodiy munosabatlar, yangiliklar, valyuta, inflyatsiya, savdo haqida bo'lsa.
    BIZNES - agar foyda, zarar, biznes strategiya, jadval tahlili haqida bo'lsa.
    BIRJA - agar kripto, aksiyalar, treyding haqida bo'lsa.
    SOLIQ - agar soliqlar, soliq kodeksi, PDF hujjat haqida bo'lsa.
    Savol: {text}"""
    res = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res.choices[0].message.content.strip().upper()

def ask_advisor(category, user_text, chat_history, doc_context="", web_context=""):
    base_instruction = (
        "Sen foydalanuvchining shaxsiy strategik iqtisodiy maslahatchisi va hamkorisan. "
        "Faqat o'zbek tilida, chuqur tahliliy, mantiqiy va amaliy tavsiyalar berib suhbat olib bor. "
        "Javob oxirida suhbatni davom ettirish uchun foydalanuvchiga 1-2 ta o'ylantiruvchi savol yoki keyingi qadam bo'yicha taklif ber."
    )
    
    if "MAKRO" in category:
        role = f"{base_instruction} Yo'nalishing: Makroiqtisodiyot va xalqaro/mahalliy yangi iqtisodiy munosabatlar tahlili."
        live_info = f"Markaziy bank kurslari: {rates}"
    elif "BIRJA" in category:
        role = f"{base_instruction} Yo'nalishing: Kripto va fond birjalari tahlili. Oxirida 'Bu moliyaviy maslahat emas' deb eslat."
        live_info = f"Kripto narxlar: {crypto}"
    elif "BIZNES" in category:
        role = f"{base_instruction} Yo'nalishing: Biznes-analitika va moliyaviy direktor (CFO). Aniq hisob-kitob va tavsiyalar ber."
        live_info = ""
    else:
        role = f"{base_instruction} Yo'nalishing: O'zbekiston soliq qonunchiligi va soliq optimizatsiyasi."
        live_info = ""

    system_msg = f"{role}\n\nJoriy ma'lumotlar: {live_info}\nYuklangan hujjat ma'lumotlari: {doc_context}\nInternetdan topilgan so'nggi yangiliklar: {web_context}"
    
    # Oxirgi 10 ta xabarni xotira sifatida uzatish
    messages_payload = [{"role": "system", "content": system_msg}]
    for m in chat_history[-10:]:
        messages_payload.append({"role": m["role"], "content": m["content"]})
    messages_payload.append({"role": "user", "content": user_text})

    res = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=messages_payload,
        temperature=0.6
    )
    return res.choices[0].message.content

# --- 5. ASOSIY OYNANI CHIZISH ---
st.header("💼 Strategik AI Iqtisodchi Maslahatchi")

doc_summary = ""
if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in reader.pages[:15]:  # Dastlabki 15 sahifani o'qish
            pdf_text += page.extract_text() + "\n"
        doc_summary = f"PDF hujjat ({uploaded_file.name}) matni:\n{pdf_text[:6000]}"
        st.success(f"📄 PDF hujjat yuklandi: {uploaded_file.name} ({len(reader.pages)} sahifa)")
    else:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        with st.expander("📊 Yuklangan jadval va avtomatik grafikni ko'rish", expanded=True):
            st.dataframe(df.head(10))
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                st.bar_chart(df[num_cols].head(20))
        doc_summary = f"Jadval ustunlari: {list(df.columns)}. Statistika:\n{df.describe().to_string()}\nBirinchi qatorlar:\n{df.head(10).to_string()}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Assalomu alaykum! Men sizning shaxsiy iqtisodiy maslahatchingizman. Bugun yangi iqtisodiy munosabatlar, bozor tendensiyalari, biznesingiz raqamlari yoki soliq masalalaridan qaysi birini tahlil qilamiz?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Fikringizni yoki savolingizni yozing..."):
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Yangiliklar o'rganilmoqda va tahlil tayyorlanmoqda..."):
            try:
                selected_cat = route_query(prompt) if mode == "Avtomatik (Router)" else mode.upper()
                web_data = search_economic_news(prompt) if use_web else ""
                answer = ask_advisor(selected_cat, prompt, st.session_state.messages, doc_summary, web_data)
                final_text = f"**📌 Bo'lim:** `{selected_cat}`\n\n{answer}"
                st.markdown(final_text)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": final_text})
            except Exception as e:
                st.error(f"Texnik xatolik: {e}")
                
