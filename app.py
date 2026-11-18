import os
import requests
import pandas as pd
import streamlit as st
from google import genai

st.set_page_config(page_title="AI Iqtisodchi Agent", page_icon="📊", layout="wide")

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

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

with st.sidebar:
    st.title("📊 Iqtisodiy AI Tizim")
    mode = st.selectbox(
        "Mutaxassisni tanlang:",
        ["Avtomatik (Router)", "1. Makroiqtisodiyot", "2. Biznes tahlil", "3. Kripto va Birja", "4. Soliq maslahatchisi"]
    )
    st.divider()
    st.subheader("💱 Jonli kurslar")
    rates = get_cbu_rates()
    st.metric("USD (Markaziy bank)", f"{rates.get('USD')} so'm")
    crypto = get_crypto_prices()
    if crypto:
        st.metric("Bitcoin (BTC)", f"${crypto['bitcoin']['usd']:,}")
        st.metric("Toncoin (TON)", f"${crypto['toncoin']['usd']}")
    st.divider()
    uploaded_file = st.file_uploader("Excel yoki CSV fayl yuklang", type=["xlsx", "csv"])

def route_query(text):
    prompt = f"Foydalanuvchi savolini tahlil qil va faqat bitta so'z bilan javob ber (MAKRO, BIZNES, BIRJA yoki SOLIQ):\nSavol: {text}"
    res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return res.text.strip().upper()

def ask_agent(category, user_text, file_context=""):
    context = ""
    if "MAKRO" in category:
        role = "Sen O'zbekiston va jahon makroiqtisodiyoti bo'yicha tahlilchisan. Aniq raqamlar bilan javob ber."
        context = f"Joriy CBU valyuta kurslari: {rates}"
    elif "BIRJA" in category:
        role = "Sen kripto va fond birjasi tahlilchisisan. Oxirida 'Bu moliyaviy maslahat emas' deb eslat."
        context = f"Joriy kripto narxlar: {crypto}"
    elif "BIZNES" in category:
        role = "Sen biznes-analitik va moliyaviy direktorsan (CFO). Jadval va aniq hisob-kitoblar bilan tushuntir."
        context = file_context
    else:
        role = "Sen O'zbekiston soliq qonunchiligi bo'yicha mutaxassissan. O'zR Soliq kodeksiga asoslanib javob ber."

    full_prompt = f"Roling: {role}\nQo'shimcha ma'lumot: {context}\n\nSavol: {user_text}"
    res = client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
    return res.text

st.header("💼 4+1 Multi-Agent Iqtisodiy Yordamchi")

file_summary = ""
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.dataframe(df.head(5))
    file_summary = f"Yuklangan jadval ustunlari: {list(df.columns)}. Statistika:\n{df.describe().to_string()}"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Iqtisodiy savolingizni yozing..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Mutaxassis tahlil qilmoqda..."):
            try:
                selected_cat = route_query(prompt) if mode == "Avtomatik (Router)" else mode.upper()
                answer = ask_agent(selected_cat, prompt, file_summary)
                final_text = f"**📌 Bo'lim:** `{selected_cat}`\n\n{answer}"
                st.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
            except Exception as e:
                st.error(f"Texnik xatolik tafsiloti: {e}")
