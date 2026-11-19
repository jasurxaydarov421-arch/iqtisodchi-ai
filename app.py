import os, json, re, requests
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from duckduckgo_search import DDGS

st.set_page_config(page_title="AI Iqtisodchi Pro", page_icon="📊", layout="wide")

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
            res = list(d.text(f"{q} iqtisodiyot", max_results=3))
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

with st.sidebar:
    st.title("📊 AI Iqtisodchi Pro")
    mode = st.selectbox("Yo'nalish:", ["Avtomatik (Router)", "MAKRO", "BIZNES", "BIRJA", "SOLIQ"])
    use_web = st.checkbox("🌐 Jonli yangiliklar", value=True)
    st.divider()
    st.metric("USD (CBU)", f"{cbu.get('USD')} so'm")
    if crypto:
        st.metric("Bitcoin", f"${crypto['bitcoin']['usd']:,}")
    file = st.file_uploader("Fayl (Excel, CSV, PDF)", type=["xlsx", "csv", "pdf"])
    if st.button("🗑️ Xotirani tozalash"):
        st.session_state.messages = []
        st.rerun()

st.header("💼 Strategik AI Iqtisodchi Maslahatchi")

doc_info = ""
if file:
    if file.name.endswith(".pdf"):
        doc_info = "".join([p.extract_text() for p in PdfReader(file).pages[:12]])[:5000]
        st.success(f"📄 PDF yuklandi: {file.name}")
    else:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.dataframe(df.head(5))
        nums = df.select_dtypes(include="number").columns.tolist()
        if nums:
            st.bar_chart(df[nums].head(15))
        doc_info = df.describe().to_string()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "**Assalomu alaykum!** Yangi iqtisodiy munosabatlar, bozor raqamlari yoki soliq masalalarini jadvallar va grafiklar bilan tahlil qilishga tayyorman."}]

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("fig"):
            st.plotly_chart(m["fig"], use_container_width=True, key=f"c_{i}")

if prompt := st.chat_input("Savolingizni yozing..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Tahlil qilinmoqda..."):
            try:
                r_res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": f"Faqat bitta so'z yoz (MAKRO, BIZNES, BIRJA, SOLIQ): {prompt}"}], temperature=0)
                cat = r_res.choices[0].message.content.strip().upper() if mode == "Avtomatik (Router)" else mode
                web = web_search(prompt) if use_web else ""
                sys = f"Sen strategik iqtisodiy maslahatchisan ({cat}). O'zbek tilida, birinchi gapda **qalin** xulosa, keyin punktlar va jadval bilan javob ber, oxirida keyingi qadam uchun savol ber. Grafik kerak bo'lsa oxirida qo'sh: ```chart\n{{\"type\":\"bar\",\"title\":\"Nomi\",\"x_label\":\"X\",\"y_label\":\"Y\",\"x\":[\"A\",\"B\"],\"y\":[10,20]}}\n```\nKurslar: {cbu}, Kripto: {crypto}, Hujjat: {doc_info}, Yangiliklar: {web}"
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
            except Exception as e:
                st.error(f"Xatolik: {e}")
