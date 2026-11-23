import base64
import json, math, re, requests
from duckduckgo_search import DDGS
from groq import Groq
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
import streamlit as st

ICON_SVG = """<svg xmlns="http://w3.org" viewBox="0 0 100 100">
  <rect x="6" y="6" width="88" height="17" fill="#16354D"/>
  <rect x="6" y="6" width="17" height="88" fill="#16354D"/>
  <rect x="27" y="27" width="16" height="36" fill="#FF3B22"/>
  <rect x="47" y="27" width="16" height="44" fill="#FF3B22"/>
  <rect x="67" y="27" width="16" height="26" fill="#FF3B22"/>
  <rect x="87" y="27" width="7" height="15" fill="#FF3B22"/>
  <rect x="47" y="75" width="16" height="19" fill="#16354D"/>
  <rect x="67" y="57" width="16" height="37" fill="#16354D"/>
  <rect x="87" y="46" width="7" height="48" fill="#16354D"/>
</svg>"""

ICON_DATA_URI = (
    f"data:image/svg+xml;base64,{base64.b64encode(ICON_SVG.encode()).decode()}"
)

st.set_page_config(
    page_title="Iqtisodchi, Moliya va Buxgalteriya AI",
    page_icon=ICON_DATA_URI,
    layout="centered",
)

st.markdown(
    """
<style>
    @import url('https://googleapis.com');
    
    .stApp { background-color: #F9F8F6; color: #1F1F1F; font-family: 'Inter', -apple-system, sans-serif; }
    header[data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] { background-color: #F2F0EC !important; border-right: 1px solid #E5E2DC !important; }
    
    .claude-hero { text-align: center; margin-top: 1.8rem; margin-bottom: 1.2rem; }
    .tmii-logo-box { display: inline-flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 12px; text-align: left; }
    .tmii-text { font-family: 'Oswald', 'Inter', sans-serif; font-weight: 600; font-size: 16px; line-height: 1.15; letter-spacing: 0.4px; }
    .tmii-red { color: #FF3B22; }
    .tmii-blue { color: #16354D; }
    .claude-title { font-size: 1.65rem; font-weight: 600; color: #1A1918; margin: 0.3rem 0; letter-spacing: -0.02em; }
    .claude-sub { font-size: 0.95rem; color: #6B6863; }

    div[data-testid="stChatMessage"] { background-color: transparent !important; padding: 14px 0 !important; gap: 14px !important; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        background-color: #EFECE6 !important;
        border-radius: 20px !important;
        padding: 12px 18px !important;
        margin-left: auto !important;
        max-width: 82%;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageAvatarUser"] { display: none; }

    div[data-testid="stChatMessageContent"] p { font-size: 1rem !important; line-height: 1.65 !important; color: #1F1F1F !important; margin-bottom: 0.75rem !important; }
    div[data-testid="stChatMessageContent"] h3 { font-size: 1.15rem !important; font-weight: 600 !important; color: #16354D !important; margin-top: 1.1rem !important; margin-bottom: 0.5rem !important; }
    div[data-testid="stChatMessageContent"] table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.92rem; background: #FFFFFF; border-radius: 10px; overflow: hidden; }
    div[data-testid="stChatMessageContent"] th { background-color: #EFECE6; color: #16354D; font-weight: 600; text-align: left; padding: 10px 12px; border-bottom: 1px solid #DCD8D0; }
    div[data-testid="stChatMessageContent"] td { padding: 9px 12px; border-bottom: 1px solid #EFECE6; }

    div[data-testid="stBottom"] { background-color: #F9F8F6 !important; border-top: none !important; }
    div[data-testid="stChatInput"] { border-radius: 24px !important; border: 1px solid #DCD8D0 !important; background-color: #FFFFFF !important; }
    div[data-testid="stPopover"] > button { border-radius: 50% !important; width: 44px !important; height: 44px !important; border: 1px solid #DCD8D0 !important; background-color: #FFFFFF !important; color: #16354D !important; }
    div.stButton > button { border-radius: 14px !important; border: 1px solid #E2DDD5 !important; background-color: #FFFFFF !important; color: #2D2B28 !important; font-size: 0.88rem !important; }
    div.stButton > button:hover { border-color: #FF3B22 !important; color: #FF3B22 !important; }
</style>
""",
    unsafe_allow_html=True,
)

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))


@st.cache_data(ttl=3600)
def get_model():
  for m in (
      "llama-3.3-70b-versatile",
      "openai/gpt-oss-120b",
      "llama-3.1-8b-instant",
  ):
    try:
      client.chat.completions.create(
          model=m, messages=[{"role": "user", "content": "1"}], max_tokens=1
      )
      return m
    except Exception:
      pass
  return "llama-3.1-8b-instant"


MODEL = get_model()


@st.cache_data(ttl=600)
def get_rates():
  try:
    r = requests.get(
        "https://cbu.uz", timeout=5
    ).json()
    return {
        i.get("Ccy"): i.get("Rate", "-")
        for i in r
        if i.get("Ccy") in ("USD", "EUR", "RUB")
    }
  except Exception:
    return {"USD": "-", "EUR": "-", "RUB": "-"}


cbu = get_rates()


def web_search(q, is_acc=False):
  try:
    tag = "buxgalteriya BHMS soliq" if is_acc else "biznes moliya iqtisodiyot"
    with DDGS() as d:
      return "\n".join([
          f"- {x.get('title', '')}: {x.get('body', '')}"
          for x in d.text(f"{q} {tag}", max_results=3)
      ])
  except Exception:
    return ""


def run_finance_calc(text):
  calc_results = []
  matches = re.findall(r"```calc\s*(.*?)\s*```", text, re.DOTALL)
  clean_text = re.sub(
      r"```calc\s*.*?\s*```", "", text, flags=re.DOTALL
  ).strip()
  for code in matches:
    try:
      safe_env = {"math": math, "round": round, "sum": sum, "pow": pow}
      local_vars = {}
      exec(code, {"__builtins__": {}}, {**safe_env, **local_vars})
      res = local_vars.get("natija", "Hisoblandi")
      calc_results.append(f"Aniq hisob natijasi: {res}")
    except Exception as e:
      calc_results.append(f"Hisob xatosi: {e}")
  return clean_text, "\n".join(calc_results)


def parse_chart(text):
  m = re.search(r"```chart\s*(.*?)\s*```", text, re.DOTALL)
  clean = re.sub(r"```chart\s*.*?\s*```", "", text, flags=re.DOTALL).strip()
  fig = None
  if m:
    try:
      d = json.loads(m.group(1))
      xl = d.get("x_label", "X")
      yl = d.get("y_label", "Y")
      t = d.get("title", "Grafik")
      df = pd.DataFrame({xl: d.get("x"), yl: d.get("y")})
      if d.get("type") == "line":
        fig = px.line(
            df,
            x=xl,
            y=yl,
            title=t,
            markers=True,
            color_discrete_sequence=["#FF3B22"],
        )
      else:
        fig = px.bar(
            df,
            x=xl,
            y=yl,
            title=t,
            text_auto=True,
            color_discrete_sequence=["#16354D"],
        )
      fig.update_layout(
          template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF"
      )
    except Exception:
      pass
  return clean, fig


if "messages" not in st.session_state:
  st.session_state.messages = []
if "mode" not in st.session_state:
  st.session_state.mode = "general"  # 'general' yoki 'accounting'

with st.sidebar:
  st.markdown(
      f'<div style="display:flex; align-items:center; gap:10px;'
      f' margin-bottom:10px;"><img src="{ICON_DATA_URI}" width="32"><h3'
      ' style="margin:0; color:#16354D;">Iqtisodchi AI</h3></div>',
      unsafe_allow_html=True,
  )
  if st.button("➕ Yangi suhbat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()
  st.divider()
  st.metric("🇺🇸 USD", f"{cbu.get('USD', '-')} so'm")
  st.metric("🇪🇺 EUR", f"{cbu.get('EUR', '-')} so'm")
  st.metric("🇷🇺 RUB", f"{cbu.get('RUB', '-')} so'm")
  st.divider()
  st.caption(
      "🔒 Shaxsiy va maxfiy chat. Bank kartasi yoki shaxsiy ma'lumotlar"
      " so'ralmaydi."
  )

# Rejimni tanlash tugmalari (Umumiy vs Alohida Buxgalteriya)
m1, m2 = st.columns(2)
with m1:
  if st.button(
      "🌐 Umumiy (Iqtisodiyot, Moliya, Biznes)",
      use_container_width=True,
      type="primary" if st.session_state.mode == "general" else "secondary",
  ):
    st.session_state.mode = "general"
    st.rerun()
with m2:
  if st.button(
      "📒 Buxgalteriya hisobi (BHMS)",
      use_container_width=True,
      type="primary" if st.session_state.mode == "accounting" else "secondary",
  ):
    st.session_state.mode = "accounting"
    st.rerun()

is_acc = st.session_state.mode == "accounting"
selected_prompt = None

if not st.session_state.messages:
  hero_title = (
      "Buxgalteriya hisobi va soliq tahlili"
      if is_acc
      else "Bugun qanday iqtisodiy yoki moliyaviy tahlil qilamiz?"
  )
  hero_sub = (
      "O'zbekiston BHMS (21-son), provodkalar, balans va soliq hisob-kitoblari"
      if is_acc
      else "Shaxsiy AI iqtisodchi, biznes tahlilchi va moliyachi yordamchingiz"
  )

  st.markdown(
      f"""
    <div class="claude-hero">
        <div class="tmii-logo-box">
            <svg width="68" height="68" viewBox="0 0 100 100" xmlns="http://w3.org">
                <rect x="4" y="4" width="92" height="17" fill="#16354D"/>
                <rect x="4" y="4" width="17" height="92" fill="#16354D"/>
                <rect x="26" y="26" width="16" height="38" fill="#FF3B22"/>
                <rect x="46" y="26" width="16" height="46" fill="#FF3B22"/>
                <rect x="66" y="26" width="16" height="27" fill="#FF3B22"/>
                <rect x="86" y="26" width="10" height="16" fill="#FF3B22"/>
                <rect x="46" y="76" width="16" height="20" fill="#16354D"/>
                <rect x="66" y="57" width="16" height="39" fill="#16354D"/>
                <rect x="86" y="46" width="10" height="50" fill="#16354D"/>
            </svg>
            <div class="tmii-text">
            
