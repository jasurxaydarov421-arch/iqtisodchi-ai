import json, math, re, requests
from duckduckgo_search import DDGS
from groq import Groq
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
import streamlit as st

st.set_page_config(page_title="Iqtisodchi AI", page_icon="🏛️", layout="centered")

CSS = "<style>.stApp{background-color:#F9F8F6;color:#1F1F1F;font-family:sans-serif;}header[data-testid='stHeader']{background:transparent;}section[data-testid='stSidebar']{background-color:#F2F0EC!important;}.claude-hero{text-align:center;margin-top:1rem;margin-bottom:1rem;}.tmii-logo-box{display:inline-flex;align-items:center;justify-content:center;gap:12px;margin-bottom:10px;text-align:left;}.tmii-text{font-weight:700;font-size:15px;line-height:1.15;}.tmii-red{color:#FF3B22;}.tmii-blue{color:#16354D;}.claude-title{font-size:1.5rem;font-weight:600;color:#1A1918;}.claude-sub{font-size:0.9rem;color:#6B6863;}div[data-testid='stChatMessage']{background-color:transparent!important;padding:10px 0!important;}div[data-testid='stChatMessage']:has(div[data-testid='stChatMessageAvatarUser']){flex-direction:row-reverse;background-color:#EFECE6!important;border-radius:18px!important;padding:10px 16px!important;margin-left:auto!important;max-width:82%;}div[data-testid='stChatMessage']:has(div[data-testid='stChatMessageAvatarUser']) div[data-testid='stChatMessageAvatarUser']{display:none;}div[data-testid='stBottom']{background-color:#F9F8F6!important;}div[data-testid='stChatInput']{border-radius:24px!important;border:1px solid #DCD8D0!important;background-color:#FFFFFF!important;}div.stButton>button{border-radius:14px!important;border:1px solid #E2DDD5!important;background-color:#FFFFFF!important;}</style>"
st.markdown(CSS, unsafe_allow_html=True)

LOGO_HTML = '<div style="width:64px;height:64px;position:relative;flex-shrink:0;"><div style="position:absolute;top:0;left:0;width:64px;height:11px;background:#16354D;"></div><div style="position:absolute;top:0;left:0;width:11px;height:64px;background:#16354D;"></div><div style="position:absolute;top:15px;left:15px;width:10px;height:25px;background:#FF3B22;"></div><div style="position:absolute;top:15px;left:28px;width:10px;height:31px;background:#FF3B22;"></div><div style="position:absolute;top:15px;left:41px;width:10px;height:18px;background:#FF3B22;"></div><div style="position:absolute;top:15px;left:54px;width:10px;height:10px;background:#FF3B22;"></div><div style="position:absolute;top:49px;left:28px;width:10px;height:15px;background:#16354D;"></div><div style="position:absolute;top:36px;left:41px;width:10px;height:28px;background:#16354D;"></div><div style="position:absolute;top:28px;left:54px;width:10px;height:36px;background:#16354D;"></div></div>'

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

@st.cache_data(ttl=3600)
def get_model():
    for m in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
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

def web_search(q, is_acc=False):
    try:
        tag = "buxgalteriya BHMS soliq" if is_acc else "biznes moliya iqtisodiyot"
        with DDGS() as d:
            return "\n".join(["- " + x.get("title", "") + ": " + x.get("body", "") for x in d.text(q + " " + tag, max_results=3)])
    except Exception:
        return ""

def run_calc(text):
    res_list = []
    matches = re.findall(r"```calc\s*(.*?)\s*```", text, re.DOTALL)
    clean = re.sub(r"```calc\s*.*?\s*```", "", text, flags=re.DOTALL).strip()
    for code in matches:
        try:
            env = {"math": math, "round": round, "sum": sum, "pow": pow}
            loc = {}
            exec(code, {"__builtins__": {}}, {**env, **loc})
            res_list.append("Natija: " + str(loc.get("natija", "Hisoblandi")))
        except Exception as e:
            res_list.append("Xato: " + str(e))
    return clean, "\n".join(res_list)

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
                fig = px.line(df, x=xl, y=yl, title=t, markers=True, color_discrete_sequence=["#FF3B22"])
            else:
                fig = px.bar(df, x=xl, y=yl, title=t, text_auto=True, color_discrete_sequence=["#16354D"])
            fig.update_layout(template="plotly_white")
        except Exception:
            pass
    return clean, fig

def ask_ai(prompt, is_acc, doc_info, history):
    try:
        s_res = web_search(prompt, is_acc)
        if is_acc:
            sys_p = "Sen TMII Bosh Buxgalter AI ekspertsan. O'zbekiston 21-sonli BHMS va Soliq kodeksiga asoslan. Faqat ravon o'zbek tilida yoz. Birinchi jumlada natijani **bold** qil. Masala berilsa avval tushuntirib ishla, provodkalarni | Operatsiya | Debet | Kredit | Summa | jadvalida ber, oxirida formula va soliq stavkalarini ko'rsat. Murakkab hisob uchun ```calc blokida Python yoz (natija = ...). CBU: " + str(cbu) + ". Fayl: " + doc_info + ". Internet: " + s_res
        else:
            sys_p = "Sen TMII AI iqtisodchi, biznes tahlilchi va moliyachi-ekspertsan. Faqat ravon o'zbek tilida yoz. Masala berilsa DARHOL avval tushuntirib ishla, birinchi jumlada yakuniy natijani **bold** qil va oxirida foydalanilgan formulani ko'rsat. Murakkab hisob uchun ```calc blokida Python yoz (natija = ...). Grafik kerak bo'lsa ```chart blokida JSON yoz (type, title, x_label, y_label, x, y). CBU: " + str(cbu) + ". Fayl: " + doc_info + ". Internet: " + s_res

        msgs = [{"role": "system", "content": sys_p}] + [{"role": x["role"], "content": x["content"]} for x in history] + [{"role": "user", "content": prompt}]
        res = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.2)
        ans = next(iter(res.choices)).message.content

        ans_clean, c_out = run_calc(ans)
        if c_out:
            msgs.append({"role": "assistant", "content": ans})
            msgs.append({"role": "user", "content": "Aniq hisob: " + c_out + ". Shu natija bilan tushuntirilgan yechim va formulani yoz (```calc yozma)."})
            res2 = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.2)
            ans_clean = next(iter(res2.choices)).message.content

        return parse_chart(ans_clean)
    except Exception as e:
        return "Xatolik: " + str(e), None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "general"

with st.sidebar:
    st.markdown("### 🏛️ Iqtisodchi AI")
    if st.button("➕ Yangi suhbat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.metric("🇺🇸 USD", str(cbu.get("USD", "-")) + " so'm")
    st.metric("🇪🇺 EUR", str(cbu.get("EUR", "-")) + " so'm")
    st.metric("🇷🇺 RUB", str(cbu.get("RUB", "-")) + " so'm")
    st.divider()
    file = st.file_uploader("📎 PDF, Excel, CSV yuklash", type=["pdf", "xlsx", "csv"])

doc_info = ""
if file:
    if file.name.endswith(".pdf"):
        doc_info = "".join([p.extract_text() or "" for p in PdfReader(file).pages])
    else:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        doc_info = df.to_string()

m1, m2 = st.columns(2)
with m1:
    if st.button("🌐 Umumiy (Iqtisodiyot, Moliya, Biznes)", use_container_width=True):
        st.session_state.mode = "general"
        st.rerun()
with m2:
    if st.button("📒 Buxgalteriya hisobi (BHMS)", use_container_width=True):
        st.session_state.mode = "accounting"
        st.rerun()

is_acc = st.session_state.mode == "accounting"
selected_prompt = None

if not st.session_state.messages:
    h_title = "Buxgalteriya hisobi va soliq tahlili" if is_acc else "Bugun qanday iqtisodiy tahlil qilamiz?"
    h_sub = "O'zbekiston BHMS (21-son), provodkalar va balans" if is_acc else "Shaxsiy AI iqtisodchi, biznes tahlilchi va moliyachi"
    hero = '<div class="claude-hero"><div class="tmii-logo-box">' + LOGO_HTML + '<div class="tmii-text"><div class="tmii-red">TOSHKENT</div><div class="tmii-blue">MENEJMENT VA</div><div class="tmii-blue">IQTISODIYOT</div><div class="tmii-blue">INSTITUTI</div></div></div><div class="claude-title">' + h_title + '</div><div class="claude-sub">' + h_sub + '</div></div>'
    st.markdown(hero, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📒 Xomashyo provodkasi" if is_acc else "📊 Biznes va Unit-ekonomika", use_container_width=True):
            selected_prompt = "Korxona 11 200 000 so'mga (QQS 12% bilan) xomashyo oldi. BHMS provodkalarini jadvalda yoz." if is_acc else "Kichik qahvaxona misolida unit-ekonomika va zararsizlik nuqtasini hisobla."
        if st.button("🏭 Amortizatsiya hisobi" if is_acc else "📈 Valyuta kurslari tahlili", use_container_width=True):
            selected_prompt = "60 mln so'mlik uskuna 5 yilga to'g'ri chiziqli usulda amortizatsiya qilinadi. Oylik summani hisobla." if is_acc else "Bugungi USD, EUR va RUB kurslari holatini qisqacha tahlil qilib ber."
    with c2:
        if st.button("🧾 Ish haqi va soliqlar" if is_acc else "🧮 Kredit va NPV masalasi", use_container_width=True):
    
