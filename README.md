# The Economics AI (iqtisodchi-ai)

Python + Streamlit platformasi. Rejimlar: **Umumiy (Economics AI)**, **Buxgalteriya (Accounting AI)**,
**Prognoz (Economic Forecast AI)** va yangi **Matematika (Mathematics AI)**.

## Ishga tushirish

Python 3.11 yoki yangiroq.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Groq kalitini bering (faylni GitHub'ga qo'shmang, u `.gitignore` da):

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "kalitingiz"
```

yoki `GROQ_API_KEY` muhit o'zgaruvchisi. So'ng:

```bash
streamlit run app.py
```

Streamlit Community Cloud'da kalit **Settings → Secrets** bo'limiga yoziladi.

## Testlar

```bash
pip install -r requirements-dev.txt
python -m pytest -v -rs
```

`-rs` o'tkazib yuborilgan (SKIP) testlar sababini ko'rsatadi. Barcha kutubxonalar o'rnatilgan muhitda SKIP bo'lmasligi kerak.
GitHub'ga yuklanganda `.github/workflows/tests.yml` testlarni avtomatik bajaradi.

**Regressiya testi:** `test_hisob.py` dagi eski v0 natijalari bilan taqqoslash testi `golden_v0.json` mavjud bo'lsagina ishlaydi.
Arxivda bu etalon fayl bo'lmagani uchun tegishli test avtomatik `SKIP` qilinadi; qolgan buxgalteriya testlari ishlashda davom etadi.
Agar asl `golden_v0.json` faylingiz bo'lsa, uni `test_hisob.py` yoniga joylashtiring.

## Mathematics AI

| Daraja | Kalkulyatordagi amallar |
|---|---|
| School | ifoda, soddalashtirish, tenglama, tengsizlik, tizim, statistika |
| University | yuqoridagilar + hosila, integral, limit, matritsa, ODE, ehtimollik |
| Advanced & Applied | yuqoridagilar + gradiyent, optimallashtirish, 1D issiqlik tenglamasi (PDE), Fourier, Laplas |

- **Quick Answer** — qisqa javob; **Step-by-Step** — bosqichli yechim.
- **Chat:** AI (Groq) hisobni dasturiy asbobga beradi; tekshiruv bloki kod tomonidan qo'shiladi. Asbob ishlatilmasa,
  javob «tekshirilmagan» deb belgilanadi. Chatdagi bosqichma-bosqich izohni AI yozadi va u alohida tekshirilmaydi.
- **Kalkulyator:** AI'siz ishlaydi.
- **Rasm:** avval masala matni ko'chiriladi; noaniq bo'lsa yechilmaydi, aniqlashtirish so'raladi (ko'pi bilan 3 rasm).
- **Model:** avtomatik tanlanadi. Majburlash uchun `MATH_MODEL` va `MATH_VISION_MODEL` muhit o'zgaruvchilari.
  Groq modellari almashib turadi: https://console.groq.com/docs/deprecations

### Yozuv qoidalari

`^` daraja, `sqrt(x)`, `log(x)` (natural), `log(x, 2)`, `abs(x)`, `pi`, `e`, `inf`. O'zgaruvchi bitta harf.
O'nlik kasr nuqta bilan (`1.5`). Matritsa: `1 2; 3 4`.

### Fayllar

| Fayl | Vazifa |
|---|---|
| `matematika.py` | amallar, Groq bilan yechish, Streamlit paneli |
| `math_parser.py` | `eval()` siz, ruxsat ro'yxatiga asoslangan parser |
| `math_symbolic.py` | SymPy (aniq natija), alohida jarayonda, 20 s chegara (mpmath SymPy ichida ishlatiladi) |
| `math_verifier.py` | mustaqil sonli tekshiruvlar |
| `math_visualizer.py` | Matplotlib grafiklari |

### Cheklovlar

- 100% aniqlik kafolatlanmaydi. «Tekshiruvdan o'tdi» — sonli sinovda xato topilmadi degani, isbot emas.
- Nochiziqli tenglama va tizimlarda sonli qidiruv barcha ildizlarni topishiga kafolat yo'q.
- Juda tor cho'qqili integrallar sonli usulda noto'g'ri chiqishi mumkin.
- Aniqmas integral va soddalashtirish faqat SymPy bilan ishlaydi.
- Tengsizlik, parametrli yechim va Fourier'ning aniq (simvolik) natijasi to'liq avtomatik tekshirilmaydi.
- PDE: faqat 1D issiqlik tenglamasi (Dirixle shartlari).
- Grafiklar suhbat tarixiga saqlanmaydi. Chatda «Internetdan qidirish» matematika rejimida ishlatilmaydi.
