# T6 — Collaborative filtering tavsiya tizimi

Video-striming platformasi uchun **"Siz uchun"** tavsiya tizimi: har bir
foydalanuvchiga keyingi 14 kunda ko'radigan **10 ta yangi kontent**.
Faqat xatti-harakat ma'lumotlari ishlatiladi (janr/tavsif/aktyor yo'q) — bu
sof collaborative filtering masalasi.

- Baza: `ml_t6_recsys_cf` (login `t6_student`, ulanish `.env` da)
- Asosiy metrika: **NDCG@10** — o'tish `≥ 0.055`, a'lo `≥ 0.070`
- Cheklov: 15 daqiqa, 15 000 foydalanuvchi × 10 = 150 000 qator

## Tez boshlash

```bash
# 1) kutubxonalar
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) ulanish ma'lumotlari (mentor bergan qiymatlar)
cp .env.example .env      # va PGPASSWORD ni yozing

# 3) ma'lumotni tekshirish (Hafta-1 auditi, universitet tarmog'i kerak)
python -m src.audit

# 5) modellarni offline solishtirish (leaderboard'siz)
python -m src.evaluate --models popularity itemknn als --sample 3000

# 5b) ALS giperparametrlarini tanlash -> reports/tuning.csv
python -m src.tune --sample 3000

# 5c) segment tahlili -> reports/segments.csv
python -m src.segments --sample 3000

# 6) tavsiyalarni hisoblab, bazaga yozish va baholash
python -m src.recommend --run-name pop_v1 --model popularity --score
python -m src.recommend --run-name als_v1 --model als --score

# 7) testlar (bazasiz ishlaydi)
python -m unittest discover -s tests -t .
```

## Baza haqida

Baza `10.20.9.24` — ichki tarmoq, ya'ni faqat universitetda/VPN ichida
ko'rinadi. Hamma buyruqlar to'g'ridan-to'g'ri shu bazada ishlaydi (lokal
nusxa ishlatilmaydi). `eval.score_run()` ham faqat shu bazada ishlaydi.

Sinov uchun foydali bayroqlar: `--limit-users 200 --dry-run`
(bazaga hech narsa yozmaydi, faqat hisoblaydi).

## Kod tuzilishi

| Fayl | Vazifasi |
|---|---|
| `src/config.py` | `config.yaml` va `.env` ni o'qish (parol kodda yo'q) |
| `src/store.py` | yagona DB qatlami: o'qish (server-side cursor) va yozish |
| `src/data.py` | 3 signalni bitta user × item matritsasiga jamlash |
| `src/models.py` | model registri (yangi model qo'shish uchun bitta joy) |
| `src/als.py` | ALS — implicit matrix factorization (numpy, scipy'siz) |
| `src/knn.py` | item-kNN — cosine o'xshashlik (scipy) |
| `src/pipeline.py` | asosiy oqim + `select_top_k` (ko'rilganlarni chiqarish) |
| `src/recommend.py` | CLI: hisoblash → `ml.recommendations` |
| `src/evaluate.py` | offline validatsiya: oxirgi 14 kun, model × metrika jadvali |
| `src/audit.py` | ma'lumot auditi so'rovlari |
| `tests/` | metrikalar, `select_top_k`, ALS, item-kNN, butun oqim |

## Model strategiyasi (nega shu tartibda)

1. **popularity (baseline)** — oxirgi 14 kunda eng ko'p turli odam ko'rgan
   kontent. NDCG ≈ 0.04. Maqsad: pipeline to'g'ri ishlashini isbotlash.
2. **item-kNN** — "buni ko'rganlar buni ham ko'rgan". Oddiy, tushunarli,
   odatda baseline'dan sezilarli yaxshi.
3. **ALS** — matritsani `U @ V.T` ko'paytmasiga ajratadi; user va item
   faktorlarini navbat bilan yechadi. Implicit feedback uchun
   `confidence = 1 + alpha * w` ishlatiladi.
4. **tuning** — `factors`, `regularization`, `alpha`, `neighbors`.
5. Bonus (vaqt qolsa): ikki bosqichli pipeline (kandidatlar → LightGBM ranker).

## Implicit signal og'irliklari

`config.yaml → weights.scheme` bilan ikkita sxema almashtiriladi:

- **A** — uzluksiz: `watch_pct` qiymati (0.3 dan past bo'lsa 0), takror ko'rish
  bonusi `1 + 0.5·log1p(n−1)`, baho `0.2·rating`, saqlash `+1`.
- **B** — binar: `watch_pct ≥ 0.3` bo'lsa 1 ball, qolgani A bilan bir xil.

Ikkalasiga ham vaqt bo'yicha eskirish qo'llanadi: `0.5 ** (yosh / 180 kun)`.

## Natijani tekshirish

Yozilgandan keyin faqat `eval.score_run(run_id)` haqiqiy haqiqatni ko'rsatadi:

```sql
SELECT * FROM eval.score_run(<run_id>);
SELECT * FROM eval.leaderboard;
```

Kutilgan tekshiruvlar: `already_seen_share = 0` (train'da ko'rilganlar
tavsiya qilinmagan), `users_covered = 1.0`, har userda aynan 10 ta takrorlanmagan
kontent.

## Qoidalar (buzilsa — diskvalifikatsiya)

- Parol kodda yozilmaydi; `.env` `.gitignore` da (allaqachon shunday).
- Kontent metadatasini tashqi manbadan to'ldirish taqiqlanadi — faqat CF.
- `raw` sxemasi faqat o'qish uchun: `INSERT/UPDATE` qilishga urinmang.
- Jamoalar o'rtasida kod almashish taqiqlanadi.

## O'qituvchiga qanday tushuntirish (qisqa versiya)

1. **Muammo**: 50k user, 6k kontent, faqat xatti-harakat. Har biriga 10 ta
   *yangi* kontent kerak → ranking masalasi, implicit feedback.
2. **Nega popularity yetmaydi**: hammaga bir xil ro'yxat → shaxsiylashtirish
   yo'q, coverage past, NDCG 0.04.
3. **Ma'lumot**: ko'rish (watch_pct), baho (1..10), saqlash → vaznli matritsa;
   vaqt decay eski qiziqishlarni susaytiradi.
4. **Model**: ALS matritsani faktorlarga ajratadi; item-kNN qo'shnichilik
   orqali ishlaydi; ikkalasi ham "kim nima ko'rgan" dan o'rganadi.
5. **Validatsiya**: train ichidan oxirgi 14 kunni ajratib, test qoidasini
   aynan takrorlaymiz (sana bo'yicha kesish — ma'lumot oqmasligi uchun).
6. **Filtr**: train'da ko'rilganlar chiqariladi, `watch_pct < 0.3` hisobga
   olinmaydi, train'da umuman yo'q kontent tavsiya qilinmaydi.
7. **Metrika**: NDCG@10 — to'g'ri kontent qanchalik yuqori o'rinda turishi.
