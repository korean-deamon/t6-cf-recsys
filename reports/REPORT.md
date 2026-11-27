# T6 — Collaborative filtering tavsiya tizimi: hisobot

**Baza:** `ml_t6_recsys_cf` (`10.20.9.24:5434`), foydalanuvchi `t6_student`
**Raund:** 3 · `as_of` = 2026-11-02 · test davri 2026-11-03 .. 2026-11-16
**Metrika:** NDCG@10 (katta — yaxshi), qo'shimcha: Recall@10, catalog coverage

---

## 1. Ma'lumot auditi

| Ko'rsatkich | Qiymat |
|---|---|
| Eventlar | 1 331 903 |
| Foydalanuvchilar | 7 031 |
| Kontent (train'da uchragan) | 4 882 |
| Katalog (raw.content) | 5 003 |
| Davr | 2026-03-01 .. 2026-11-03 |
| Test foydalanuvchilar (`raw.test_users`) | 4 577 |
| Matritsa to'ldirilganligi (density) | 0.0388 (3.9%) |

**Event turlari**

| Tur | Soni | Ulushi | Foydalanuvchi |
|---|---|---|---|
| view | 1 193 484 | 89.61% | 7 031 |
| like | 117 084 | 8.79% | 6 451 |
| share | 21 335 | 1.60% | 4 247 |

**Watch pct taqsimoti (view)**

| Oraliq | Ulush |
|---|---|
| 0.0–0.1 | 14.65% |
| 0.1–0.3 | 18.82% |
| 0.3–0.6 | 37.40% |
| 0.6–0.9 | 27.54% |
| 0.9–1.0 | 1.59% |

O'rtacha ko'rish ulushi ≈ 0.44, ya'ni ko'rishlarning uchdan bir qismi
`min_watch_pct = 0.3` chegarasidan past — ular "sifatli ko'rish" sifatida
hisobga olinmaydi, lekin "ko'rilgan" ro'yxatiga kiradi (takror tavsiya
qilinmasin).

**Anomaliyalar**

- `(user_id, content_id, event_ts)` bo'yicha 21 335 dublikat — bu share
  eventlari view bilan bir sekundda yozilgani bilan izohlanadi; model
  darajasida `(user, item)` juftligi yig'ilib ketadi, shuning uchun ta'siri yo'q.
- 121 kontent hech qachon ko'rilmagan (2.4% katalog) — ular modelda
  umuman hisobga olinmaydi.
- `sandbox.t6_bots` da **30 ta shubhali akkaunt** (eng aktivi 19 903 event,
  4 091 content). Ular mashhurlikni buzadi, shuning uchun
  `data.exclude_bots: true` bilan o'qitishdan chiqarildi.

---

## 2. Signallar va model

| Signal | Og'irlik | Izoh |
|---|---|---|
| view | `1.0 × watch_pct` | `watch_seconds / duration_s`, 0.3 dan past — 0 |
| like | `0.6` | niyat signali |
| share | `1.2` | eng kuchli niyat signali |
| takror ko'rish | `× (1 + 0.5·log1p(n−1))` | |
| vaqt eskirishi | `0.5^(age/180 kun)` | eski qiziqish kamroq ahamiyatga ega |

Ikkinchi sxema (TZ talabi) — `weights.scheme: "B"`: `watch_pct` o'rniga
binar chegara (`watch_pct ≥ 0.3 → 1`). Solishtirish `src/tune.py` da.

Modellar:

1. **Popularity** — oxirgi 14 kun, unikal foydalanuvchilar soni (baseline).
2. **Item-kNN** — cosine o'xshashlik + shrink, 200 qo'shni.
3. **ALS** (implicit, `confidence = 1 + α·w`) + recent-popularity prior:
   `score = minmax(ALS) + 0.9 · log1p(pop_recent)`.

Top-10 da: ko'rilganlar chiqariladi, train'da uchramagan kontent chiqariladi,
10 ta yetmasa — mashhurlik bilan to'ldiriladi (cold start).

---

## 3. Offline validatsiya

Train: 2026-03-01 .. 2026-10-19 · validatsiya: 2026-10-19 .. 2026-11-02
(3 000 foydalanuvchi, o'rtacha 4.8 relevant kontent, botlar chiqarilgan).

| Model | NDCG@10 | Recall@10 | Coverage |
|---|---|---|---|
| popularity | 0.0232 | 0.0320 | 0.005 |
| itemknn | 0.0301 | 0.0405 | 0.357 |
| **als** | **0.0652** | **0.0853** | 0.494 |

### Giperparametrlar tanlovi (`reports/tuning.csv`)

| Variant | NDCG@10 | Coverage |
|---|---|---|
| ALS 32 faktor, α=20 (prior yo'q) | 0.0652 | 0.494 |
| + pop_prior 0.6 | 0.0932 | 0.209 |
| + pop_prior 0.9 | 0.0963 | 0.175 |
| **+ pop_prior 0.9, α=30** | **0.0971** | 0.177 |
| 64 faktor, α=20 | 0.0632 | 0.626 |
| scheme B | 0.0509 | 0.728 |
| half-life 90 kun | 0.0596 | 0.634 |

Xulosa: sifat uchun **o'lcham emas, mashhurlik priorisi** muhim. Kattaroq
faktor soni siyrak matritsada o'qib ketadi (overfit); prior esa "keyingi
14 kunda nima ko'p ko'riladi" signalini qo'shadi. Coverage 0.494 → 0.177
tushadi, lekin TZ talabidagi 0.10 dan yuqori.

---

## 4. Leaderboard natijalari

| Run | Model | NDCG@10 | Baseline | Farq | Catalog cov. | Status |
|---|---|---|---|---|---|---|
| 43 | `als` (prior yo'q, α=20) | 0.07172 | 0.05751 | +24.7% | 0.369 | below_pass |
| 44 | `als` + pop_prior 0.9, α=30 | **0.18756** | 0.05751 | **+226.1%** | 0.0855 | excellent |
| **45** | `als` + pop_prior 0.6, α=30 | **0.18210** | 0.05751 | **+216.6%** | **0.1043** | **excellent** |

Raund 3 chegaralari (relative): o'tish = baseline × 1.4103 = **0.0811**,
a'lo = baseline × 1.7949 = **0.1032**. Run 44 va 45 ikkalasi ham chegaradan
2 baravar yuqori. TZ rubricida a'lo uchun **coverage ≥ 0.10** ham talab
qilinadi — run 44 da 0.0855 (yetarli emas), shuning uchun topshiriladigan
variant — **run 45**: NDCG 0.1821 + coverage 0.1043.

Barcha runlarda: coverage 1.000 (4 577/4 577 foydalanuvchi, 45 770 qator),
hit rate @10 = 0.573, 3 407 foydalanuvchida label bor.

Leaderboard (`eval.leaderboard`): 1-o'rin, `5-guruh t6`, run 45,
`improvement_pct = 216.65`, eng yaxshi urinish 226.13% (run 44).

Kunlik kvota: 10 tadan 3 tasi ishlatildi (7 ta qoldi).

### Nega prior shunchalik katta farq qildi?

Test labeli — "keyingi 14 kunda foydalanuvchi qaysi YANGI kontentni ko'radi".
Bunday maqsadda kontentning yaqin tarixdagi mashhurligi juda kuchli signal:
ALS "sizga o'xshagan odamlar nima ko'rdi"ni beradi, prior esa "butun
auditoriya hozir nima ko'ryapti"ni qo'shadi. Ikkalasi bir-birini to'ldiradi.
Prior juda katta bo'lsa (1.5+) tavsiyalar bir xil bo'lib qoladi va coverage
0.14 gacha tushadi.

---

## 5. Segment tahlili (`reports/segments.csv`)

Run 45 modeli, validatsiya oynasi, 3 000 foydalanuvchi:

| Segment | N | NDCG@10 |
|---|---|---|
| jami | 3 000 | 0.0956 |
| faollik: kam (≤33%) | 1 001 | **0.1105** |
| faollik: o'rta | 999 | 0.0888 |
| faollik: yuqori (top-33%) | 1 000 | 0.0874 |
| cold start (train'da yo'q) | 1 792 | 0.0928 |
| tarixi bor | 1 208 | 0.0997 |

Top-10 dagi kontent mashhurligi bo'yicha: **head 99.9%**, mid 0.09%, tail 0.03%.

**Xulosa:**

- Kam faol foydalanuvchilarda sifat **yuqoriroq** (0.1105) — ularning didi
  "ommaviy" bo'lib, prior yaxshi ishlaydi.
- Eng faol 33% da sifat pastroq (0.0874): ularning ko'p ko'rgan kontenti
  "ko'rilgan" ro'yxatidan chiqarilib, tanlov torayadi.
- Cold start foydalanuvchilar ham yaxshi natija beradi (0.0928) — mashhurlik
  fallback'i ishlayotganining isboti.
- **Asosiy cheklov:** tavsiyalar deyarli butunlay head kontentdan (top-33%
  mashhurlik) iborat. Shu sababli catalog_coverage 0.1043 — chegara yaqinida.
  Prior 0.9 da ball 0.1876 bo'ladi, lekin coverage 0.0855 ga tushadi
  (a'lo shartidan chiqadi). Kelgusi qadam: tail kontentni ko'proq ko'rsatish
  uchun prior + MMR/xilma-xillik re-rank.

## 6. Inference vaqti va yakuniy run

| Bosqich | Vaqt |
|---|---|
| Eventlarni bazadan o'qish (758 862 qator, join) | ~65 s |
| Matritsa qurish (6 983 × 3 726, 346 093 juftlik) | ~10 s |
| ALS o'qitish (32 faktor, 12 iteratsiya) | 8.0 s |
| 4 577 user uchun top-10 + yozish | ~19 s |
| **Jami (run 44)** | **102.8 s** |
| Inference (segment tahlilida o'lchangan) | 6 040 user/s |

TZ chegarasi: so'rov ≤ 15 daqiqa, ≤ 30 ulanish. Oqim bitta ulanish,
5 000 user uchun ~2 daqiqa — limitdan ancha past.

**Yakuniy run: `run_id = 45`**, `model_name = als`, `model_version =
als_pop06_v1`, round 3, NDCG@10 = 0.182101, coverage = 0.1043, status = excellent.

Hyperparametr jurnali: `reports/tuning.csv` (8 + 7 variant: faktorlar,
iteratsiyalar, α, regularizatsiya, scheme A/B, half-life, pop_prior).

## 7. Qanday takrorlash

```bash
cp .env.example .env          # mentor paroli
python -m src.export          # data/raw.npz (uyda ishlash uchun)
python -m src.audit
python -m src.evaluate --models popularity itemknn als --sample 3000
python -m src.tune --source local --sample 3000
python -m src.recommend --run-name als_pop_v1 --model als --score
python -m unittest discover -s tests -t .
```

## 8. Cheklovlar

- Offline validatsiya — bitta 14 kunlik oyna; raundlar orasida taqsimot
  siljishi (drift) bo'lsa, ball farq qiladi.
- `watch_seconds` ba'zi view'larda `NULL` — 0 deb olindi (14% past ulush).
- Baza kvotasi: kuniga 10 ta baholash; har bir run bir marta baholanadi.
- Botlar chiqarilgani uchun mashhurlik "toza" auditoriyaga asoslangan.
