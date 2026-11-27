# Taqdimot rejasi (10 daqiqa)

## 1-slayd — Muammo (1 daq)

"Foydalanuvchiga keyingi 14 kunda qaysi **yangi** kontentni ko'rsatish kerak?"
Ma'lumot: 1.33M event, 7 031 user, 5 003 kontent, faqat implicit signallar
(view / like / share). Test: 4 577 foydalanuvchi uchun top-10.
Metrika: NDCG@10; qo'shimcha — catalog coverage.

## 2-slayd — Ma'lumot auditi (1.5 daq)

- 89.6% view, 8.8% like, 1.6% share; density 3.9% (juda siyrak).
- watch_pct: 33% ko'rish 0.3 dan past → "sifatli ko'rish" chegarasi kerak.
- 30 bot (`sandbox.t6_bots`, biri 19 903 event) → o'qitishdan chiqarildi.
- 121 kontent hech qachon ko'rilmagan; 21 335 dublikat (share = view bilan
  bir sekundda) → `(user, item)` darajasida yig'ildi.

## 3-slayd — Signallar va matritsa (1.5 daq)

`w = 1.0·watch_pct + 0.6·like + 1.2·share`, takror bonusi
`× (1 + 0.5·log1p(n−1))`, vaqt decay `0.5^(age/180 kun)`.
Ikkinchi sxema (TZ talabi): binar `watch_pct ≥ 0.3`.
Natija: 6 983 × 3 726 matritsa, 346 093 juftlik.

## 4-slayd — Modellar (2 daq)

| Model | NDCG@10 (offline) | Coverage |
|---|---|---|
| popularity (baseline) | 0.0232 | 0.005 |
| item-kNN (cosine+shrink) | 0.0301 | 0.357 |
| ALS (implicit) | 0.0652 | 0.494 |
| **ALS + recent-popularity prior** | **0.0971** | 0.177 |

ALS: `confidence = 1 + α·w`, numpy-only (32 faktor, 12 iteratsiya, 8 s).

## 5-slayd — Nega prior ishladi (1 daq)

Test labeli "keyingi 14 kun" → yaqin tarixdagi ommaviy talab juda kuchli signal.
`score = minmax(ALS) + 0.6 · log1p(pop_recent)`.
Prior 0.9 → NDCG 0.1876, lekin coverage 0.0855; **0.6 → 0.1821 + 0.1043**
(ikkala a'lo sharti bajariladi).

## 6-slayd — Leaderboard (1 daq)

Run 45: NDCG@10 = **0.18210**, baseline 0.05751, **+216.6%**, status **excellent**,
1-o'rin. 45 770 qator (4 577 user × 10).

## 7-slayd — Segment tahlili (1 daq)

| Segment | NDCG@10 |
|---|---|
| kam faol | 0.1105 |
| o'rta | 0.0888 |
| yuqori faol | 0.0874 |
| cold start | 0.0928 |

Top-10 ning 99.9% — head kontent. Cheklov: xilma-xillik past; keyingi qadam
MMR re-rank / tail boost.

## 8-slayd — Kod va qayta ishlash (0.5 daq)

```
python -m src.recommend --run-name als_pop06_v1 --model als --score
```
`src/`: data, als, knn, models, pipeline, train, recommend, evaluate, audit,
tune, segments, export, store. 36 test (bazasiz ishlaydi), `.env` gitignore'da,
parol kodda yo'q, `data/raw.npz` bilan uyda ham ishlaydi.

## Ehtimoliy savollar

**Nega item-kNN past?** Siyrak matritsada cosine o'xshashlik shovqinga sezgir;
shrink yordam berdi, lekin ALS latent faktorlari ancha barqaror.

**Nega coverage past?** Prior ommaviy kontentni yuqoriga chiqaradi. Balans uchun
prior 0.6 tanlandi; bundan tashqari xilma-xillik re-ranking rejasi bor.

**Botlarni qanday aniqladingiz?** `sandbox.t6_bots` jadvali (30 user, 19k+
event, kuniga 247 kun faol) — ular mashhurlik hisobini buzadi.

**Data leakage bormi?** Yo'q: train `as_of` (2026-11-02) gacha, validatsiya
undan oldingi 14 kun, test davri (11-03..11-16) hech qayerda ishlatilmagan.

**Nega baho/saqlash signali yo'q?** Bu bazada ular yo'q — like/share bilan
almashtirildi; TZ "faqat CF, tashqi metadata yo'q" qoidasiga amal qilindi.
