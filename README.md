# digital_analyst

**Python Finansal Analiz Sistemi** — Gerçek analistler gibi hisse senetlerini değerlendiren, 12 aylık kısa vadeli trading odaklı bir hedef fiyat belirleme ve tam analist raporu üretme sistemi.

---

## Modüller

| Modül | Açıklama | Durum |
|-------|----------|-------|
| `data_collector.py` | Veri toplama pipeline (yfinance) | ✅ |
| `analyzer.py` | DCF + Çarpan + Comps + Duyarlılık matrisi | ✅ |
| `comps.py` | Comparable Company Analysis (Peer karşılaştırma) | ✅ |
| `report_generator.py` | Renkli terminal analist raporu | ✅ |
| `screener.py` | Paralel toplu tarayıcı + Kalite Skoru | ✅ |
| `main.py` | Tek hisse CLI | ✅ |

---

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Kullanım

### Tek hisse raporu

```bash
python3 main.py MSFT
python3 main.py AAPL --save aapl_report.txt
python3 main.py TSLA --json
```

### Karşılaştırmalı analiz

```bash
python3 main.py MSFT AAPL GOOGL
```

### Toplu Tarayıcı (Screener)

```bash
# Doğrudan hisseler
python3 screener.py MSFT AAPL GOOGL META NVDA

# Watchlist dosyasından
python3 screener.py --watchlist watchlist.txt

# JSON olarak kaydet
python3 screener.py MSFT AAPL --save results.json

# Paralel worker sayısı
python3 screener.py --watchlist watchlist.txt --workers 6
```

### Python API

```python
# Tek hisse tam analiz + rapor
from report_generator import generate_report
generate_report("MSFT", save_path="msft_report.txt")

# Toplu tarama
from screener import Screener
sc = Screener(["MSFT", "AAPL", "GOOGL", "META", "NVDA"])
results = sc.run()
sc.print_summary(results)
sc.save_json(results, "screener_results.json")

# Sadece kalite skoru
from screener import compute_quality_score, quality_grade
from data_collector import collect_ticker_data
from analyzer import Analyzer
data     = collect_ticker_data("MSFT")
analysis = Analyzer(data).analyze()
analysis["_raw_data"] = data
qs = compute_quality_score(analysis)
print(f"Kalite Skoru: {qs['composite']} / {quality_grade(qs['composite'])}")
```

---

## Değerleme Metodolojisi

### DCF (2-Aşamalı + Gordon Growth)

```
Stage 1 (Yıl 1–5)  : stage1_growth (yüksek büyüme)
Stage 2 (Yıl 6–10) : stage2_growth (yavaşlama)
Terminal Value      : TV = FCF₁₀ × (1+g) / (WACC – g)
```

| Senaryo | Stage1 | Stage2 | Terminal |
|---------|--------|--------|----------|
| 🐂 Bull | %18    | %10    | %3.0     |
| ⚖ Base  | %12    | %7     | %2.5     |
| 🐻 Bear | %6     | %4     | %2.0     |

> Senaryo parametreleri şirketin 3Y Revenue CAGR'ına göre otomatik ayarlanır.

### WACC (CAPM)

```
Re   = Rf + β × ERP      (ERP = %5.5, Damodaran)
Rf   = Güncel ^TNX (10Y ABD Hazine, canlı)
WACC = (E/V) × Re + (D/V) × Rd × (1 – Tax Rate)
```

### Duyarlılık Matrisi

5×5 WACC × Terminal Growth Rate matrisi. Baz senaryodan ±2pp WACC ve ±1pp terminal büyüme değişiminde içsel fiyat. Mevcut fiyata göre renklendirme:
- 🟢 Yeşil: +%10 üstü değer
- 🟡 Sarı: mevcut fiyat üstü
- 🔴 Kırmızı: mevcut fiyat altı

### Hedef Fiyat

```
Hedef = %50 × DCF (ağırlıklı senaryo) + %25 × Çarpan ortalaması + %25 × Comps
```

**Çarpan yöntemleri:** Fwd P/E × NTM EPS · P/S × NTM Rev · EV/EBITDA · Konsensüs

### Comps (Comparable Company Analysis)

Hedef şirketin sektörel peer’larıyla karşılaştırmalı göreli değerleme:

1. **Peer Grubu**: Sektör/Endüstri bazında 20 sektör grubundan otomatik seçim
2. **Çarpan Toplama**: Forward P/E, EV/EBITDA, EV/Revenue, Price/FCF, PEG (paralel çekme)
3. **Outlier Filtre**: P/E > 200 veya negatif değerler dışlanır
4. **Büyüme Primi**: `(hedef_büyüme – peer_medyan_büyüme) / peer_medyan_büyüme` (±50% sınırlı)
5. **İma Edilen Fiyat**: Düzeltilmiş çarpan × hedef metrik
6. **Comps Fair Value**: İma edilen fiyatların medyanı

### Öneri Eşikleri

| Upside    | Öneri        |
|-----------|--------------|
| ≥ +20%    | STRONG BUY   |
| ≥ +10%    | BUY          |
| ≥ −5%     | HOLD         |
| ≥ −15%    | SELL         |
| < −15%    | STRONG SELL  |

---

## Temel Kalite Skoru (0–100)

| Kriter | Ağırlık | Maks Sınır |
|--------|---------|------------|
| ROE | %20 | ≥30% → 100p |
| ROIC | %20 | ≥20% → 100p |
| Revenue 3Y CAGR | %15 | ≥20% → 100p |
| EPS 3Y CAGR | %15 | ≥20% → 100p |
| Debt/EBITDA (ters) | %15 | ≤1x → 100p |
| FCF Marjı | %15 | ≥25% → 100p |

**Harf Notu:** A+ (≥85) · A (≥75) · B+ (≥65) · B (≥55) · C+ (≥45) · C (≥35) · D

---

## Örnek Screener Çıktısı

```
══════════════════════════════════════════════════════════════════
                      SCREENER SONUÇLARI
══════════════════════════════════════════════════════════════════

  Ticker  İsim               Mevcut   Hedef  Upside  Öneri       Kalite  Not
  META    Meta Platforms    $602.61 $579.74   -3.8%  HOLD          98.7  A+
  MSFT    Microsoft         $417.42 $313.50  -24.9%  STRONG SELL   88.5  A+
  NVDA    NVIDIA            $220.61 $160.21  -27.4%  STRONG SELL  100.0  A+
  AAPL    Apple Inc.        $298.97 $210.28  -29.7%  STRONG SELL   75.8   A
  GOOGL   Alphabet Inc.     $387.66 $248.24  -36.0%  STRONG SELL   90.3  A+

  ▲ En Yüksek Upside : META (-3.8%)
  ★ En Yüksek Kalite : NVDA (100.0 / A+)
```

---

## Proje Yapısı

```
digital_analyst/
├── main.py              # Tek hisse CLI
├── screener.py          # Toplu tarayıcı + Kalite Skoru
├── data_collector.py    # Aşama 1 – Veri toplama
├── analyzer.py          # Aşama 2 – DCF + Çarpan + Comps + Duyarlılık
├── comps.py             # Aşama 3 – Comparable Company Analysis
├── report_generator.py  # Aşama 4 – Analist raporu
├── watchlist.txt        # Örnek izleme listesi
├── requirements.txt
├── __init__.py
└── README.md
```
