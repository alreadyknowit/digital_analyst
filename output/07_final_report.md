# Digital Analyst -- Final Rapor

> **Olusturulma:** 2026-09-20T16:50:58
> **Pipeline:** 06_fusion_layer -> 07_investment_engine
> **Hisseler:** TSLA

---

## Executive Summary

Bu rapor **1 FAANG hissesinin** kapsamli multi-modal analizini sunar. Attention-based feature fusion (51 ozellik, 6 modul) uzerine kurulu karar motoru her hisse icin AL/SAT/BEKLE sinyali uretmektedir.

- **En iyi firsat:** `TSLA` -- +9.70% beklenti | Sinyal: **AL**
- **Ortalama beklenti:** +9.70%
- **Ortalama guven:** 85.0%
- AL onerileri: TSLA
- SAT onerileri: Yok
- BEKLE onerileri: Yok

---

## Karar Motoru Sonuclari

| Ticker | Mevcut $ | Hedef $ | Beklenti | Guven | Risk | Sinyal |
|--------|----------|---------|----------|-------|------|--------|
| **TSLA** | $364.27 | $399.59 | +9.70% | 85.0% | 34.6% | **AL** (100%) |

---

## Backtesting Sonuclari (1Y Simulasyon)

> **Baslangic Sermayesi:** $100,000 | **Islem Maliyeti:** 0.1% | **Stop-Loss:** $5

| Ticker | Bitis $ | Getiri | B&H | Sharpe | MaxDD | Islem | Win% |
|--------|---------|--------|-----|--------|-------|-------|------|
| **TSLA** | $78,972 | -21.0% | -14.5% | -0.74 | -29.2% | 40 | 12% |

---

## En Iyi Model Konfigurasyonu

| Parametre | Deger |
|-----------|-------|
| Konfigurasyon | `02_tech_correlation` |
| Fusion Yontemi | `multi_head` |
| MAPE | **265.0500%** |
| Feature Sayisi | 10 |

---

## Akademik Model Karsilastirmasi

| Model | MAPE (%) | Sharpe | MaxDD (%) | Kaynak |
|-------|----------|--------|-----------|--------|
| ARIMA (baseline) | 2.5 | 0.65 | -18.5 | Lin et al. 2021, literatür ortalaması |
| LSTM (vanilla) | 1.8 | 0.82 | -15.2 | Fischer & Krauss 2018, J. Financial Economics |
| Pin-GAN (orijinal) | 1.2 | 1.05 | -12.8 | Koshiyama et al. 2021, arXiv:2106.01127 |
| Transformer (FinBERT) | 1.05 | 1.18 | -11.4 | Yang et al. 2023, Applied Soft Computing |
| **DigitalAnalyst (bizim -- 06+07)** | **265.05** | -0.74 | -29.2 | Bu calisma -- Ablation 06_fusion_layer.py |

---

## Metodoloji

```
  Modul 01: Teknik Analiz      (10 ozellik)
  Modul 02: Cross-Asset Korel. ( 6 ozellik)
  Modul 03: FinBERT Sentiment  ( 6 ozellik)
  Modul 04: Makro Gostergeler  (21 ozellik)
  Modul 05: Temel Analiz+ESG   ( 8 ozellik)
            --------------------------------
  TOPLAM                       51 ozellik
            v
  ConcatFusion / MultiHeadFusion --> 128-dim context
            v
  Karar Motoru --> AL / SAT / BEKLE
```

---

## Risk Uyarisi

> Bu rapor **egitim ve arastirma amaclidir**. Gercek yatirim karari icin kullanilmamalidir.
> Model tahminleri gecmis verilere dayalidir ve gelecekteki performansi garanti etmez.

*Rapor olusturuldu: 2026-09-20T16:50:58 | Digital Analyst v0.7*