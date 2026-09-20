"""
macro.py
────────
Aşama 5 – Makroekonomik Bağlam Modülü

Analistler değerleme yaparken makro ortamı da göz önünde bulundurur.
Faiz yükselirse büyüme hisselerinin değeri düşer (WACC artar, DCF
değeri düşer). Enflasyon yükselirse tüketici şirketleri zarar görür.
Bu etkiyi sayısal hale getiriyoruz.

Modül çıktıları
---------------
  1. Güncel makroekonomik göstergeler (Fed faiz, Treasury, CPI, VIX …)
  2. Sektöre özel makro hassasiyet skoru (-2 ile +2 arası)
  3. WACC ayarlaması (pp cinsinden, ±0.50 sınırlı)

Kullanım
--------
>>> from macro import compute_macro_context, get_macro_wacc_adjustment
>>> ctx = compute_macro_context("Technology")
>>> adj = get_macro_wacc_adjustment("Technology")
"""

import logging
import warnings
from typing import Any

import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

logger = logging.getLogger("macro")


# ─── Sabitler ─────────────────────────────────────────────────────────────────

# Son 5 yıl (2021–2025) ortalamaları — z-score normalizasyonu için referans
# Kaynak: FRED / Yahoo Finance tarihsel veriler
HISTORICAL_MEANS = {
    "fed_funds_rate": 3.10,    # %
    "treasury_10y":   3.50,    # %
    "cpi_yoy":        4.20,    # %
    "unemployment":   4.00,    # %
    "vix":           20.00,    # puan
    "sp500_pe":      20.00,    # forward P/E
    "usd_index":    103.00,    # DXY
    "oil_price":     75.00,    # $/varil
}

HISTORICAL_STDS = {
    "fed_funds_rate": 2.00,
    "treasury_10y":   0.80,
    "cpi_yoy":        2.50,
    "unemployment":   1.00,
    "vix":            6.00,
    "sp500_pe":       3.50,
    "usd_index":      5.00,
    "oil_price":     15.00,
}

# Makro WACC ayarlama katsayısı ve sınırları
MACRO_WACC_COEFFICIENT = -0.20   # skor × bu = pp ayarlama
MACRO_WACC_MAX_ADJ     =  0.50   # ±0.50pp sınır

# ─── Sektör Bazlı Makro Hassasiyet ───────────────────────────────────────────

SECTOR_SENSITIVITY: dict[str, dict[str, float]] = {
    "Technology": {
        "interest_rate": -0.8,   # Yüksek faiz teknoloji hisselerine çok zarar verir
        "inflation":     -0.3,
        "usd_strength":  -0.2,   # Güçlü dolar yurt dışı gelirlerini etkiler
        "vix":           -0.5,
    },
    "Energy": {
        "interest_rate": -0.3,
        "oil_price":     +0.9,   # Petrol yükselince enerji hisseleri yükselir
        "inflation":     +0.4,
        "vix":           -0.3,
    },
    "Financials": {
        "interest_rate": +0.7,   # Yüksek faiz banka marjını artırır
        "inflation":     +0.2,
        "vix":           -0.6,
    },
    "Healthcare": {
        "interest_rate": -0.2,
        "inflation":     -0.1,
        "usd_strength":  -0.1,
        "vix":           -0.2,
    },
    "Consumer Cyclical": {
        "interest_rate": -0.5,
        "inflation":     -0.6,
        "usd_strength":  -0.3,
        "vix":           -0.4,
        "oil_price":     -0.2,
    },
    "Consumer Defensive": {
        "interest_rate": -0.1,
        "inflation":     -0.2,
        "usd_strength":  -0.1,
        "vix":           -0.1,
        "oil_price":     -0.1,
    },
    "Industrials": {
        "interest_rate": -0.4,
        "inflation":     -0.2,
        "usd_strength":  -0.3,
        "vix":           -0.3,
        "oil_price":     -0.3,
    },
    "Utilities": {
        "interest_rate": -0.7,
        "inflation":     +0.1,
        "vix":           -0.2,
    },
    "Real Estate": {
        "interest_rate": -0.9,
        "inflation":     -0.1,
        "vix":           -0.3,
    },
    "Communication Services": {
        "interest_rate": -0.4,
        "inflation":     -0.2,
        "usd_strength":  -0.1,
        "vix":           -0.4,
    },
    "Basic Materials": {
        "interest_rate": -0.3,
        "inflation":     +0.3,
        "usd_strength":  -0.4,
        "vix":           -0.3,
        "oil_price":     +0.3,
    },
}

# Makro değişken → z-score eşleme
# Her makro gösterge hangi hassasiyet anahtarına karşılık geliyor
MACRO_TO_SENSITIVITY_KEY = {
    "fed_funds_rate": "interest_rate",
    "treasury_10y":   "interest_rate",  # faiz göstergesi olarak birleştirilir
    "cpi_yoy":        "inflation",
    "unemployment":   None,             # doğrudan skorda kullanılır (ters yönlü)
    "vix":            "vix",
    "sp500_pe":       None,             # referans amaçlı
    "usd_index":      "usd_strength",
    "oil_price":      "oil_price",
}


# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def _safe(value: Any, default=None) -> Any:
    if value is None:
        return default
    try:
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _r2(value: Any) -> float | None:
    v = _safe(value)
    return round(float(v), 2) if v is not None else None


def _fetch_yf_close(symbol: str, period: str = "5d") -> float | None:
    """yfinance ile en son kapanış fiyatını çek."""
    try:
        data = yf.download(symbol, period=period, progress=False)
        if data is not None and not data.empty:
            close = data["Close"]
            if hasattr(close, "iloc"):
                val = close.iloc[-1]
                # Eğer val bir Series ise (multi-level columns), ilk değeri al
                if hasattr(val, "iloc"):
                    val = val.iloc[0]
                return float(val)
    except Exception as e:
        logger.warning("yfinance %s çekilemedi: %s", symbol, e)
    return None


def _fetch_fred(series_id: str) -> float | None:
    """FRED'den en son veriyi çek (pandas_datareader ile)."""
    try:
        import pandas_datareader as pdr
        df = pdr.get_data_fred(series_id)
        if df is not None and not df.empty:
            val = df.iloc[-1].values[0]
            return float(val)
    except ImportError:
        logger.debug("pandas_datareader kurulu değil — FRED verisi atlanıyor")
    except Exception as e:
        logger.warning("FRED %s çekilemedi: %s", series_id, e)
    return None


def _fetch_fred_pct_change(series_id: str, periods: int = 12) -> float | None:
    """FRED'den yıllık % değişim hesapla (CPI gibi)."""
    try:
        import pandas_datareader as pdr
        df = pdr.get_data_fred(series_id)
        if df is not None and len(df) > periods:
            pct = df.pct_change(periods).iloc[-1].values[0]
            return float(pct) * 100  # yüzde olarak
    except ImportError:
        logger.debug("pandas_datareader kurulu değil — FRED verisi atlanıyor")
    except Exception as e:
        logger.warning("FRED %s yıllık değişim hesaplanamadı: %s", series_id, e)
    return None


# ─── Ana veri toplama ─────────────────────────────────────────────────────────

def collect_macro_data() -> dict:
    """
    Güncel makroekonomik göstergeleri topla.

    Hibrit yaklaşım: önce pandas_datareader/FRED dene, başarısız olursa
    yfinance veya fallback sabit değerler kullan.

    Döndürür
    --------
    dict : Makro gösterge adı → değer eşlemesi
    """
    logger.info("▶ Makro veriler toplanıyor…")

    data: dict[str, float | None] = {}
    sources: dict[str, str] = {}

    # ── 1. Fed Funds Rate ────────────────────────────────────────────────
    val = _fetch_fred("FEDFUNDS")
    if val is not None:
        data["fed_funds_rate"] = val
        sources["fed_funds_rate"] = "FRED"
    else:
        # Fallback: ^IRX (3-month T-bill) yaklaşık olarak
        val = _fetch_yf_close("^IRX")
        if val is not None:
            data["fed_funds_rate"] = val
            sources["fed_funds_rate"] = "yfinance (^IRX proxy)"
        else:
            data["fed_funds_rate"] = None
            sources["fed_funds_rate"] = "unavailable"

    # ── 2. 10Y Treasury (^TNX) ───────────────────────────────────────────
    val = _fetch_yf_close("^TNX")
    if val is not None:
        data["treasury_10y"] = val
        sources["treasury_10y"] = "yfinance (^TNX)"
    else:
        data["treasury_10y"] = None
        sources["treasury_10y"] = "unavailable"

    # ── 3. CPI YoY ──────────────────────────────────────────────────────
    val = _fetch_fred_pct_change("CPIAUCSL", periods=12)
    if val is not None:
        data["cpi_yoy"] = val
        sources["cpi_yoy"] = "FRED"
    else:
        data["cpi_yoy"] = None
        sources["cpi_yoy"] = "unavailable"

    # ── 4. Unemployment Rate ─────────────────────────────────────────────
    val = _fetch_fred("UNRATE")
    if val is not None:
        data["unemployment"] = val
        sources["unemployment"] = "FRED"
    else:
        data["unemployment"] = None
        sources["unemployment"] = "unavailable"

    # ── 5. VIX ───────────────────────────────────────────────────────────
    val = _fetch_yf_close("^VIX")
    if val is not None:
        data["vix"] = val
        sources["vix"] = "yfinance (^VIX)"
    else:
        data["vix"] = None
        sources["vix"] = "unavailable"

    # ── 6. S&P 500 Forward P/E (yaklaşık) ────────────────────────────────
    # yfinance üzerinden doğrudan alınamaz; SPY'ın trailing P/E'si kullanılır
    try:
        spy_info = yf.Ticker("SPY").info
        sp500_pe = _safe(spy_info.get("trailingPE"))
        if sp500_pe:
            data["sp500_pe"] = float(sp500_pe)
            sources["sp500_pe"] = "yfinance (SPY trailing P/E)"
        else:
            data["sp500_pe"] = None
            sources["sp500_pe"] = "unavailable"
    except Exception:
        data["sp500_pe"] = None
        sources["sp500_pe"] = "unavailable"

    # ── 7. USD Index (DXY) ───────────────────────────────────────────────
    val = _fetch_yf_close("DX-Y.NYB")
    if val is not None:
        data["usd_index"] = val
        sources["usd_index"] = "yfinance (DX-Y.NYB)"
    else:
        data["usd_index"] = None
        sources["usd_index"] = "unavailable"

    # ── 8. Oil Price (WTI Crude) ─────────────────────────────────────────
    val = _fetch_yf_close("CL=F")
    if val is not None:
        data["oil_price"] = val
        sources["oil_price"] = "yfinance (CL=F)"
    else:
        data["oil_price"] = None
        sources["oil_price"] = "unavailable"

    # Çekilen gösterge sayısı
    available = sum(1 for v in data.values() if v is not None)
    logger.info(
        "✔ Makro veriler toplandı: %d/%d gösterge mevcut",
        available, len(data),
    )

    return {"values": data, "sources": sources}


# ─── Z-Score Normalizasyonu ───────────────────────────────────────────────────

def _compute_z_scores(macro_values: dict[str, float | None]) -> dict[str, float | None]:
    """
    Her makro gösterge için z-score hesapla.

    z = (değer − ortalama) / std

    Pozitif z-score → tarihsel ortalamanın üzerinde
    Negatif z-score → tarihsel ortalamanın altında
    """
    z_scores: dict[str, float | None] = {}

    for key, value in macro_values.items():
        if value is None:
            z_scores[key] = None
            continue

        mean = HISTORICAL_MEANS.get(key)
        std = HISTORICAL_STDS.get(key)

        if mean is None or std is None or std == 0:
            z_scores[key] = None
            continue

        z = (float(value) - mean) / std
        z_scores[key] = round(z, 2)

    return z_scores


# ─── Makro Ortam Skoru ────────────────────────────────────────────────────────

def _compute_macro_score(
    z_scores: dict[str, float | None],
    sector: str,
) -> tuple[float, dict[str, float]]:
    """
    Sektör hassasiyetiyle ağırlıklandırılmış makro ortam skoru hesapla.

    Her z-score ilgili sektör hassasiyet katsayısıyla çarpılır ve toplanır.

    Parametreler
    ------------
    z_scores : Makro gösterge z-score'ları
    sector   : Şirketin sektörü (yfinance formatında)

    Döndürür
    --------
    (toplam_skor, katkı_dict)
    """
    sensitivity = SECTOR_SENSITIVITY.get(sector, {})

    # Sektör bulunamazsa varsayılan ağırlıklar
    if not sensitivity:
        logger.warning(
            "Sektör '%s' hassasiyet tablosunda bulunamadı — varsayılan kullanılıyor",
            sector,
        )
        sensitivity = {
            "interest_rate": -0.4,
            "inflation":     -0.2,
            "usd_strength":  -0.2,
            "vix":           -0.3,
        }

    contributions: dict[str, float] = {}
    total_score = 0.0

    # Faiz göstergeleri (fed_funds ve treasury) birleştirilir — ortalaması alınır
    interest_z_values = []
    for key in ("fed_funds_rate", "treasury_10y"):
        z = z_scores.get(key)
        if z is not None:
            interest_z_values.append(z)

    if interest_z_values and "interest_rate" in sensitivity:
        avg_interest_z = sum(interest_z_values) / len(interest_z_values)
        contrib = avg_interest_z * sensitivity["interest_rate"]
        contributions["interest_rate"] = round(contrib, 3)
        total_score += contrib

    # CPI → inflation hassasiyeti
    cpi_z = z_scores.get("cpi_yoy")
    if cpi_z is not None and "inflation" in sensitivity:
        contrib = cpi_z * sensitivity["inflation"]
        contributions["inflation"] = round(contrib, 3)
        total_score += contrib

    # VIX → vix hassasiyeti
    vix_z = z_scores.get("vix")
    if vix_z is not None and "vix" in sensitivity:
        contrib = vix_z * sensitivity["vix"]
        contributions["vix"] = round(contrib, 3)
        total_score += contrib

    # USD Index → usd_strength hassasiyeti
    usd_z = z_scores.get("usd_index")
    if usd_z is not None and "usd_strength" in sensitivity:
        contrib = usd_z * sensitivity["usd_strength"]
        contributions["usd_strength"] = round(contrib, 3)
        total_score += contrib

    # Oil Price → oil_price hassasiyeti
    oil_z = z_scores.get("oil_price")
    if oil_z is not None and "oil_price" in sensitivity:
        contrib = oil_z * sensitivity["oil_price"]
        contributions["oil_price"] = round(contrib, 3)
        total_score += contrib

    # İşsizlik → doğrudan katkı (yüksek işsizlik = negatif)
    unemp_z = z_scores.get("unemployment")
    if unemp_z is not None:
        contrib = unemp_z * -0.2   # sabit ağırlık
        contributions["unemployment"] = round(contrib, 3)
        total_score += contrib

    return round(total_score, 2), contributions


def _macro_label(score: float) -> str:
    """Makro ortam skoruna göre etiket döndür."""
    if score >= 1.0:
        return "Çok Elverişli"
    elif score >= 0.3:
        return "Elverişli"
    elif score >= -0.3:
        return "Nötr"
    elif score >= -1.0:
        return "Olumsuz"
    else:
        return "Çok Olumsuz"


def _direction_icon(z_score: float | None) -> str:
    """Z-score yönünü gösteren ikon döndür."""
    if z_score is None:
        return "─"
    if z_score > 0.5:
        return "▲▲"
    elif z_score > 0:
        return "▲"
    elif z_score > -0.5:
        return "▼"
    else:
        return "▼▼"


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_macro_context(sector: str) -> dict:
    """
    Tam makroekonomik bağlam analizi.

    Parametreler
    ------------
    sector : Şirketin sektörü (ör: "Technology", "Energy")

    Döndürür
    --------
    dict : {
        "macro_data":     {gösterge → değer},
        "sources":        {gösterge → kaynak},
        "z_scores":       {gösterge → z-score},
        "sector":         sektör adı,
        "sensitivity":    sektöre uygulanan hassasiyet katsayıları,
        "contributions":  {hassasiyet → skor katkısı},
        "macro_score":    toplam skor (genellikle -2 ile +2),
        "macro_label":    skor etiketi,
        "wacc_adjustment_pp": WACC ayarlaması (pp cinsinden),
    }
    """
    logger.info("▶ Makro bağlam hesaplanıyor (sektör: %s)…", sector)

    # 1. Makro verileri topla
    raw = collect_macro_data()
    macro_values = raw["values"]
    sources = raw["sources"]

    # 2. Z-score normalizasyonu
    z_scores = _compute_z_scores(macro_values)

    # 3. Sektöre özel makro skoru
    macro_score, contributions = _compute_macro_score(z_scores, sector)

    # 4. Etiket
    macro_label = _macro_label(macro_score)

    # 5. WACC ayarlaması
    wacc_adj = macro_score * MACRO_WACC_COEFFICIENT
    wacc_adj = max(-MACRO_WACC_MAX_ADJ, min(MACRO_WACC_MAX_ADJ, wacc_adj))
    wacc_adj = round(wacc_adj, 2)

    logger.info(
        "✔ Makro skor: %.2f (%s) | WACC ayarlama: %+.2f pp | sektör: %s",
        macro_score, macro_label, wacc_adj, sector,
    )

    return {
        "macro_data":         {k: _r2(v) for k, v in macro_values.items()},
        "sources":            sources,
        "z_scores":           z_scores,
        "sector":             sector,
        "sensitivity":        SECTOR_SENSITIVITY.get(sector, {}),
        "contributions":      contributions,
        "macro_score":        macro_score,
        "macro_label":        macro_label,
        "wacc_adjustment_pp": wacc_adj,
    }


def get_macro_wacc_adjustment(sector: str) -> float:
    """
    Sadece WACC ayarlamasını döndür (pp cinsinden).

    Pozitif değer → WACC artır (makro olumsuz)
    Negatif değer → WACC düşür (makro elverişli)

    Parametreler
    ------------
    sector : Şirketin sektörü

    Döndürür
    --------
    float : WACC ayarlaması (pp), ±0.50 sınırlı
    """
    try:
        ctx = compute_macro_context(sector)
        return ctx["wacc_adjustment_pp"]
    except Exception as e:
        logger.error("Makro WACC ayarlaması hesaplanamadı: %s", e)
        return 0.0


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    # Kullanım: python3 macro.py [TICKER]
    # Ticker verilirse sektörünü otomatik çeker; yoksa Technology varsayar
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    sector = "Technology"  # varsayılan

    if ticker:
        try:
            info = yf.Ticker(ticker.upper()).info
            sector = info.get("sector", "Technology")
            print(f"\n  Ticker: {ticker.upper()} | Sektör: {sector}\n")
        except Exception:
            pass

    result = compute_macro_context(sector)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
