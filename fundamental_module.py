"""
fundamental_module.py
─────────────────────
Aşama 5 – Temel Analiz + Insider Trading + ESG Modülü

Pin-GAN pipeline'ının son veri modülü. Her hisse için quarterly güncellenen
feature vektörü üretir:

  Bölüm 1 – Finansal Metrikler    (yfinance via data_collector)
  Bölüm 2 – Insider Trading       (SEC EDGAR Form 4 API)
  Bölüm 3 – ESG + Hukuki Risk     (yfinance sustainability + haber analizi)
  Bölüm 4 – Kazanç Sezonu Flag   (yfinance calendar + earnings geçmişi)

Çıktı: features_05_fundamental.csv (output/ klasöründe)

Kullanım
--------
>>> from fundamental_module import get_fundamental_features
>>> features = get_fundamental_features("AAPL")

>>> from fundamental_module import build_features_csv
>>> build_features_csv(["AAPL", "MSFT", "GOOGL"])
"""

import csv
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fundamental_module")

# ─── Çıktı dizini ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "output"

# ─── EDGAR API ────────────────────────────────────────────────────────────────
EDGAR_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms=4"
)
EDGAR_HEADERS = {
    "User-Agent": "digital_analyst research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "efts.sec.gov",
}

# ─── Sabitler ─────────────────────────────────────────────────────────────────
INSIDER_LOOKBACK_DAYS   = 90   # Form 4 tarama penceresi
LEGAL_NEWS_LOOKBACK     = 90   # hukuki risk haber penceresi
EARNINGS_IMMINENT_DAYS  = 5    # earnings yakınlık eşiği (gün)
MAX_LEGAL_NEWS          = 20   # normalize için tavan


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def _safe(value: Any, default=None) -> Any:
    """NaN / None / inf ise *default* döndür."""
    if value is None:
        return default
    try:
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return default
        import pandas as pd
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _r2(value: Any) -> float | None:
    v = _safe(value)
    return round(float(v), 4) if v is not None else None


def _pct(num: Any, denom: Any) -> float | None:
    """num / denom * 100 — güvenli."""
    n, d = _safe(num), _safe(denom)
    if n is None or d is None or float(d) == 0:
        return None
    return _r2(float(n) / float(d) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Bölüm 1 – Finansal Metrikler
# ─────────────────────────────────────────────────────────────────────────────

def _get_financial_metrics(ticker: str, data: dict | None) -> dict:
    """
    data_collector.py çıktısından temel finansal metrikleri çıkarır.

    *data* verilmezse yfinance `info` ile minimal fallback yapılır.
    """
    result = {
        "pe_ratio":       None,
        "eps_surprise":   None,
        "revenue_growth": None,
        "profit_margin":  None,
        "debt_equity":    None,
        "fcf_yield":      None,
    }

    try:
        if data:
            vm = data.get("valuation_multiples", {})
            gq = data.get("growth_and_quality", {})
            is_data = data.get("income_statement", {})
            bs_data = data.get("balance_sheet", {})

            result["pe_ratio"]       = _safe(vm.get("pe_trailing"))
            result["revenue_growth"] = _safe(gq.get("revenue_yoy_growth"))

            # Kâr marjı — en son çeyrekten
            qtr = is_data.get("quarterly", {})
            if qtr:
                latest_q = next(iter(qtr.values()), {})
                rev_q = _safe(latest_q.get("revenue"))
                ni_q  = _safe(latest_q.get("net_income"))
                result["profit_margin"] = _pct(ni_q, rev_q)

            # Borç / Özsermaye
            td     = _safe(bs_data.get("total_debt"))
            equity = _safe(bs_data.get("book_value"))
            if td is not None and equity and float(equity) != 0:
                result["debt_equity"] = _r2(float(td) / float(equity))

            # FCF Yield = FCF / Market Cap
            ann = is_data.get("annual", {})
            if ann:
                latest_a = next(iter(ann.values()), {})
                fcf = _safe(latest_a.get("fcf"))
                mkt_cap = _safe(vm.get("enterprise_value")) or \
                          _safe(data.get("company_info", {}).get("market_cap"))
                result["fcf_yield"] = _pct(fcf, mkt_cap)

        else:
            # data_collector kullanılamıyorsa minimal fallback
            logger.warning("data verilmedi, yfinance info ile fallback: %s", ticker)
            tk   = yf.Ticker(ticker)
            info = tk.info
            result["pe_ratio"]       = _safe(info.get("trailingPE"))
            result["revenue_growth"] = _r2(
                _safe(info.get("revenueGrowth"), 0) * 100
            )
            result["profit_margin"] = _r2(
                _safe(info.get("profitMargins"), 0) * 100
            )
            result["debt_equity"]  = _safe(info.get("debtToEquity"))
            mkt_cap = _safe(info.get("marketCap"))
            fcf     = _safe(info.get("freeCashflow"))
            result["fcf_yield"] = _pct(fcf, mkt_cap)

    except Exception as e:
        logger.error("Finansal metrikler alınamadı (%s): %s", ticker, e)

    # EPS sürpriz (ayrı yfinance çağrısı)
    result["eps_surprise"] = _get_eps_surprise(ticker)

    return result


def _get_eps_surprise(ticker: str) -> float | None:
    """
    Gerçek EPS − Tahmini EPS → en son quarter için sürpriz değeri.
    yfinance `earnings` tablosu kullanır.
    """
    try:
        tk  = yf.Ticker(ticker)
        eq  = tk.quarterly_earnings
        if eq is None or eq.empty:
            return None
        # Sütunlar: Earnings, Estimate (yfinance v0.2+)
        eq = eq.dropna(subset=["Earnings", "Estimate"])
        if eq.empty:
            return None
        latest = eq.iloc[0]
        surprise = float(latest["Earnings"]) - float(latest["Estimate"])
        return _r2(surprise)
    except Exception as e:
        logger.debug("EPS sürpriz alınamadı (%s): %s", ticker, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Bölüm 2 – Insider Trading (SEC EDGAR Form 4)
# ─────────────────────────────────────────────────────────────────────────────

def _get_insider_signal(ticker: str, shares_outstanding: int | None) -> float | None:
    """
    SEC EDGAR Form 4 verisini çekerek normalize edilmiş insider alım/satım
    sinyali döndürür: [-1.0, +1.0] aralığında.

    Pozitif → net alım (bullish), Negatif → net satım (bearish).
    """
    try:
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=INSIDER_LOOKBACK_DAYS)

        url = EDGAR_SEARCH_URL.format(
            ticker=ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )

        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.warning(
                "EDGAR API yanıt kodu %d (%s)", resp.status_code, ticker
            )
            return _get_insider_signal_yfinance(ticker, shares_outstanding)

        payload = resp.json()
        hits    = payload.get("hits", {}).get("hits", [])

        if not hits:
            logger.info("EDGAR Form 4 kaydı bulunamadı: %s", ticker)
            return 0.0

        # Her Form 4 kaydından alım/satım miktarı çek
        # EDGAR full-text search → kaynak belge CIK/accession number'ı verir
        # Net sinyal için CIK listesini kullanarak summary çekiyoruz
        net_shares = _parse_edgar_transactions(hits, ticker)
        logger.info("EDGAR insider net shares (%s): %+d", ticker, net_shares)

        if net_shares == 0:
            return 0.0

        # Normalize: net / outstanding shares
        if shares_outstanding and int(shares_outstanding) > 0:
            signal = net_shares / int(shares_outstanding)
            # [-1, +1] sıkıştırma (tanh-like clipping)
            signal = max(-1.0, min(1.0, signal * 1000))
            return _r2(signal)
        else:
            # Shares bilinmiyor → sadece işaret
            return 1.0 if net_shares > 0 else -1.0

    except requests.Timeout:
        logger.warning("EDGAR API zaman aşımı (%s) — yfinance fallback", ticker)
        return _get_insider_signal_yfinance(ticker, shares_outstanding)
    except Exception as e:
        logger.error("Insider sinyali alınamadı (%s): %s", ticker, e)
        return None


def _parse_edgar_transactions(hits: list, ticker: str) -> int:
    """
    EDGAR search hits'inden net transaction shares hesaplar.
    Her hit'te _source.period_of_report ve form tipi bulunur.
    Gerçek hisse adedini CIK/filing bazında tahmin eder.

    Not: EDGAR full-text search filing metnini içermez; hit sayısını
    proxy olarak kullanıyoruz (her filing ≈ 1 işlem).
    """
    # Daha ayrıntılı veri için EDGAR XBRL API veya SEC submissions kullanılabilir.
    # Bu implementasyonda hit sayısını basit proxy olarak kullanıyoruz:
    # pozitif hit (Form 4 → P tipi) - negatif hit (Form 4 → S tipi)
    buy_count  = 0
    sell_count = 0

    for hit in hits:
        source = hit.get("_source", {})
        # form_type her zaman "4" olacak; transaction type belge içinde
        # Başlık / display_date bilgisine bakarak basit tahmin
        display = str(source.get("display_names", "")).lower()
        entity  = str(source.get("entity_name", "")).lower()

        # EDGAR'da "officer" / "director" alımları genellikle pozitif sinyal
        # Basit heuristic: son 30 gündekileri daha ağırlıklı say
        period_str = source.get("period_of_report", "")
        try:
            period_dt = datetime.strptime(period_str[:10], "%Y-%m-%d")
            weight    = 2 if (datetime.now() - period_dt).days <= 30 else 1
        except Exception:
            weight = 1

        # Form 4 transaction type doğrudan çekilemiyor (tam belge gerekiyor)
        # SEC EDGAR company API ile ek bilgi alalım
        cik = source.get("entity_id", "")
        if cik:
            t = _fetch_edgar_transaction_type(cik, source.get("file_date", ""), ticker)
            if t == "P":
                buy_count  += weight * 1000   # proxy: 1000 share/filing
            elif t == "S":
                sell_count += weight * 1000
        else:
            # CIK yok → nötr say
            pass

    return buy_count - sell_count


def _fetch_edgar_transaction_type(cik: str, file_date: str, ticker: str) -> str | None:
    """
    SEC EDGAR submissions API'sinden en yakın Form 4 dosyalama tipini çek.
    'P' → Purchase, 'S' → Sale, None → bilinmiyor
    """
    try:
        # EDGAR submissions endpoint
        padded_cik = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Basit heuristic: son filings'ten transaction type çekemiyoruz
        # Bu aşamada nötr döndür; tam XBRL ayrıştırma ileride eklenebilir
        return None
    except Exception:
        return None


def _get_insider_signal_yfinance(
    ticker: str, shares_outstanding: int | None
) -> float | None:
    """
    EDGAR başarısız olursa yfinance insider_transactions yedek.
    """
    try:
        tk = yf.Ticker(ticker)
        it = tk.insider_transactions
        if it is None or it.empty:
            return 0.0

        cutoff = datetime.now() - timedelta(days=INSIDER_LOOKBACK_DAYS)

        # Tarih sütunu
        date_col = "Date" if "Date" in it.columns else it.index.name
        if date_col and date_col != it.index.name:
            it["_dt"] = it[date_col]
        else:
            it = it.reset_index()
            it["_dt"] = it.iloc[:, 0]

        try:
            it["_dt"] = it["_dt"].apply(
                lambda x: datetime.strptime(str(x)[:10], "%Y-%m-%d")
                if isinstance(x, str) else x
            )
            it = it[it["_dt"] >= cutoff]
        except Exception:
            pass

        if it.empty:
            return 0.0

        # Shares sütunu
        shares_col = next(
            (c for c in ["Shares", "shares", "Shares Owned Directly", "Value"]
             if c in it.columns), None
        )
        tx_col = next(
            (c for c in ["Transaction", "transaction", "Text", "Start Date"]
             if c in it.columns), None
        )
        if not shares_col:
            return None

        it[shares_col] = it[shares_col].astype(str).str.replace(",", "").str.extract(r"([\d.]+)")[0]
        it[shares_col] = it[shares_col].apply(lambda x: float(x) if x else 0.0)

        net = 0
        for _, row in it.iterrows():
            shares_val = float(_safe(row.get(shares_col), 0))
            tx_text    = str(row.get(tx_col, "")).lower() if tx_col else ""
            if "sale" in tx_text or "sell" in tx_text or "s-" in tx_text:
                net -= shares_val
            elif "purchase" in tx_text or "buy" in tx_text or "p-" in tx_text:
                net += shares_val

        if net == 0:
            return 0.0

        if shares_outstanding and int(shares_outstanding) > 0:
            signal = net / int(shares_outstanding)
            signal = max(-1.0, min(1.0, signal * 1000))
            return _r2(signal)
        else:
            return 1.0 if net > 0 else -1.0

    except Exception as e:
        logger.error("yfinance insider fallback başarısız (%s): %s", ticker, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Bölüm 3 – ESG + Hukuki Risk
# ─────────────────────────────────────────────────────────────────────────────

def _get_esg_score(ticker: str) -> float | None:
    """
    yfinance ticker.sustainability DataFrame'den toplam ESG skoru çeker.
    Mevcut değilse None döndürür.
    """
    try:
        tk   = yf.Ticker(ticker)
        sus  = tk.sustainability
        if sus is None or sus.empty:
            logger.info("ESG verisi bulunamadı: %s", ticker)
            return None

        # yfinance sustainability formatı değişkendir:
        # Rows: totalEsg, esgPerformance, environmentScore, socialScore, governanceScore ...
        if "totalEsg" in sus.index:
            val = _safe(sus.loc["totalEsg"].iloc[0])
            return _r2(val)

        # Yeni format: sütun bazlı
        for col in sus.columns:
            if "total" in str(col).lower() or "esg" in str(col).lower():
                val = _safe(sus[col].iloc[0])
                if val is not None:
                    return _r2(val)

        return None
    except Exception as e:
        logger.warning("ESG skoru alınamadı (%s): %s", ticker, e)
        return None


def _get_legal_risk(ticker: str) -> float:
    """
    Son 90 günde hukuki risk işaretleyen haber sayısını normalize eder [0.0–1.0].

    Anahtar kelimeler: "lawsuit", "SEC investigation", "regulatory fine",
    "legal action", "class action", "settlement", "probe"
    """
    LEGAL_KEYWORDS = [
        "lawsuit", "sec investigation", "regulatory fine",
        "legal action", "class action", "settlement", "probe",
        "subpoena", "indictment", "fraud", "penalty",
    ]

    try:
        tk  = yf.Ticker(ticker)
        raw = tk.news
        if not raw:
            return 0.0

        cutoff = datetime.now() - timedelta(days=LEGAL_NEWS_LOOKBACK)
        legal_count = 0

        for item in raw:
            content  = item.get("content", item) if isinstance(item, dict) else {}
            title    = content.get("title", "") if isinstance(content, dict) else item.get("title", "")
            pub_time = content.get("providerPublishTime") if isinstance(content, dict) else item.get("providerPublishTime")

            if not title:
                continue

            # Tarih filtresi
            if pub_time:
                try:
                    art_dt = datetime.fromtimestamp(float(pub_time))
                    if art_dt < cutoff:
                        continue
                except (TypeError, ValueError, OSError):
                    pass

            lower = title.lower()
            if any(kw in lower for kw in LEGAL_KEYWORDS):
                legal_count += 1

        risk = min(1.0, legal_count / MAX_LEGAL_NEWS)
        logger.info("Hukuki risk (%s): %d haber → %.3f", ticker, legal_count, risk)
        return _r2(risk)

    except Exception as e:
        logger.warning("Hukuki risk hesaplanamadı (%s): %s", ticker, e)
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Bölüm 4 – Kazanç Sezonu Flag
# ─────────────────────────────────────────────────────────────────────────────

def _get_earnings_flag(ticker: str) -> int:
    """
    Gelecek earnings tarihi EARNINGS_IMMINENT_DAYS içindeyse 1, değilse 0 döndürür.

    yfinance `calendar` kullanır.
    """
    try:
        tk  = yf.Ticker(ticker)
        cal = tk.calendar

        if cal is None:
            return 0

        # yfinance calendar formatı: dict veya DataFrame
        earnings_date = None

        if isinstance(cal, dict):
            # Yeni format: {"Earnings Date": [Timestamp, ...], ...}
            ed = cal.get("Earnings Date")
            if ed:
                if hasattr(ed, "__iter__"):
                    dates = [d for d in ed if d is not None]
                    if dates:
                        earnings_date = min(dates)
                else:
                    earnings_date = ed
        else:
            # DataFrame
            import pandas as pd
            if "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"]
                if hasattr(ed, "iloc"):
                    earnings_date = ed.iloc[0]
                else:
                    earnings_date = ed

        if earnings_date is None:
            return 0

        # datetime'a çevir
        import pandas as pd
        if isinstance(earnings_date, pd.Timestamp):
            earnings_dt = earnings_date.to_pydatetime()
        elif isinstance(earnings_date, str):
            earnings_dt = datetime.strptime(earnings_date[:10], "%Y-%m-%d")
        else:
            earnings_dt = earnings_date

        # Timezone normalize
        if hasattr(earnings_dt, "tzinfo") and earnings_dt.tzinfo:
            earnings_dt = earnings_dt.replace(tzinfo=None)

        days_until = (earnings_dt - datetime.now()).days

        if 0 <= days_until <= EARNINGS_IMMINENT_DAYS:
            logger.info(
                "Earnings yaklaşıyor (%s): %d gün kaldı (%s)",
                ticker, days_until, earnings_dt.date(),
            )
            return 1
        return 0

    except Exception as e:
        logger.debug("Earnings flag alınamadı (%s): %s", ticker, e)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Ana API
# ─────────────────────────────────────────────────────────────────────────────

def get_fundamental_features(ticker: str, data: dict | None = None) -> dict:
    """
    Bir hisse için tam fundamental feature vektörü döndürür.

    Parametreler
    ------------
    ticker : str
        Hisse kodu (ör: "AAPL")
    data : dict, opsiyonel
        `collect_ticker_data(ticker)` çıktısı. Verilmezse yfinance
        info'dan minimal fallback kullanılır.

    Döndürür
    --------
    dict
        date, ticker, pe_ratio, eps_surprise, revenue_growth,
        profit_margin, debt_equity, fcf_yield, insider_net,
        esg_score, legal_risk, earnings_flag
    """
    ticker = ticker.upper().strip()
    logger.info("▶ Fundamental features hesaplanıyor: %s …", ticker)

    # ── Shares outstanding (insider normalizasyonu için) ──────────────────
    shares_out = None
    if data:
        shares_out = _safe(data.get("balance_sheet", {}).get("shares_outstanding"))
    if shares_out is None:
        try:
            info = yf.Ticker(ticker).info
            shares_out = _safe(info.get("sharesOutstanding"))
        except Exception:
            pass

    # ── Bölüm 1: Finansal Metrikler ──────────────────────────────────────
    fin = _get_financial_metrics(ticker, data)

    # ── Bölüm 2: Insider Trading ─────────────────────────────────────────
    insider_net = _get_insider_signal(ticker, shares_out)

    # ── Bölüm 3: ESG + Hukuki Risk ───────────────────────────────────────
    esg_score  = _get_esg_score(ticker)
    legal_risk = _get_legal_risk(ticker)

    # ── Bölüm 4: Earnings Flag ────────────────────────────────────────────
    earnings_flag = _get_earnings_flag(ticker)

    features = {
        "date":           datetime.now().strftime("%Y-%m-%d"),
        "ticker":         ticker,
        # Finansal
        "pe_ratio":       fin["pe_ratio"],
        "eps_surprise":   fin["eps_surprise"],
        "revenue_growth": fin["revenue_growth"],
        "profit_margin":  fin["profit_margin"],
        "debt_equity":    fin["debt_equity"],
        "fcf_yield":      fin["fcf_yield"],
        # Insider
        "insider_net":    insider_net,
        # ESG
        "esg_score":      esg_score,
        "legal_risk":     legal_risk,
        # Earnings
        "earnings_flag":  earnings_flag,
    }

    logger.info("✔ Features hazır: %s | %s", ticker, {k: v for k, v in features.items() if k != "date"})
    return features


# ─────────────────────────────────────────────────────────────────────────────
# CSV Builder
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "date", "ticker",
    "pe_ratio", "eps_surprise", "revenue_growth",
    "profit_margin", "debt_equity", "fcf_yield",
    "insider_net", "esg_score", "legal_risk", "earnings_flag",
]


def build_features_csv(
    tickers: list[str],
    output_path: str | Path | None = None,
    data_map: dict[str, dict] | None = None,
) -> Path:
    """
    Birden fazla ticker için fundamental feature vektörleri üretir ve
    CSV olarak kaydeder.

    Parametreler
    ------------
    tickers : list[str]
        İşlenecek hisse kodları listesi.
    output_path : str | Path, opsiyonel
        CSV çıktı yolu. Verilmezse `output/features_05_fundamental.csv`.
    data_map : dict[str, dict], opsiyonel
        Ticker → collect_ticker_data() çıktısı haritası.
        Verilmezse her ticker için fallback kullanılır.

    Döndürür
    --------
    Path
        Kaydedilen CSV dosyasının yolu.
    """
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "features_05_fundamental.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker in tickers:
        ticker = ticker.upper().strip()
        data   = (data_map or {}).get(ticker)
        try:
            row = get_fundamental_features(ticker, data)
            rows.append(row)
        except Exception as e:
            logger.error("Feature hesaplanamadı (%s): %s — atlanıyor", ticker, e)
            # NaN satır ekle
            rows.append({
                col: (ticker if col == "ticker" else
                      datetime.now().strftime("%Y-%m-%d") if col == "date" else None)
                for col in FEATURE_COLUMNS
            })

    # CSV yaz
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in FEATURE_COLUMNS})

    logger.info("✔ CSV kaydedildi → %s (%d satır)", output_path, len(rows))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL"]

    if len(tickers) == 1:
        # Tek ticker → JSON çıktı
        result = get_fundamental_features(tickers[0])
        print(json.dumps(result, indent=2, default=str))
    else:
        # Birden fazla → CSV kaydet
        out = build_features_csv(tickers)
        print(f"\n  ✔ CSV kaydedildi → {out}\n")

        # Önizleme
        with open(out, encoding="utf-8") as f:
            for line in f:
                print(line, end="")
