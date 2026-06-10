"""
comps.py
────────
Aşama 3 – Comparable Company Analysis (Comps) Modülü

Hedef şirketin sektörel peer'larıyla karşılaştırmalı
göreli değerlemesini yapan modül.

Akış
----
  1. Hedef şirketin sektörüne göre otomatik peer grubu seç
  2. Her peer için çarpanları çek (yfinance .info)
  3. Peer medyan çarpanlarını hesapla (outlier filtreli)
  4. Büyüme primi ile düzeltilmiş çarpan → implied fiyat
  5. Comps fair value = implied fiyatların medyanı

Kullanım
--------
>>> from comps import comps_valuation
>>> result = comps_valuation("MSFT", target_data)
"""

import logging
import statistics
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

logger = logging.getLogger("comps")


# ─── Peer Universe ────────────────────────────────────────────────────────────
# Sektör → peer ticker listesi
# yfinance info["sector"] + info["industry"] → en yakın eşleşme

PEER_UNIVERSE: dict[str, list[str]] = {
    # Teknoloji
    "Technology/Software": [
        "MSFT", "GOOGL", "META", "ORCL", "CRM", "ADBE", "NOW", "INTU", "SNOW",
    ],
    "Technology/Semiconductor": [
        "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TSM", "AMAT", "LRCX", "MU",
    ],
    "Technology/Hardware": [
        "AAPL", "DELL", "HPQ", "CSCO", "IBM", "KEYS", "ZBRA",
    ],
    # Tüketici
    "Consumer/E-commerce": [
        "AMZN", "BABA", "JD", "MELI", "SE", "ETSY", "SHOP", "EBAY",
    ],
    "Consumer/Retail": [
        "WMT", "COST", "TGT", "HD", "LOW", "DG", "DLTR",
    ],
    "Consumer/Discretionary": [
        "TSLA", "NKE", "SBUX", "MCD", "YUM", "CMG", "DPZ",
    ],
    # Finans
    "Financials/Banks": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC",
    ],
    "Financials/Insurance": [
        "BRK-B", "AIG", "MET", "PRU", "ALL", "TRV", "CB",
    ],
    "Financials/Fintech": [
        "V", "MA", "PYPL", "SQ", "FIS", "FISV", "GPN",
    ],
    # Sağlık
    "Healthcare/Pharma": [
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "GILD", "AMGN",
    ],
    "Healthcare/Biotech": [
        "MRNA", "REGN", "VRTX", "BIIB", "ILMN", "SGEN", "ALNY",
    ],
    "Healthcare/Devices": [
        "MDT", "ABT", "SYK", "BSX", "ISRG", "EW", "ZBH",
    ],
    # Enerji
    "Energy/Oil & Gas": [
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX",
    ],
    "Energy/Renewable": [
        "NEE", "ENPH", "SEDG", "FSLR", "RUN", "PLUG",
    ],
    # Endüstriyel
    "Industrials/Aerospace": [
        "BA", "LMT", "RTX", "NOC", "GD", "HII", "TDG",
    ],
    "Industrials/General": [
        "GE", "HON", "MMM", "CAT", "DE", "EMR", "ETN",
    ],
    # İletişim
    "Communication/Telecom": [
        "T", "VZ", "TMUS", "CHTR", "CMCSA",
    ],
    "Communication/Media": [
        "DIS", "NFLX", "WBD", "PARA", "SPOT", "ROKU",
    ],
    # Gayrimenkul
    "Real Estate/REITs": [
        "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "PSA",
    ],
    # Temel Tüketim
    "Consumer Staples": [
        "PG", "KO", "PEP", "CL", "MDLZ", "PM", "MO",
    ],
    # Malzeme
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX",
    ],
    # Kamu Hizmetleri
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC",
    ],
}

# Sektör eşleme tablosu: yfinance sector/industry → PEER_UNIVERSE key
_SECTOR_MAP: dict[str, str] = {
    # Technology
    "technology":                  "Technology/Software",
    "software":                    "Technology/Software",
    "software—infrastructure":     "Technology/Software",
    "software—application":        "Technology/Software",
    "internet content & information": "Technology/Software",
    "semiconductors":              "Technology/Semiconductor",
    "semiconductor equipment & materials": "Technology/Semiconductor",
    "consumer electronics":        "Technology/Hardware",
    "computer hardware":           "Technology/Hardware",
    "communication equipment":     "Technology/Hardware",
    "electronic components":       "Technology/Hardware",
    "information technology services": "Technology/Software",
    # Consumer Cyclical
    "consumer cyclical":           "Consumer/Discretionary",
    "internet retail":             "Consumer/E-commerce",
    "specialty retail":            "Consumer/Retail",
    "discount stores":             "Consumer/Retail",
    "home improvement retail":     "Consumer/Retail",
    "restaurants":                 "Consumer/Discretionary",
    "auto manufacturers":          "Consumer/Discretionary",
    "apparel retail":              "Consumer/Retail",
    # Financial Services
    "financial services":          "Financials/Banks",
    "banks—diversified":           "Financials/Banks",
    "banks—regional":              "Financials/Banks",
    "capital markets":             "Financials/Banks",
    "insurance—diversified":       "Financials/Insurance",
    "insurance—life":              "Financials/Insurance",
    "insurance—property & casualty": "Financials/Insurance",
    "credit services":             "Financials/Fintech",
    "financial data & stock exchanges": "Financials/Fintech",
    # Healthcare
    "healthcare":                  "Healthcare/Pharma",
    "drug manufacturers—general":  "Healthcare/Pharma",
    "drug manufacturers—specialty & generic": "Healthcare/Pharma",
    "biotechnology":               "Healthcare/Biotech",
    "medical devices":             "Healthcare/Devices",
    "medical instruments & supplies": "Healthcare/Devices",
    "diagnostics & research":      "Healthcare/Biotech",
    # Energy
    "energy":                      "Energy/Oil & Gas",
    "oil & gas integrated":        "Energy/Oil & Gas",
    "oil & gas e&p":               "Energy/Oil & Gas",
    "oil & gas equipment & services": "Energy/Oil & Gas",
    "solar":                       "Energy/Renewable",
    "utilities—renewable":         "Energy/Renewable",
    # Industrials
    "industrials":                 "Industrials/General",
    "aerospace & defense":         "Industrials/Aerospace",
    "farm & heavy construction machinery": "Industrials/General",
    "specialty industrial machinery": "Industrials/General",
    "conglomerates":               "Industrials/General",
    # Communication Services
    "communication services":      "Communication/Media",
    "telecom services":            "Communication/Telecom",
    "entertainment":               "Communication/Media",
    "broadcasting":                "Communication/Media",
    # Real Estate
    "real estate":                 "Real Estate/REITs",
    "reit—specialty":              "Real Estate/REITs",
    "reit—industrial":             "Real Estate/REITs",
    # Consumer Defensive
    "consumer defensive":          "Consumer Staples",
    "household & personal products": "Consumer Staples",
    "beverages—non-alcoholic":     "Consumer Staples",
    "packaged foods":              "Consumer Staples",
    "tobacco":                     "Consumer Staples",
    # Basic Materials
    "basic materials":             "Materials",
    "specialty chemicals":         "Materials",
    "gold":                        "Materials",
    "copper":                      "Materials",
    # Utilities
    "utilities":                   "Utilities",
    "utilities—regulated electric": "Utilities",
    "utilities—diversified":       "Utilities",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe(value: Any, default=None) -> Any:
    """Return *value* if it is not NaN/None/inf, else *default*."""
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


def _median(values: list[float]) -> float | None:
    """Return median of non-empty list, or None."""
    if not values:
        return None
    return round(statistics.median(values), 2)


# ─── Sector matching ─────────────────────────────────────────────────────────

def identify_peer_group(
    sector: str | None,
    industry: str | None,
    ticker: str | None = None,
) -> tuple[str, list[str]]:
    """
    yfinance sector/industry → PEER_UNIVERSE key + peer listesi.

    Önce industry, sonra sector ile eşleşme aranır.
    Hedef ticker peer listesinden çıkarılır.

    Returns
    -------
    (group_name, peer_tickers)
    """
    # Önce industry ile eşleşme dene
    if industry:
        key = _SECTOR_MAP.get(industry.lower().strip())
        if key and key in PEER_UNIVERSE:
            peers = [t for t in PEER_UNIVERSE[key] if t != (ticker or "").upper()]
            return key, peers

    # Sonra sector ile dene
    if sector:
        key = _SECTOR_MAP.get(sector.lower().strip())
        if key and key in PEER_UNIVERSE:
            peers = [t for t in PEER_UNIVERSE[key] if t != (ticker or "").upper()]
            return key, peers

    # Fallback: tüm PEER_UNIVERSE key'lerinde sektör adı ara
    if sector:
        sector_lower = sector.lower()
        for universe_key, tickers in PEER_UNIVERSE.items():
            if sector_lower in universe_key.lower():
                peers = [t for t in tickers if t != (ticker or "").upper()]
                return universe_key, peers

    return "Unknown", []


# ─── Peer data fetching ──────────────────────────────────────────────────────

# Çekilecek çarpan alanları: (output_key, yfinance_info_key)
_MULTIPLE_FIELDS = [
    ("forward_pe",    "forwardPE"),
    ("trailing_pe",   "trailingPE"),
    ("ev_ebitda",     "enterpriseToEbitda"),
    ("ev_revenue",    "enterpriseToRevenue"),
    ("peg_ratio",     "pegRatio"),
    ("price_to_book", "priceToBook"),
]


def _fetch_peer_data(ticker: str) -> dict | None:
    """
    Tek bir peer ticker için çarpanları çek.

    Returns None if the ticker fetch fails entirely.
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        if not info or not info.get("currentPrice"):
            return None

        result = {"ticker": ticker}

        # Standart çarpanlar
        for out_key, info_key in _MULTIPLE_FIELDS:
            val = _safe(info.get(info_key))
            result[out_key] = _r2(val) if val else None

        # Price / FCF (hesaplamalı)
        mcap = _safe(info.get("marketCap"))
        fcf  = _safe(info.get("freeCashflow"))
        if mcap and fcf and float(fcf) > 0:
            result["price_fcf"] = _r2(float(mcap) / float(fcf))
        else:
            result["price_fcf"] = None

        # Büyüme verileri (growth premium için)
        result["revenue_growth"] = _r2(info.get("revenueGrowth"))
        result["earnings_growth"] = _r2(info.get("earningsGrowth"))

        # Ek bilgiler (raporlama için)
        result["name"]          = info.get("shortName", ticker)[:30]
        result["market_cap"]    = mcap
        result["current_price"] = _safe(info.get("currentPrice"))

        return result

    except Exception as e:
        logger.warning("Failed to fetch peer data for %s: %s", ticker, e)
        return None


def fetch_peers_data(
    peer_tickers: list[str],
    max_workers: int = 4,
) -> list[dict]:
    """
    Peer listesi için çarpan verilerini paralel olarak çek.

    Parameters
    ----------
    peer_tickers : Peer ticker listesi
    max_workers  : Paralel thread sayısı

    Returns
    -------
    Başarılı peer verilerinin listesi (None'lar filtrelenir)
    """
    results: list[dict] = []

    logger.info("Fetching data for %d peers …", len(peer_tickers))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_peer_data, t): t for t in peer_tickers}
        for fut in as_completed(futures):
            data = fut.result()
            if data is not None:
                results.append(data)

    logger.info("Successfully fetched %d / %d peers", len(results), len(peer_tickers))
    return results


# ─── Outlier filtering ────────────────────────────────────────────────────────

def _filter_values(
    peers_data: list[dict],
    key: str,
    max_val: float = 200.0,
    allow_negative: bool = False,
) -> list[float]:
    """
    Peer verilerinden belirli bir çarpanın geçerli değerlerini al.

    Filtreleme kuralları:
      - None/NaN çıkar
      - max_val üstü çıkar (P/E > 200 gibi aşırı değerler)
      - allow_negative=False ise negatifler çıkar
    """
    values: list[float] = []
    for peer in peers_data:
        v = _safe(peer.get(key))
        if v is None:
            continue
        v = float(v)
        if not allow_negative and v <= 0:
            continue
        if abs(v) > max_val:
            continue
        values.append(v)
    return values


# ─── Growth premium ──────────────────────────────────────────────────────────

def _calc_growth_premium(
    target_growth: float | None,
    peer_median_growth: float | None,
    cap: float = 0.50,
) -> float:
    """
    Hedef şirketin büyüme primini hesapla.

    premium = (target_growth – peer_median_growth) / peer_median_growth
    [-cap, +cap] aralığında sınırlandırılır.

    Yüksek büyüme → pozitif prim → çarpan yukarı ayarlanır.
    """
    if target_growth is None or peer_median_growth is None:
        return 0.0

    tg = float(target_growth)
    pg = float(peer_median_growth)

    if abs(pg) < 0.001:  # sıfıra bölmeyi önle
        return 0.0

    premium = (tg - pg) / abs(pg)
    return max(-cap, min(cap, premium))


# ─── Main valuation function ─────────────────────────────────────────────────

def comps_valuation(
    ticker: str,
    target_data: dict,
    peers_data: list[dict] | None = None,
    max_workers: int = 4,
) -> dict:
    """
    Comparable Company Analysis — ana değerleme fonksiyonu.

    Parameters
    ----------
    ticker      : Hedef şirketin ticker'ı
    target_data : data_collector.collect() çıktısı (hedef şirketin tüm verileri)
    peers_data  : Önceden çekilmiş peer verileri (None → otomatik çekilir)
    max_workers : Peer verisi çekerken paralel thread sayısı

    Returns
    -------
    Comps değerleme sonuç sözlüğü:
        peers_used, peer_medians, target_multiples, growth_premium,
        implied_prices, comps_fair_value
    """
    logger.info("▶ Running Comps valuation for %s …", ticker)

    result: dict[str, Any] = {
        "peers_used": [],
        "peer_group": "Unknown",
        "peer_count": 0,
        "peer_medians": {},
        "target_multiples": {},
        "growth_premium": 0.0,
        "implied_prices": {},
        "comps_fair_value": None,
    }

    # ── 1. Peer grubu belirle ─────────────────────────────────────────────────
    company_info = target_data.get("company_info", {})
    sector       = company_info.get("sector")
    industry     = company_info.get("industry")

    group_name, peer_tickers = identify_peer_group(sector, industry, ticker)
    result["peer_group"] = group_name

    if not peer_tickers:
        logger.warning("No peers found for %s (sector=%s, industry=%s)",
                        ticker, sector, industry)
        return result

    logger.info("Peer group: %s → %s", group_name, peer_tickers)

    # ── 2. Peer verilerini çek ────────────────────────────────────────────────
    if peers_data is None:
        peers_data = fetch_peers_data(peer_tickers, max_workers=max_workers)

    if not peers_data:
        logger.warning("No valid peer data fetched for %s", ticker)
        return result

    result["peers_used"] = [p["ticker"] for p in peers_data]
    result["peer_count"] = len(peers_data)

    # ── 3. Peer medyanlarını hesapla ──────────────────────────────────────────
    multiple_keys = [
        "forward_pe", "trailing_pe", "ev_ebitda", "ev_revenue",
        "price_fcf", "peg_ratio", "price_to_book",
    ]

    peer_medians: dict[str, float | None] = {}
    for key in multiple_keys:
        max_v = 200.0 if "pe" in key else 100.0
        if key == "price_fcf":
            max_v = 200.0
        if key == "peg_ratio":
            max_v = 10.0
        vals = _filter_values(peers_data, key, max_val=max_v)
        peer_medians[key] = _median(vals)

    # Peer büyüme medyanı
    growth_vals = _filter_values(peers_data, "revenue_growth",
                                  max_val=5.0, allow_negative=True)
    peer_medians["revenue_growth"] = _median(growth_vals) if growth_vals else None

    result["peer_medians"] = peer_medians

    # ── 4. Hedef şirketin kendi çarpanları ────────────────────────────────────
    vm  = target_data.get("valuation_multiples", {})
    bs  = target_data.get("balance_sheet", {})
    inc = target_data.get("income_statement", {})

    # Price / FCF hesapla
    mcap = _safe(company_info.get("market_cap"))
    ann  = inc.get("annual", {})
    latest_fcf = None
    if ann:
        latest_year = sorted(ann.keys(), reverse=True)[0]
        latest_fcf  = _safe(ann[latest_year].get("fcf"))
    target_price_fcf = None
    if mcap and latest_fcf and float(latest_fcf) > 0:
        target_price_fcf = _r2(float(mcap) / float(latest_fcf))

    target_multiples = {
        "forward_pe":    _r2(vm.get("pe_forward")),
        "trailing_pe":   _r2(vm.get("pe_trailing")),
        "ev_ebitda":     _r2(vm.get("ev_ebitda")),
        "ev_revenue":    _r2(vm.get("price_to_sales")),  # yaklaşık
        "price_fcf":     target_price_fcf,
        "peg_ratio":     _r2(vm.get("peg_ratio")),
        "price_to_book": _r2(vm.get("price_to_book")),
    }
    result["target_multiples"] = target_multiples

    # ── 5. Büyüme primi ──────────────────────────────────────────────────────
    # Hedef şirketin büyümesi: yfinance'tan veya growth_and_quality'den
    gq = target_data.get("growth_and_quality", {})
    target_rev_growth = _safe(gq.get("revenue_yoy_growth"))
    if target_rev_growth is not None:
        target_rev_growth = float(target_rev_growth) / 100.0  # % → oran

    growth_premium = _calc_growth_premium(
        target_growth      = target_rev_growth,
        peer_median_growth = peer_medians.get("revenue_growth"),
    )
    result["growth_premium"] = _r2(growth_premium)

    # ── 6. İma edilen fiyatlar ────────────────────────────────────────────────
    current_price = _safe(vm.get("current_price"))
    shares        = _safe(bs.get("shares_outstanding")) or 0
    ae            = target_data.get("analyst_expectations", {}) if "analyst_expectations" in target_data else {}
    net_debt      = float(_safe(bs.get("net_debt")) or 0)

    implied_prices: dict[str, float | None] = {}

    # --- P/E Comps: adjusted_fwd_pe × NTM EPS ---
    pm_fwd_pe = peer_medians.get("forward_pe")
    ntm_eps   = _safe(ae.get("ntm_eps_estimate")) or _safe(vm.get("pe_forward"))
    # If ntm_eps is actually a P/E, derive EPS from current price
    if ntm_eps and pm_fwd_pe and current_price:
        # Get NTM EPS from forward PE and current price
        fwd_pe_val = _safe(vm.get("pe_forward"))
        if fwd_pe_val and float(fwd_pe_val) > 0:
            derived_eps = float(current_price) / float(fwd_pe_val)
        else:
            derived_eps = None

        actual_ntm_eps = _safe(ae.get("ntm_eps_estimate"))
        eps_to_use = float(actual_ntm_eps) if actual_ntm_eps else derived_eps

        if eps_to_use and eps_to_use > 0:
            adj_pe = float(pm_fwd_pe) * (1 + growth_premium * 0.5)
            implied_prices["pe_comps"] = _r2(adj_pe * eps_to_use)

    # --- EV/EBITDA Comps ---
    pm_ev_ebitda = peer_medians.get("ev_ebitda")
    latest_ebitda = None
    if ann:
        latest_year = sorted(ann.keys(), reverse=True)[0]
        latest_ebitda = _safe(ann[latest_year].get("ebitda"))

    if pm_ev_ebitda and latest_ebitda and shares:
        adj_ev_ebitda = float(pm_ev_ebitda) * (1 + growth_premium * 0.5)
        implied_ev    = adj_ev_ebitda * float(latest_ebitda)
        eq_val        = implied_ev - net_debt
        if eq_val > 0:
            implied_prices["ev_ebitda_comps"] = _r2(eq_val / float(shares))

    # --- EV/Revenue Comps ---
    pm_ev_rev    = peer_medians.get("ev_revenue")
    latest_rev   = None
    if ann:
        latest_year = sorted(ann.keys(), reverse=True)[0]
        latest_rev  = _safe(ann[latest_year].get("revenue"))

    if pm_ev_rev and latest_rev and shares:
        adj_ev_rev = float(pm_ev_rev) * (1 + growth_premium * 0.5)
        implied_ev = adj_ev_rev * float(latest_rev)
        eq_val     = implied_ev - net_debt
        if eq_val > 0:
            implied_prices["ev_revenue_comps"] = _r2(eq_val / float(shares))

    # --- Price/FCF Comps ---
    pm_price_fcf = peer_medians.get("price_fcf")
    if pm_price_fcf and latest_fcf and shares and float(shares) > 0:
        adj_pfcf     = float(pm_price_fcf) * (1 + growth_premium * 0.5)
        fcf_per_share = float(latest_fcf) / float(shares)
        if fcf_per_share > 0:
            implied_prices["price_fcf_comps"] = _r2(adj_pfcf * fcf_per_share)

    result["implied_prices"] = implied_prices

    # ── 7. Comps fair value = medyan implied fiyat ────────────────────────────
    valid_prices = [v for v in implied_prices.values() if v is not None and v > 0]
    if valid_prices:
        result["comps_fair_value"] = _r2(statistics.median(valid_prices))

    # ── 8. Peer detay tablosu (raporlama için) ────────────────────────────────
    peer_details: list[dict] = []
    for p in peers_data:
        peer_details.append({
            "ticker":       p["ticker"],
            "name":         p.get("name", p["ticker"]),
            "forward_pe":   p.get("forward_pe"),
            "ev_ebitda":    p.get("ev_ebitda"),
            "ev_revenue":   p.get("ev_revenue"),
            "price_fcf":    p.get("price_fcf"),
            "peg_ratio":    p.get("peg_ratio"),
            "rev_growth":   p.get("revenue_growth"),
        })
    result["peer_details"] = peer_details

    logger.info(
        "✔ Comps complete: %d peers | fair value: $%s | growth premium: %.1f%%",
        len(peers_data),
        result["comps_fair_value"],
        (growth_premium or 0) * 100,
    )

    return result


# ─── Convenience ──────────────────────────────────────────────────────────────

def run_comps(ticker: str) -> dict:
    """
    Shortcut: veri topla + comps analizi çalıştır.

    >>> from comps import run_comps
    >>> result = run_comps("MSFT")
    """
    from data_collector import collect_ticker_data
    data = collect_ticker_data(ticker)
    return comps_valuation(ticker, data)


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    result = run_comps(ticker)

    class _Enc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):   return int(obj)
            if isinstance(obj, (np.floating,)):  return float(obj)
            if isinstance(obj, np.ndarray):      return obj.tolist()
            return super().default(obj)

    print(json.dumps(result, indent=2, cls=_Enc))
