"""
data_collector.py
─────────────────
Aşama 1 – Veri Toplama Pipeline

Bir ticker alıp o şirketle ilgili tüm temel verileri çeken modül.
Öncelik sırası:
  1. yfinance (ücretsiz, ana kaynak)
  2. macrotrends.net scraping (yedek)
  3. Mock data (son çare – loglanır)

yfinance v1.3+ API notu:
  .financials / .cashflow / .balance_sheet → rows = metric labels, columns = Timestamps
  Erişim: df.loc["Total Revenue", col]  (NOT df.get("Total Revenue", {}).get(col))
"""

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_collector")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe(value: Any, default=None) -> Any:
    """Return *value* if it is not NaN/None/inf, else *default*."""
    if value is None:
        return default
    try:
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return default
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _loc(df: pd.DataFrame, label: str, col) -> Any:
    """
    Safe .loc accessor: returns None if *label* is missing or value is NaN.
    Handles yfinance's row-indexed DataFrames (metrics as rows, dates as cols).
    """
    try:
        if label not in df.index:
            return None
        return _safe(df.loc[label, col])
    except (KeyError, TypeError):
        return None


def _pct(value: Any) -> Any:
    """Express a ratio as a percentage (×100), rounded to 2 dp."""
    v = _safe(value)
    if v is None:
        return None
    return round(float(v) * 100, 2)


def _round2(value: Any) -> Any:
    v = _safe(value)
    if v is None:
        return None
    return round(float(v), 2)


def _cagr(start: float, end: float, years: int) -> float | None:
    """Compound Annual Growth Rate (%)."""
    try:
        if start and end and years and float(start) > 0:
            return round(((float(end) / float(start)) ** (1 / years) - 1) * 100, 2)
    except (TypeError, ZeroDivisionError):
        pass
    return None


def _yoy(current: float, previous: float) -> float | None:
    try:
        if previous and current and float(previous) != 0:
            return round(((float(current) - float(previous)) / abs(float(previous))) * 100, 2)
    except (TypeError, ZeroDivisionError):
        pass
    return None


def _fmt_b(value: Any) -> str | None:
    """Format large number as $XB or $XM string (for display)."""
    v = _safe(value)
    if v is None:
        return None
    v = float(v)
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"


# ─── Main collector class ─────────────────────────────────────────────────────

class FinancialDataCollector:
    """
    Collect, organise, and validate fundamental financial data for *ticker*.

    Usage
    -----
    >>> collector = FinancialDataCollector("MSFT")
    >>> data = collector.collect()
    """

    def __init__(self, ticker: str):
        self.ticker_symbol = ticker.upper().strip()
        self._tk = yf.Ticker(self.ticker_symbol)
        self._mock_fields: list[str] = []

        # Cache raw DataFrames (fetched once)
        self._ann:    pd.DataFrame | None = None
        self._ann_cf: pd.DataFrame | None = None
        self._ann_bs: pd.DataFrame | None = None
        self._qtr:    pd.DataFrame | None = None
        self._qtr_cf: pd.DataFrame | None = None
        self._info:   dict | None = None

    # ── Lazy-loaded raw data ───────────────────────────────────────────────────

    def _get_ann(self) -> pd.DataFrame:
        if self._ann is None:
            self._ann = self._tk.financials
        return self._ann

    def _get_ann_cf(self) -> pd.DataFrame:
        if self._ann_cf is None:
            self._ann_cf = self._tk.cashflow
        return self._ann_cf

    def _get_ann_bs(self) -> pd.DataFrame:
        if self._ann_bs is None:
            self._ann_bs = self._tk.balance_sheet
        return self._ann_bs

    def _get_qtr(self) -> pd.DataFrame:
        if self._qtr is None:
            self._qtr = self._tk.quarterly_financials
        return self._qtr

    def _get_qtr_cf(self) -> pd.DataFrame:
        if self._qtr_cf is None:
            self._qtr_cf = self._tk.quarterly_cashflow
        return self._qtr_cf

    def _get_info(self) -> dict:
        if self._info is None:
            self._info = self._tk.info
        return self._info

    # ── Public API ────────────────────────────────────────────────────────────

    def collect(self) -> dict:
        """Run the full pipeline and return a unified data dictionary."""
        logger.info("▶ Collecting data for %s …", self.ticker_symbol)

        data = {
            "ticker":               self.ticker_symbol,
            "company_info":         self._get_company_info(),
            "income_statement":     self._get_income_statement(),
            "balance_sheet":        self._get_balance_sheet(),
            "valuation_multiples":  self._get_valuation_multiples(),
            "growth_and_quality":   self._get_growth_and_quality(),
            "analyst_expectations": self._get_analyst_expectations(),
        }

        if self._mock_fields:
            logger.warning("⚠ Mock/missing data for: %s", ", ".join(self._mock_fields))
        data["mock_fields"] = self._mock_fields

        logger.info("✔ Data collection complete for %s", self.ticker_symbol)
        return data

    # ── Section 0 – Company Info ──────────────────────────────────────────────

    def _get_company_info(self) -> dict:
        try:
            raw = self._get_info()
            return {
                "name":        raw.get("longName") or raw.get("shortName", self.ticker_symbol),
                "sector":      raw.get("sector"),
                "industry":    raw.get("industry"),
                "country":     raw.get("country"),
                "currency":    raw.get("currency", "USD"),
                "exchange":    raw.get("exchange"),
                "market_cap":  _safe(raw.get("marketCap")),
                "employees":   _safe(raw.get("fullTimeEmployees")),
                "description": (raw.get("longBusinessSummary") or "")[:500],
                "website":     raw.get("website"),
            }
        except Exception as e:
            logger.error("company_info failed: %s", e)
            self._mock_fields.append("company_info")
            return {"name": self.ticker_symbol, "currency": "USD"}

    # ── Section 1 – Income Statement ──────────────────────────────────────────

    def _income_row(self, df: pd.DataFrame, cf: pd.DataFrame, col) -> dict:
        """Extract one period's income / cash-flow metrics from yfinance DataFrames."""
        rev   = _loc(df,  "Total Revenue",   col)
        gross = _loc(df,  "Gross Profit",    col)
        ni    = _loc(df,  "Net Income",      col)
        ebit  = _loc(df,  "EBIT",            col)
        ebitda = _loc(df, "EBITDA",          col)
        if ebitda is None:
            da = (
                _loc(df,  "Reconciled Depreciation", col)
                or _loc(cf, "Depreciation And Amortization", col)
                or _loc(cf, "Depreciation Amortization Depletion", col)
                or 0
            )
            ebitda = (ebit + da) if ebit is not None else None

        eps = _loc(df, "Diluted EPS", col)
        if eps is None and ni:
            shares = _loc(df, "Diluted Average Shares", col)
            eps = ni / shares if shares else None

        op_cf = _loc(cf, "Operating Cash Flow", col)
        fcf   = _loc(cf, "Free Cash Flow",       col)
        capex = _loc(cf, "Capital Expenditure",   col)
        if capex is not None:
            capex = abs(capex)
        if fcf is None and op_cf is not None and capex is not None:
            fcf = op_cf - capex

        return {
            "revenue":       rev,
            "gross_profit":  gross,
            "gross_margin":  _pct(gross / rev) if (gross and rev) else None,
            "ebitda":        ebitda,
            "net_income":    ni,
            "eps":           _round2(eps),
            "operating_cf":  op_cf,
            "capex":         capex,
            "fcf":           fcf,
        }

    def _get_income_statement(self) -> dict:
        result = {"annual": {}, "quarterly": {}}

        # --- Annual (last 5 fiscal years) ---
        try:
            ann    = self._get_ann()
            ann_cf = self._get_ann_cf()
            for col in ann.columns[:5]:
                yr = str(col.year)
                result["annual"][yr] = self._income_row(ann, ann_cf, col)
        except Exception as e:
            logger.error("Annual income statement failed: %s", e)
            self._mock_fields.append("income_statement.annual")

        # --- Quarterly (last 4 quarters) ---
        try:
            qtr    = self._get_qtr()
            qtr_cf = self._get_qtr_cf()
            for col in qtr.columns[:4]:
                q_label = f"{col.year}Q{col.quarter}"
                result["quarterly"][q_label] = self._income_row(qtr, qtr_cf, col)
        except Exception as e:
            logger.error("Quarterly income statement failed: %s", e)
            self._mock_fields.append("income_statement.quarterly")

        return result

    # ── Section 2 – Balance Sheet ─────────────────────────────────────────────

    def _get_balance_sheet(self) -> dict:
        try:
            bs   = self._get_ann_bs()
            info = self._get_info()
            col  = bs.columns[0]   # most recent annual

            cash         = _loc(bs, "Cash And Cash Equivalents", col)
            total_cash   = _loc(bs, "Cash Cash Equivalents And Short Term Investments", col) or cash
            total_debt   = _loc(bs, "Total Debt", col)
            net_debt     = _loc(bs, "Net Debt", col)
            if net_debt is None and total_debt is not None and total_cash is not None:
                net_debt = total_debt - total_cash

            total_assets = _loc(bs, "Total Assets", col)
            equity       = _loc(bs, "Stockholders Equity", col) or _loc(bs, "Common Stock Equity", col)
            shares_out   = (
                _loc(bs, "Share Issued", col)
                or _loc(bs, "Ordinary Shares Number", col)
                or _safe(info.get("sharesOutstanding"))
            )
            bvps = (equity / shares_out) if (equity and shares_out) else None

            return {
                "as_of_date":             str(col.date()),
                "total_debt":             total_debt,
                "net_debt":               net_debt,
                "cash_and_equiv":         total_cash,
                "total_assets":           total_assets,
                "book_value":             equity,
                "book_value_per_share":   _round2(bvps),
                "shares_outstanding":     shares_out,
            }
        except Exception as e:
            logger.error("Balance sheet failed: %s", e)
            self._mock_fields.append("balance_sheet")
            return {}

    # ── Section 3 – Valuation Multiples ──────────────────────────────────────

    def _get_valuation_multiples(self) -> dict:
        try:
            info = self._get_info()
            return {
                "pe_trailing":      _round2(info.get("trailingPE")),
                "pe_forward":       _round2(info.get("forwardPE")),
                "ev_ebitda":        _round2(info.get("enterpriseToEbitda")),
                "price_to_sales":   _round2(info.get("priceToSalesTrailing12Months")),
                "price_to_book":    _round2(info.get("priceToBook")),
                "peg_ratio":        _round2(info.get("pegRatio")),
                "enterprise_value": _safe(info.get("enterpriseValue")),
                "current_price":    _safe(info.get("currentPrice") or info.get("regularMarketPrice")),
                "52w_high":         _safe(info.get("fiftyTwoWeekHigh")),
                "52w_low":          _safe(info.get("fiftyTwoWeekLow")),
                "beta":             _round2(info.get("beta")),
                "dividend_yield":   _round2(info.get("dividendYield")),   # already in % form from yfinance
            }
        except Exception as e:
            logger.error("Valuation multiples failed: %s", e)
            self._mock_fields.append("valuation_multiples")
            return {}

    # ── Section 4 – Growth & Quality ──────────────────────────────────────────

    def _get_growth_and_quality(self) -> dict:
        try:
            ann    = self._get_ann()
            ann_cf = self._get_ann_cf()
            bs     = self._get_ann_bs()
            info   = self._get_info()
            cols   = ann.columns.tolist()

            def rev(i):
                return _loc(ann, "Total Revenue", cols[i]) if i < len(cols) else None

            def eps(i):
                e = _loc(ann, "Diluted EPS", cols[i]) if i < len(cols) else None
                if e:
                    return e
                ni = _loc(ann, "Net Income", cols[i]) if i < len(cols) else None
                sh = _loc(ann, "Diluted Average Shares", cols[i]) if i < len(cols) else None
                return (ni / sh) if (ni and sh) else None

            # Revenue growth
            rev_yoy  = _yoy(rev(0), rev(1))
            rev_cagr = _cagr(rev(3), rev(0), 3)

            # EPS growth
            eps_yoy  = _yoy(eps(0), eps(1))
            eps_cagr = _cagr(eps(3), eps(0), 3)

            # ROE
            roe = None
            if info.get("returnOnEquity"):
                roe = _round2(info["returnOnEquity"] * 100)

            # ROIC = NOPAT / Invested Capital
            roic = None
            try:
                col0    = cols[0]
                bs_col  = bs.columns[0]
                ebit    = _loc(ann, "EBIT", col0) or 0
                tax_r   = _loc(ann, "Tax Rate For Calcs", col0) or 0.21
                nopat   = ebit * (1 - tax_r)
                equity  = _loc(bs, "Stockholders Equity", bs_col) or _loc(bs, "Common Stock Equity", bs_col) or 0
                td      = _loc(bs, "Total Debt", bs_col) or 0
                cash    = _loc(bs, "Cash And Cash Equivalents", bs_col) or 0
                inv_cap = equity + td - cash
                roic    = round(nopat / inv_cap * 100, 2) if inv_cap else None
            except Exception:
                pass

            # Debt / EBITDA
            debt_ebitda = None
            try:
                col0   = cols[0]
                bs_col = bs.columns[0]
                td     = _loc(bs, "Total Debt", bs_col)
                ebitda = _loc(ann, "EBITDA", col0) or _loc(ann, "Normalized EBITDA", col0)
                debt_ebitda = round(float(td) / float(ebitda), 2) if (td and ebitda) else None
            except Exception:
                pass

            return {
                "revenue_yoy_growth": rev_yoy,
                "revenue_3y_cagr":    rev_cagr,
                "eps_yoy_growth":     eps_yoy,
                "eps_3y_cagr":        eps_cagr,
                "roe":                roe,
                "roic":               roic,
                "debt_to_ebitda":     debt_ebitda,
            }
        except Exception as e:
            logger.error("Growth & quality failed: %s", e)
            self._mock_fields.append("growth_and_quality")
            return {}

    # ── Section 5 – Analyst Expectations ─────────────────────────────────────

    def _get_analyst_expectations(self) -> dict:
        result = {}
        try:
            info = self._get_info()

            result["price_target_mean"]    = _round2(info.get("targetMeanPrice"))
            result["price_target_low"]     = _round2(info.get("targetLowPrice"))
            result["price_target_high"]    = _round2(info.get("targetHighPrice"))
            result["price_target_median"]  = _round2(info.get("targetMedianPrice"))
            result["recommendation"]       = info.get("recommendationKey")
            result["num_analyst_opinions"] = info.get("numberOfAnalystOpinions")

            # Buy / Hold / Sell distribution (from recommendations table)
            try:
                rec_df = self._tk.recommendations
                if rec_df is not None and not rec_df.empty:
                    latest = rec_df.iloc[0]
                    result["strong_buy"]  = int(_safe(latest.get("strongBuy"),  0))
                    result["buy"]         = int(_safe(latest.get("buy"),         0))
                    result["hold"]        = int(_safe(latest.get("hold"),        0))
                    result["sell"]        = int(_safe(latest.get("sell"),        0))
                    result["strong_sell"] = int(_safe(latest.get("strongSell"),  0))
            except Exception:
                pass

            # NTM EPS estimate
            try:
                eet = self._tk.earnings_estimate
                if eet is not None and not eet.empty and "+1y" in eet.index:
                    result["ntm_eps_estimate"] = _round2(_safe(eet.loc["+1y", "avg"]))
            except Exception:
                pass
            if not result.get("ntm_eps_estimate"):
                result["ntm_eps_estimate"] = _round2(info.get("forwardEps"))

            # NTM Revenue estimate
            try:
                ret = self._tk.revenue_estimate
                if ret is not None and not ret.empty and "+1y" in ret.index:
                    result["ntm_revenue_estimate"] = _safe(ret.loc["+1y", "avg"])
            except Exception:
                pass

            # Upside to consensus target
            price = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))
            if price and result.get("price_target_mean"):
                result["upside_to_target"] = _round2(
                    (result["price_target_mean"] / price - 1) * 100
                )

        except Exception as e:
            logger.error("Analyst expectations failed: %s", e)
            self._mock_fields.append("analyst_expectations")
        return result


# ─── Convenience function ─────────────────────────────────────────────────────

def collect_ticker_data(ticker: str) -> dict:
    """
    Shortcut: instantiate collector and return raw data dict.

    Example
    -------
    >>> from data_collector import collect_ticker_data
    >>> data = collect_ticker_data("AAPL")
    """
    return FinancialDataCollector(ticker).collect()


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    data   = collect_ticker_data(ticker)

    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return super().default(obj)

    print(json.dumps(data, indent=2, cls=_Encoder))
