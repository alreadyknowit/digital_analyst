"""
analyzer.py
───────────
Aşama 2 – Değerleme Modülü

data_collector.py dosyasından gelen veriyi kullanarak:
  1. DCF (Discounted Cash Flow) analizi
  2. Çarpan (Multiples) analizi
  3. Ağırlıklı hedef fiyat hesabı

üretir.

Kullanım
--------
>>> from data_collector import collect_ticker_data
>>> from analyzer import Analyzer
>>> data = collect_ticker_data("MSFT")
>>> result = Analyzer(data).analyze()
"""

import logging
import warnings
from typing import Any

import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

logger = logging.getLogger("analyzer")

# ─── Sabitler ─────────────────────────────────────────────────────────────────

EQUITY_RISK_PREMIUM = 0.055   # Damodaran default (%)
DEFAULT_TAX_RATE    = 0.21    # ABD kurumlar vergisi
RISK_FREE_TICKER    = "^TNX"  # 10 yıllık ABD Hazine faizi (yfinance)

# Senaryo büyüme parametreleri (otomatik ayarlanabilir)
DEFAULT_SCENARIOS: dict[str, dict] = {
    "bull": {
        "stage1_growth":    0.18,
        "stage2_growth":    0.10,
        "terminal_growth":  0.03,
    },
    "base": {
        "stage1_growth":    0.12,
        "stage2_growth":    0.07,
        "terminal_growth":  0.025,
    },
    "bear": {
        "stage1_growth":    0.06,
        "stage2_growth":    0.04,
        "terminal_growth":  0.02,
    },
}

# Ağırlıklar (base ağırlıklı)
SCENARIO_WEIGHTS = {"bull": 0.25, "base": 0.50, "bear": 0.25}

# Hedef fiyat harmanlama ağırlıkları (DCF / Çarpan / Comps)
DCF_WEIGHT       = 0.50
MULTIPLES_WEIGHT = 0.25
COMPS_WEIGHT     = 0.25


# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def _safe(value: Any, default=None) -> Any:
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
    return round(float(v), 2) if v is not None else None


def _get_risk_free_rate() -> float:
    """Güncel 10 yıllık ABD Hazine faizini yfinance'tan çeker (% → oran)."""
    try:
        tkr  = yf.Ticker(RISK_FREE_TICKER)
        hist = tkr.history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1]) / 100.0
            logger.info("Risk-free rate (^TNX): %.4f", rate)
            return rate
    except Exception as e:
        logger.warning("Risk-free rate fetch failed: %s — using 4.5%%", e)
    return 0.045   # fallback


# ─── Ana sınıf ────────────────────────────────────────────────────────────────

class Analyzer:
    """
    data_collector.py çıktısını alıp tam değerleme analizi üretir.

    Parametreler
    ------------
    data      : collect_ticker_data() sonucu
    scenarios : Özel senaryo tanımı (None → DEFAULT_SCENARIOS kullanılır)
    """

    def __init__(self, data: dict, scenarios: dict | None = None):
        self.data      = data
        self.ticker    = data.get("ticker", "N/A")
        self.scenarios = scenarios or DEFAULT_SCENARIOS
        self._rf_rate: float | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        """Tam analizi çalıştır ve sonuç sözlüğü döndür."""
        logger.info("▶ Analyzing %s …", self.ticker)

        # 1. WACC
        wacc_result = self._calc_wacc()

        # 2. Senaryo büyüme oranlarını veriye göre otomatik ayarla
        scenarios = self._tune_scenarios()

        # 3. Baz FCF
        base_fcf = self._get_base_fcf()

        # 4. Her senaryo için DCF
        dcf_results: dict[str, dict] = {}
        for name, params in scenarios.items():
            dcf_results[name] = self._run_dcf(
                base_fcf  = base_fcf,
                wacc      = wacc_result["wacc"],
                growth    = params,
            )

        # 5. Ağırlıklı DCF değeri
        weighted_dcf = self._weighted_dcf_price(dcf_results)

        # 6. Çarpan analizi
        multiples_result = self._run_multiples_analysis()

        # 7. Comps (Comparable Company Analysis)
        comps_result = self._run_comps()

        # 8. Nihai hedef fiyat (3-yönlü harmanlama)
        target_price = self._blend_target(weighted_dcf, multiples_result, comps_result)

        # 9. Mevcut fiyat & upside
        current_price = self._current_price()
        upside = _r2((target_price / current_price - 1) * 100) if (target_price and current_price) else None

        # 10. Duyarlılık tablosu (WACC × Terminal Growth)
        base_scen   = scenarios.get("base", {})
        sensitivity = self._sensitivity_table(
            base_fcf   = base_fcf,
            base_wacc  = wacc_result["wacc"],
            base_g1    = base_scen.get("stage1_growth", 0.12),
            base_g2    = base_scen.get("stage2_growth", 0.07),
            base_g_t   = base_scen.get("terminal_growth", 0.025),
        )

        result = {
            "ticker":           self.ticker,
            "current_price":    current_price,
            "target_price":     _r2(target_price),
            "upside_pct":       upside,
            "recommendation":   self._recommendation(upside),
            "wacc":             wacc_result,
            "base_fcf":         _r2(base_fcf),
            "scenarios":        scenarios,
            "dcf":              dcf_results,
            "weighted_dcf_price": _r2(weighted_dcf),
            "multiples":        multiples_result,
            "comps":            comps_result,
            "dcf_weight":       DCF_WEIGHT,
            "multiples_weight": MULTIPLES_WEIGHT,
            "comps_weight":     COMPS_WEIGHT,
            "sensitivity":      sensitivity,
        }

        logger.info(
            "✔ %s | Target: $%.2f | Upside: %s%% | Rec: %s",
            self.ticker,
            target_price or 0,
            upside,
            result["recommendation"],
        )
        return result

    # ── 1. WACC ───────────────────────────────────────────────────────────────

    def _calc_wacc(self) -> dict:
        """
        WACC = (E/V) * Re + (D/V) * Rd * (1 – Tax Rate)

        Re  : CAPM → Rf + β × ERP
        Rd  : interest expense / total debt
        E/V : equity / (equity + debt)  [market-cap ağırlıklı]
        """
        bs        = self.data.get("balance_sheet", {})
        vm        = self.data.get("valuation_multiples", {})
        g_q       = self.data.get("growth_and_quality", {})
        inc       = self.data.get("income_statement", {})
        ann_years = list(inc.get("annual", {}).values())

        # Risk-free rate
        rf = self._get_rf()

        # Beta
        beta = _safe(vm.get("beta")) or 1.0

        # Cost of equity (CAPM)
        re = rf + beta * EQUITY_RISK_PREMIUM

        # Cost of debt
        total_debt = _safe(bs.get("total_debt")) or 0
        interest_expense = None
        for yr_data in ann_years:
            pass   # placeholder — extracted below

        # Try to get interest expense from income statement raw data
        try:
            tkr      = yf.Ticker(self.ticker)
            fin      = tkr.financials
            if not fin.empty:
                col0 = fin.columns[0]
                ie_val = None
                for label in ["Interest Expense", "Interest Expense Non Operating"]:
                    if label in fin.index:
                        ie_val = _safe(fin.loc[label, col0])
                        if ie_val:
                            break
                if ie_val:
                    interest_expense = abs(float(ie_val))
        except Exception:
            pass

        rd = 0.05   # fallback
        if interest_expense and total_debt and total_debt > 0:
            rd = interest_expense / total_debt

        # Tax rate
        tax_rate = DEFAULT_TAX_RATE
        try:
            tkr = yf.Ticker(self.ticker)
            fin = tkr.financials
            if not fin.empty:
                col0 = fin.columns[0]
                if "Tax Rate For Calcs" in fin.index:
                    tr = _safe(fin.loc["Tax Rate For Calcs", col0])
                    if tr and 0 < float(tr) < 1:
                        tax_rate = float(tr)
        except Exception:
            pass

        # Capital structure weights
        market_cap = _safe(self.data.get("company_info", {}).get("market_cap")) or 0
        if market_cap == 0:
            current_p = _safe(vm.get("current_price")) or 0
            shares    = _safe(bs.get("shares_outstanding")) or 0
            market_cap = current_p * shares

        total_capital = market_cap + total_debt
        e_weight = market_cap   / total_capital if total_capital else 0.8
        d_weight = total_debt   / total_capital if total_capital else 0.2

        wacc = e_weight * re + d_weight * rd * (1 - tax_rate)

        logger.info(
            "WACC: %.2f%% | Re: %.2f%% | Rd: %.2f%% | β: %.2f | Rf: %.2f%%",
            wacc * 100, re * 100, rd * 100, beta, rf * 100,
        )

        return {
            "wacc":       _r2(wacc * 100),    # %
            "re":         _r2(re  * 100),
            "rd":         _r2(rd  * 100),
            "rf":         _r2(rf  * 100),
            "beta":       _r2(beta),
            "tax_rate":   _r2(tax_rate * 100),
            "e_weight":   _r2(e_weight * 100),
            "d_weight":   _r2(d_weight * 100),
            "_wacc_dec":  wacc,               # internal decimal form
        }

    def _get_rf(self) -> float:
        if self._rf_rate is None:
            self._rf_rate = _get_risk_free_rate()
        return self._rf_rate

    # ── 2. Büyüme senaryosu otomatik ayarlama ─────────────────────────────────

    def _tune_scenarios(self) -> dict:
        """
        Şirketin geçmiş büyümesine ve sektörüne göre senaryoları ayarla.
        Yüksek büyüme şirketi (revenue CAGR > %15) daha yüksek stage1_growth alır.
        """
        import copy
        scenarios = copy.deepcopy(self.scenarios)

        gq       = self.data.get("growth_and_quality", {})
        rev_cagr = _safe(gq.get("revenue_3y_cagr"))   # % cinsinden

        # Şirket çok hızlı büyüyorsa bull/base senaryolarını yukarı çek
        if rev_cagr and float(rev_cagr) > 15:
            boost = min(float(rev_cagr) / 100 * 0.5, 0.08)   # max +8pp
            for name in ("bull", "base"):
                scenarios[name]["stage1_growth"] = round(
                    scenarios[name]["stage1_growth"] + boost, 4
                )
            logger.info("High-growth company detected (CAGR %.1f%%) — boosting stage1 by +%.1f%%",
                        rev_cagr, boost * 100)

        # Düşük/negatif büyüme → bear senaryoyu aşağı çek
        elif rev_cagr and float(rev_cagr) < 5:
            drag = min(abs(float(rev_cagr)) / 100 * 0.3, 0.03)
            scenarios["bear"]["stage1_growth"]   = max(
                scenarios["bear"]["stage1_growth"] - drag, 0.0
            )
            scenarios["bear"]["terminal_growth"] = max(
                scenarios["bear"]["terminal_growth"] - 0.005, 0.005
            )

        return scenarios

    # ── 3. Baz FCF ────────────────────────────────────────────────────────────

    def _get_base_fcf(self) -> float | None:
        """
        En güncel yıllık FCF'yi döndür.
        Yoksa Operating CF – CapEx ile hesapla.
        Hâlâ yoksa Net Income'ı yaklaşık FCF olarak kullan.
        """
        ann = self.data.get("income_statement", {}).get("annual", {})
        if not ann:
            return None

        latest_year = sorted(ann.keys(), reverse=True)[0]
        d = ann[latest_year]

        fcf = _safe(d.get("fcf"))
        if fcf:
            logger.info("Base FCF (%s): $%.2fB", latest_year, float(fcf) / 1e9)
            return float(fcf)

        op_cf = _safe(d.get("operating_cf"))
        capex = _safe(d.get("capex"))
        if op_cf and capex:
            fcf = float(op_cf) - float(capex)
            logger.info("Base FCF computed (OCF-CapEx): $%.2fB", fcf / 1e9)
            return fcf

        ni = _safe(d.get("net_income"))
        if ni:
            logger.warning("Using Net Income as FCF proxy for %s", self.ticker)
            return float(ni)

        return None

    # ── 4. DCF projeksiyonu ───────────────────────────────────────────────────

    def _run_dcf(
        self,
        base_fcf: float | None,
        wacc:     float,        # % cinsinden (örn: 9.5)
        growth:   dict,
    ) -> dict:
        """
        2-aşamalı DCF + Gordon Growth Terminal Value.

        Stage 1 : yıl 1–5   (yüksek büyüme)
        Stage 2 : yıl 6–10  (büyüme yavaşlaması)
        Terminal: Gordon Growth Model  TV = FCF₁₀ × (1+g) / (WACC – g)
        """
        if not base_fcf:
            return {"error": "base_fcf missing"}

        wacc_dec = wacc / 100.0
        g1 = growth["stage1_growth"]
        g2 = growth["stage2_growth"]
        g_t = growth["terminal_growth"]

        fcf_projections: list[float] = []
        pv_fcfs:         list[float] = []
        fcf = float(base_fcf)

        for yr in range(1, 11):
            g = g1 if yr <= 5 else g2
            fcf = fcf * (1 + g)
            pv  = fcf / ((1 + wacc_dec) ** yr)
            fcf_projections.append(_r2(fcf))
            pv_fcfs.append(_r2(pv))

        # Terminal Value
        tv       = fcf * (1 + g_t) / (wacc_dec - g_t)
        pv_tv    = tv / ((1 + wacc_dec) ** 10)

        # Enterprise Value
        pv_fcf_total = sum(pv_fcfs)
        ev           = pv_fcf_total + pv_tv

        # Net Debt düzeltmesi → Equity Value
        bs             = self.data.get("balance_sheet", {})
        net_debt       = _safe(bs.get("net_debt")) or 0
        equity_value   = ev - float(net_debt)

        shares         = _safe(bs.get("shares_outstanding")) or 0
        price_per_share = equity_value / float(shares) if shares else None

        return {
            "fcf_projections":   fcf_projections,
            "pv_fcfs":           pv_fcfs,
            "pv_fcf_total":      _r2(pv_fcf_total),
            "terminal_value":    _r2(tv),
            "pv_terminal_value": _r2(pv_tv),
            "enterprise_value":  _r2(ev),
            "net_debt":          _r2(net_debt),
            "equity_value":      _r2(equity_value),
            "shares_outstanding": shares,
            "intrinsic_price":   _r2(price_per_share),
            "tv_as_pct_ev":      _r2(pv_tv / ev * 100) if ev else None,
        }

    def _weighted_dcf_price(self, dcf_results: dict) -> float | None:
        """3 senaryonun ağırlıklı ortalama intrinsic fiyatını hesapla."""
        total_w  = 0.0
        total_wv = 0.0
        for name, weight in SCENARIO_WEIGHTS.items():
            price = _safe(dcf_results.get(name, {}).get("intrinsic_price"))
            if price is not None:
                total_wv += float(price) * weight
                total_w  += weight
        return (total_wv / total_w) if total_w > 0 else None

    def _sensitivity_table(
        self,
        base_fcf:  float | None,
        base_wacc: float,        # % (örn: 10.5)
        base_g1:   float,        # oran (örn: 0.12)
        base_g2:   float,        # oran (örn: 0.07)
        base_g_t:  float,        # oran (örn: 0.025)
        wacc_steps: int = 2,     # ±2pp
        gt_steps:   int = 2,     # ±1pp (0.5pp adım)
    ) -> dict:
        """
        WACC × Terminal Growth Rate duyarlılık matrisi.

        WACC ekseni  : base ±wacc_steps pp (1pp adım)   → 5 sütun
        g_t  ekseni  : base ±gt_steps×0.5pp (0.5pp adım) → 5 satır

        Döndürür:
            {
              "wacc_axis":   [8.5, 9.5, 10.5, 11.5, 12.5],   # %
              "gt_axis":     [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],  # %
              "matrix":      [[price, ...], ...],              # satır=g_t, sütun=wacc
              "base_wacc":   10.5,
              "base_gt":     2.5,
            }
        """
        if not base_fcf:
            return {}

        wacc_axis = [
            round(base_wacc + (i - wacc_steps) * 1.0, 2)
            for i in range(wacc_steps * 2 + 1)
        ]   # [base-2, base-1, base, base+1, base+2]  %

        gt_axis = [
            round((base_g_t * 100) + (j - gt_steps) * 0.5, 2)
            for j in range(gt_steps * 2 + 1)
        ]   # %

        bs     = self.data.get("balance_sheet", {})
        net_debt = float(_safe(bs.get("net_debt")) or 0)
        shares   = float(_safe(bs.get("shares_outstanding")) or 0)

        matrix: list[list[float | None]] = []
        for gt_pct in gt_axis:
            row: list[float | None] = []
            for wacc_pct in wacc_axis:
                wacc_dec = wacc_pct / 100.0
                g_t_dec  = gt_pct  / 100.0

                if wacc_dec <= g_t_dec:
                    row.append(None)   # geçersiz (Gordon Growth undefined)
                    continue

                # Aynı Stage1/Stage2 büyüme, sadece WACC ve g_t değişiyor
                fcf = float(base_fcf)
                pv_sum = 0.0
                for yr in range(1, 11):
                    g   = base_g1 if yr <= 5 else base_g2
                    fcf = fcf * (1 + g)
                    pv_sum += fcf / ((1 + wacc_dec) ** yr)

                tv    = fcf * (1 + g_t_dec) / (wacc_dec - g_t_dec)
                pv_tv = tv / ((1 + wacc_dec) ** 10)
                ev    = pv_sum + pv_tv
                eq_val = ev - net_debt
                price  = eq_val / shares if shares else None
                row.append(_r2(price))
            matrix.append(row)

        return {
            "wacc_axis": wacc_axis,
            "gt_axis":   gt_axis,
            "matrix":    matrix,
            "base_wacc": base_wacc,
            "base_gt":   round(base_g_t * 100, 2),
        }

    # ── 5. Çarpan analizi ─────────────────────────────────────────────────────

    def _run_multiples_analysis(self) -> dict:
        """
        NTM EPS × forward P/E, NTM Revenue × P/S çarpanları ile
        sektör medyan değerlemesi.
        """
        vm  = self.data.get("valuation_multiples", {})
        ae  = self.data.get("analyst_expectations", {})
        bs  = self.data.get("balance_sheet", {})
        inc = self.data.get("income_statement", {})

        current_price = self._current_price()
        results: dict = {}

        # -- EPS × Forward P/E --
        ntm_eps    = _safe(ae.get("ntm_eps_estimate"))
        fwd_pe     = _safe(vm.get("pe_forward"))
        trailing_pe = _safe(vm.get("pe_trailing"))

        # Şirketin uzun dönem ortalama P/E'si (5 yıllık EPS üzerinden hesap)
        hist_pe = self._historical_pe_avg()

        if ntm_eps and fwd_pe:
            results["pe_implied_price"] = _r2(float(ntm_eps) * float(fwd_pe))
        if ntm_eps and hist_pe:
            results["hist_pe_implied_price"] = _r2(float(ntm_eps) * float(hist_pe))

        # -- Revenue × P/S --
        ntm_rev = _safe(ae.get("ntm_revenue_estimate"))
        p_s     = _safe(vm.get("price_to_sales"))
        shares  = _safe(bs.get("shares_outstanding")) or 1

        if ntm_rev and p_s:
            rev_per_share = float(ntm_rev) / float(shares)
            results["ps_implied_price"] = _r2(rev_per_share * float(p_s))

        # -- EV/EBITDA çarpanı --
        ev_ebitda = _safe(vm.get("ev_ebitda"))
        ann_years = list(inc.get("annual", {}).values())
        latest_ebitda = None
        if ann_years:
            latest_ebitda = _safe(ann_years[0].get("ebitda"))

        net_debt = _safe(bs.get("net_debt")) or 0
        if ev_ebitda and latest_ebitda and shares:
            implied_ev     = float(ev_ebitda) * float(latest_ebitda)
            eq_val         = implied_ev - float(net_debt)
            results["ev_ebitda_implied_price"] = _r2(eq_val / float(shares))

        # -- Konsensüs hedef fiyat --
        if ae.get("price_target_mean"):
            results["consensus_target"] = _r2(ae["price_target_mean"])

        # Multiples'ların basit ortalaması
        price_estimates = [
            v for k, v in results.items()
            if k.endswith("_price") or k == "consensus_target"
        ]
        if price_estimates:
            valid = [p for p in price_estimates if p is not None]
            results["multiples_avg_price"] = _r2(sum(valid) / len(valid)) if valid else None

        results["current_price"] = current_price
        results["fwd_pe"]        = fwd_pe
        results["trailing_pe"]   = trailing_pe
        results["hist_pe_avg"]   = _r2(hist_pe)
        results["ev_ebitda"]     = ev_ebitda
        results["ntm_eps"]       = ntm_eps
        results["ntm_revenue"]   = ntm_rev

        return results

    def _historical_pe_avg(self) -> float | None:
        """Son 5 yılın ortalama P/E'sini hesapla (fiyat / EPS)."""
        try:
            tkr  = yf.Ticker(self.ticker)
            hist = tkr.history(period="5y", interval="1mo")
            fin  = tkr.financials
            if hist.empty or fin.empty:
                return None

            pe_list = []
            for col in fin.columns[:4]:
                ni  = _safe(fin.loc["Net Income", col]) if "Net Income" in fin.index else None
                sh  = _safe(fin.loc["Diluted Average Shares", col]) if "Diluted Average Shares" in fin.index else None
                eps = (float(ni) / float(sh)) if (ni and sh) else None
                if eps and eps > 0:
                    # O tarihe en yakın kapanış fiyatı
                    close_prices = hist["Close"]
                    ts = col
                    nearest = close_prices.index.asof(ts)
                    if nearest in close_prices.index:
                        price = float(close_prices[nearest])
                        pe_list.append(price / eps)

            return round(sum(pe_list) / len(pe_list), 2) if pe_list else None
        except Exception:
            return None

    # ── 6b. Comps analizi çalıştır ─────────────────────────────────────────────

    def _run_comps(self) -> dict:
        """Comparable Company Analysis modülünü çalıştır."""
        try:
            from comps import comps_valuation
            return comps_valuation(self.ticker, self.data)
        except Exception as e:
            logger.error("Comps analysis failed: %s", e)
            return {
                "peers_used": [], "peer_group": "Error",
                "peer_count": 0, "comps_fair_value": None,
            }

    # ── 7. Hedef fiyat harmanlama ─────────────────────────────────────────────

    def _blend_target(
        self,
        dcf_price:        float | None,
        multiples_result: dict,
        comps_result:     dict | None = None,
    ) -> float | None:
        """
        Nihai hedef fiyat = %50 DCF + %25 Çarpan ortalaması + %25 Comps

        Mevcut olmayan bileşenler için ağırlıklar yeniden dağıtılır.
        """
        m_price = multiples_result.get("multiples_avg_price")
        c_price = (comps_result or {}).get("comps_fair_value")

        # (bileşen, ağırlık) çiftleri — sadece mevcut olanları al
        components: list[tuple[float, float]] = []
        if dcf_price:
            components.append((float(dcf_price), DCF_WEIGHT))
        if m_price:
            components.append((float(m_price),   MULTIPLES_WEIGHT))
        if c_price:
            components.append((float(c_price),   COMPS_WEIGHT))

        if not components:
            return None

        # Ağırlıkları normalize et (toplam = 1.0)
        total_w = sum(w for _, w in components)
        blended = sum(p * w / total_w for p, w in components)
        return blended

    # ── 7. Yardımcılar ────────────────────────────────────────────────────────

    def _current_price(self) -> float | None:
        vm = self.data.get("valuation_multiples", {})
        return _safe(vm.get("current_price"))

    def _recommendation(self, upside: float | None) -> str:
        """Upside yüzdesine göre öneri üret."""
        if upside is None:
            return "N/A"
        if upside >= 20:
            return "STRONG BUY"
        if upside >= 10:
            return "BUY"
        if upside >= -5:
            return "HOLD"
        if upside >= -15:
            return "SELL"
        return "STRONG SELL"


# ─── Kolaylık fonksiyonu ──────────────────────────────────────────────────────

def analyze_ticker(ticker: str, scenarios: dict | None = None) -> dict:
    """
    Shortcut: veri topla + analiz et, tek satırda.

    Örnek
    -----
    >>> from analyzer import analyze_ticker
    >>> result = analyze_ticker("MSFT")
    """
    from data_collector import collect_ticker_data
    data = collect_ticker_data(ticker)
    return Analyzer(data, scenarios).analyze()


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    result = analyze_ticker(ticker)

    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np, pandas as pd
            if isinstance(obj, np.integer):   return int(obj)
            if isinstance(obj, np.floating):  return float(obj)
            if isinstance(obj, np.ndarray):   return obj.tolist()
            if isinstance(obj, pd.Timestamp): return str(obj)
            return super().default(obj)

    print(json.dumps(result, indent=2, cls=_Encoder))
