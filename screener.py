"""
screener.py
───────────
Toplu Hisse Tarayıcı

Birden fazla hisseyi paralel olarak analiz eder,
Temel Kalite Skoru (0–100) hesaplar ve upside'a göre sıralar.

Kullanım
--------
  python3 screener.py MSFT AAPL GOOGL META AMZN TSLA NVDA
  python3 screener.py --watchlist tech_watchlist.txt
  python3 screener.py MSFT AAPL --save screener_results.json
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("screener")

# ─── Kalite skoru ağırlıkları ─────────────────────────────────────────────────
# Her kriter 0–100 puan alır; aşağıdaki ağırlıklarla toplanır.

QUALITY_WEIGHTS = {
    "roe":           0.20,   # Özsermaye getirisi
    "roic":          0.20,   # Yatırım getirisi
    "rev_growth":    0.15,   # Gelir büyümesi (3Y CAGR)
    "eps_growth":    0.15,   # EPS büyümesi (3Y CAGR)
    "debt_ebitda":   0.15,   # Borç/EBITDA (ters – düşük iyidir)
    "fcf_margin":    0.15,   # FCF marjı (FCF / Revenue)
}

# Benchmark değerler (sektör ortalaması varsayımı)
BENCHMARKS = {
    "roe_excellent":      30.0,   # %30 üstü → maks puan
    "roic_excellent":     20.0,
    "rev_cagr_excellent": 20.0,
    "eps_cagr_excellent": 20.0,
    "debt_ebitda_safe":    1.0,   # 1x altı → maks puan
    "debt_ebitda_risky":   5.0,   # 5x üstü → 0 puan
    "fcf_margin_excellent": 25.0, # %25 üstü → maks puan
}


# ─── Yardımcılar ──────────────────────────────────────────────────────────────

def _safe(value: Any, default=None) -> Any:
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


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _r2(v: Any) -> float | None:
    x = _safe(v)
    return round(float(x), 2) if x is not None else None


# ─── Kalite Skoru ─────────────────────────────────────────────────────────────

def compute_quality_score(analysis: dict) -> dict:
    """
    0–100 arasında Temel Kalite Skoru hesapla.

    Kriterler
    ---------
    ROE        : yüksek iyidir (≥30% → 100p)
    ROIC       : yüksek iyidir (≥20% → 100p)
    Rev CAGR   : yüksek iyidir (≥20% → 100p)
    EPS CAGR   : yüksek iyidir (≥20% → 100p)
    Debt/EBITDA: düşük iyidir  (≤1x → 100p, ≥5x → 0p)
    FCF Margin : yüksek iyidir (≥25% → 100p)
    """
    data   = analysis.get("_raw_data", {})
    gq     = data.get("growth_and_quality", {})
    inc    = data.get("income_statement", {})
    ann    = inc.get("annual", {})

    scores: dict[str, float | None] = {}

    # ROE
    roe = _safe(gq.get("roe"))
    if roe is not None:
        scores["roe"] = _clamp(float(roe) / BENCHMARKS["roe_excellent"] * 100)
    else:
        scores["roe"] = None

    # ROIC
    roic = _safe(gq.get("roic"))
    if roic is not None:
        scores["roic"] = _clamp(float(roic) / BENCHMARKS["roic_excellent"] * 100)
    else:
        scores["roic"] = None

    # Revenue 3Y CAGR
    rev_cagr = _safe(gq.get("revenue_3y_cagr"))
    if rev_cagr is not None:
        scores["rev_growth"] = _clamp(float(rev_cagr) / BENCHMARKS["rev_cagr_excellent"] * 100)
    else:
        scores["rev_growth"] = None

    # EPS 3Y CAGR
    eps_cagr = _safe(gq.get("eps_3y_cagr"))
    if eps_cagr is not None:
        scores["eps_growth"] = _clamp(float(eps_cagr) / BENCHMARKS["eps_cagr_excellent"] * 100)
    else:
        scores["eps_growth"] = None

    # Debt / EBITDA (ters)
    d_ebitda = _safe(gq.get("debt_to_ebitda"))
    if d_ebitda is not None:
        d = float(d_ebitda)
        risky  = BENCHMARKS["debt_ebitda_risky"]
        safe   = BENCHMARKS["debt_ebitda_safe"]
        if d <= safe:
            scores["debt_ebitda"] = 100.0
        elif d >= risky:
            scores["debt_ebitda"] = 0.0
        else:
            scores["debt_ebitda"] = _clamp(100.0 * (risky - d) / (risky - safe))
    else:
        scores["debt_ebitda"] = None

    # FCF Margin = FCF / Revenue (en son yıl)
    fcf_margin_score = None
    if ann:
        latest = ann[sorted(ann.keys(), reverse=True)[0]]
        fcf = _safe(latest.get("fcf"))
        rev = _safe(latest.get("revenue"))
        if fcf and rev and float(rev) > 0:
            margin_pct = float(fcf) / float(rev) * 100
            fcf_margin_score = _clamp(margin_pct / BENCHMARKS["fcf_margin_excellent"] * 100)
    scores["fcf_margin"] = fcf_margin_score

    # Ağırlıklı toplam (sadece mevcut kriterler üzerinden)
    total_w  = 0.0
    total_wv = 0.0
    for key, weight in QUALITY_WEIGHTS.items():
        v = scores.get(key)
        if v is not None:
            total_wv += v * weight
            total_w  += weight

    composite = round(total_wv / total_w, 1) if total_w > 0 else None

    return {
        "composite":  composite,
        "components": {k: _r2(v) for k, v in scores.items()},
    }


def quality_grade(score: float | None) -> str:
    """Skora göre harf notu."""
    if score is None:     return "N/A"
    if score >= 85:       return "A+"
    if score >= 75:       return "A"
    if score >= 65:       return "B+"
    if score >= 55:       return "B"
    if score >= 45:       return "C+"
    if score >= 35:       return "C"
    return "D"


# ─── Screener ─────────────────────────────────────────────────────────────────

class Screener:
    """
    Toplu hisse tarayıcı — paralel analiz, sıralama ve özet tablo.

    Kullanım
    --------
    >>> sc = Screener(["MSFT", "AAPL", "GOOGL"])
    >>> results = sc.run()
    >>> sc.print_summary(results)
    """

    def __init__(self, tickers: list[str], max_workers: int = 4):
        self.tickers     = [t.upper().strip() for t in tickers]
        self.max_workers = max_workers

    def run(self) -> list[dict]:
        """Tüm ticker'ları paralel analiz et, upside'a göre sırala."""
        from data_collector import collect_ticker_data
        from analyzer import Analyzer

        results: list[dict] = []

        logger.info("Screening %d tickers with %d workers …", len(self.tickers), self.max_workers)

        def _analyze_one(ticker: str) -> dict | None:
            try:
                data     = collect_ticker_data(ticker)
                analysis = Analyzer(data).analyze()
                analysis["_raw_data"] = data   # kalite skoru için
                qs = compute_quality_score(analysis)
                return {
                    "ticker":         ticker,
                    "name":           data.get("company_info", {}).get("name", ticker)[:30],
                    "current_price":  analysis.get("current_price"),
                    "target_price":   analysis.get("target_price"),
                    "upside_pct":     analysis.get("upside_pct"),
                    "recommendation": analysis.get("recommendation"),
                    "wacc":           analysis.get("wacc", {}).get("wacc"),
                    "base_fcf_b":     _r2((analysis.get("base_fcf") or 0) / 1e9),
                    "fwd_pe":         analysis.get("multiples", {}).get("fwd_pe"),
                    "ev_ebitda":      analysis.get("multiples", {}).get("ev_ebitda"),
                    "quality_score":  qs["composite"],
                    "quality_grade":  quality_grade(qs["composite"]),
                    "quality_detail": qs["components"],
                    "weighted_dcf":   analysis.get("weighted_dcf_price"),
                    "multiples_avg":  analysis.get("multiples", {}).get("multiples_avg_price"),
                    "consensus":      analysis.get("multiples", {}).get("consensus_target"),
                    "beta":           analysis.get("wacc", {}).get("beta"),
                }
            except Exception as e:
                logger.error("Failed to analyze %s: %s", ticker, e)
                return {"ticker": ticker, "error": str(e)}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_analyze_one, t): t for t in self.tickers}
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    results.append(r)

        # Upside'a göre sırala (None en sona)
        results.sort(
            key=lambda x: float(x.get("upside_pct") or -9999),
            reverse=True,
        )
        return results

    # ── Terminal çıktısı ──────────────────────────────────────────────────────

    def print_summary(self, results: list[dict]) -> None:
        """Renkli özet tablo yazdır."""
        try:
            from report_generator import _C as C
            color = True
        except Exception:
            color = False
            class C:
                def __getattr__(self, _): return ""

        w = 100
        print()
        print(C.CYAN + C.BOLD + "═" * w + C.RESET)
        print(C.WHITE + C.BOLD + "  SCREENER SONUÇLARI".center(w) + C.RESET)
        print(C.CYAN + "═" * w + C.RESET)
        print()

        # Başlık
        hdr = (
            f"  {'Ticker':<7} {'İsim':<28} {'Mevcut':>8} {'Hedef':>8} "
            f"{'Upside':>8} {'Öneri':<14} {'Kalite':>7} {'Not':>4} "
            f"{'Fwd PE':>7} {'EV/EBITDA':>10}"
        )
        print(C.DIM + hdr + C.RESET)
        print("  " + "─" * (w - 2))

        for r in results:
            if r.get("error"):
                print(f"  {r['ticker']:<7}  ❌ {r['error']}")
                continue

            upside = r.get("upside_pct")
            rec    = r.get("recommendation", "N/A")

            if upside and float(upside) >= 15:
                u_c = C.GREEN
            elif upside and float(upside) >= 0:
                u_c = C.YELLOW
            else:
                u_c = C.RED

            rec_colors = {
                "STRONG BUY": C.GREEN + C.BOLD,
                "BUY":        C.GREEN,
                "HOLD":       C.YELLOW,
                "SELL":       C.RED,
                "STRONG SELL":C.RED + C.BOLD,
            }
            r_c = rec_colors.get(rec, C.GRAY)

            qs = r.get("quality_score")
            q_c = C.GREEN if qs and qs >= 65 else (C.YELLOW if qs and qs >= 45 else C.RED)

            cur    = f"${r.get('current_price') or 0:>6.2f}"
            tgt    = f"${r.get('target_price') or 0:>6.2f}"
            up_str = f"{upside or 0:>+7.1f}%"
            qs_str = f"{qs or 0:>5.1f}"
            qg_str = r.get("quality_grade", "N/A")
            pe_str = f"{r.get('fwd_pe') or 0:>5.1f}x"
            ev_str = f"{r.get('ev_ebitda') or 0:>7.1f}x"

            print(
                f"  {C.WHITE}{r['ticker']:<7}{C.RESET} "
                f"{C.DIM}{r.get('name', ''):<28}{C.RESET} "
                f"{cur:>8} "
                f"{C.GREEN}{tgt:>8}{C.RESET} "
                f"{u_c}{up_str:>8}{C.RESET} "
                f"{r_c}{rec:<14}{C.RESET} "
                f"{q_c}{qs_str:>7}{C.RESET} "
                f"{q_c}{qg_str:>4}{C.RESET} "
                f"{pe_str:>7} "
                f"{ev_str:>10}"
            )

        print("  " + "─" * (w - 2))
        print()

        # En iyi / en kötü
        valid = [r for r in results if not r.get("error") and r.get("upside_pct") is not None]
        if valid:
            best  = valid[0]
            worst = valid[-1]
            top_q = max(valid, key=lambda x: x.get("quality_score") or 0)
            print(
                f"  {C.GREEN}▲ En Yüksek Upside :{C.RESET} "
                f"{best['ticker']} ({best.get('upside_pct', 0):+.1f}%)"
            )
            print(
                f"  {C.RED}▼ En Düşük Upside  :{C.RESET} "
                f"{worst['ticker']} ({worst.get('upside_pct', 0):+.1f}%)"
            )
            print(
                f"  {C.CYAN}★ En Yüksek Kalite :{C.RESET} "
                f"{top_q['ticker']} (Skor: {top_q.get('quality_score', 0):.1f} / {top_q.get('quality_grade')})"
            )
        print()

    def save_json(self, results: list[dict], path: str | Path) -> None:
        """Sonuçları JSON olarak kaydet."""
        class _Enc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, np.integer):   return int(o)
                if isinstance(o, np.floating):  return float(o)
                if isinstance(o, np.ndarray):   return o.tolist()
                if isinstance(o, pd.Timestamp): return str(o)
                return super().default(o)

        # _raw_data'yı çıkar (çok büyük)
        clean = [{k: v for k, v in r.items() if k != "_raw_data"} for r in results]
        Path(path).write_text(json.dumps(clean, indent=2, cls=_Enc), encoding="utf-8")
        logger.info("Screener results saved → %s", Path(path).resolve())


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,   # screener modunda log'u kısalt
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="screener",
        description="digital_analyst – Toplu Hisse Tarayıcı",
    )
    parser.add_argument(
        "tickers", nargs="*", metavar="TICKER",
        help="Hisse kodu(ları) — ya da --watchlist ile dosyadan oku",
    )
    parser.add_argument(
        "--watchlist", "-w", metavar="DOSYA",
        help="Her satırda bir hisse kodu içeren metin dosyası",
    )
    parser.add_argument(
        "--save", "-s", metavar="DOSYA",
        help="Sonuçları JSON olarak kaydet",
    )
    parser.add_argument(
        "--workers", "-n", type=int, default=4,
        help="Paralel worker sayısı (default: 4)",
    )
    args = parser.parse_args()

    tickers = list(args.tickers)

    if args.watchlist:
        wl_path = Path(args.watchlist)
        if not wl_path.exists():
            print(f"❌ Watchlist dosyası bulunamadı: {wl_path}", file=sys.stderr)
            sys.exit(1)
        extra = [
            line.strip().upper()
            for line in wl_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        tickers += extra

    if not tickers:
        # Demo watchlist
        tickers = ["MSFT", "AAPL", "GOOGL", "META", "AMZN"]
        print(f"ℹ️  Ticker belirtilmedi — demo liste kullanılıyor: {', '.join(tickers)}\n")

    sc      = Screener(tickers, max_workers=args.workers)
    results = sc.run()
    sc.print_summary(results)

    if args.save:
        sc.save_json(results, args.save)
        print(f"  💾 Kaydedildi → {args.save}\n")
