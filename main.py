#!/usr/bin/env python3
"""
main.py
───────
digital_analyst – CLI giriş noktası

Kullanım
--------
  python3 main.py MSFT
  python3 main.py AAPL --save aapl_report.txt
  python3 main.py TSLA --json
  python3 main.py MSFT AAPL GOOGL
  python3 main.py MSFT --json --save msft.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _make_encoder():
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):    return int(obj)
            if isinstance(obj, np.floating):   return float(obj)
            if isinstance(obj, np.ndarray):    return obj.tolist()
            if isinstance(obj, pd.Timestamp):  return str(obj)
            return super().default(obj)
    return _Encoder


def run_ticker(ticker: str, args: argparse.Namespace) -> dict | None:
    from data_collector import collect_ticker_data
    from analyzer import Analyzer
    from report_generator import ReportGenerator

    try:
        data   = collect_ticker_data(ticker)
        result = Analyzer(data).analyze()
        rg     = ReportGenerator(result)

        if not args.json:
            rg.print()

        if args.save:
            path = Path(args.save)
            if args.json:
                path.write_text(
                    json.dumps(rg.to_dict(), indent=2, cls=_make_encoder()),
                    encoding="utf-8",
                )
            else:
                rg.save(path)
            print(f"\n  💾 Rapor kaydedildi → {path.resolve()}\n")

        return rg.to_dict()

    except Exception as e:
        print(f"\n  ❌ {ticker} analizi başarısız: {e}\n", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        prog="digital_analyst",
        description="Python Finansal Analiz Sistemi — DCF + Çarpan Değerleme",
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        metavar="TICKER",
        help="Analiz edilecek hisse kodu(ları) (örn: MSFT AAPL GOOGL)",
    )
    parser.add_argument(
        "--save", "-s",
        metavar="DOSYA",
        help="Raporu dosyaya kaydet (örn: msft_report.txt)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="JSON formatında çıktı ver",
    )

    args = parser.parse_args()

    results = []
    for ticker in args.tickers:
        # Birden fazla ticker varsa save adını otomatik ayarla
        save = args.save
        if args.save and len(args.tickers) > 1:
            p    = Path(args.save)
            save_arg = argparse.Namespace(**vars(args))
            save_arg.save = str(p.parent / f"{ticker.lower()}_{p.name}")
        else:
            save_arg = args

        r = run_ticker(ticker.upper().strip(), save_arg)
        if r:
            results.append(r)

    # Birden fazla ticker → özet tablo
    if len(results) > 1:
        print("\n" + "═" * 72)
        print("  KARŞILAŞTIRMA ÖZETİ".center(72))
        print("═" * 72)
        print(f"  {'Ticker':<8} {'Mevcut':>10} {'Hedef':>10} {'Upside':>10} {'Öneri'}")
        print("  " + "─" * 60)
        for r in results:
            print(
                f"  {r['ticker']:<8}"
                f"  ${r.get('current_price') or 0:>8.2f}"
                f"  ${r.get('target_price') or 0:>8.2f}"
                f"  {(r.get('upside_pct') or 0):>8.1f}%"
                f"  {r.get('recommendation', 'N/A')}"
            )
        print("  " + "─" * 60 + "\n")

    if args.json and not args.save and len(results) == 1:
        print(json.dumps(results[0], indent=2, cls=_make_encoder()))
    elif args.json and not args.save and len(results) > 1:
        print(json.dumps(results, indent=2, cls=_make_encoder()))


if __name__ == "__main__":
    main()
