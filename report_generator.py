"""
report_generator.py
───────────────────
Aşama 3 – Hedef Fiyat & Analist Raporu Üretimi

analyzer.py çıktısını alıp gerçek bir Wall Street analist notuna benzer
yapılandırılmış bir rapor üretir.

Çıktı formatları
----------------
  • Konsol  : renkli ANSI terminal çıktısı
  • Metin   : düz .txt dosyası
  • JSON    : makinece okunabilir tam rapor

Kullanım
--------
>>> from analyzer import analyze_ticker
>>> from report_generator import ReportGenerator
>>> result = analyze_ticker("MSFT")
>>> rg = ReportGenerator(result)
>>> rg.print()          # terminale yaz
>>> rg.save("msft_report.txt")   # dosyaya yaz
>>> d = rg.to_dict()    # dict olarak al
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("report_generator")

# ─── ANSI renk kodları ────────────────────────────────────────────────────────

class _C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"

    BG_DARK  = "\033[40m"
    BG_GREEN = "\033[42m"
    BG_RED   = "\033[41m"


# ─── Yardımcılar ──────────────────────────────────────────────────────────────

def _fmt(value: Any, suffix: str = "", prefix: str = "", decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
        return f"{prefix}{v:,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_b(value: Any) -> str:
    """Büyük sayıyı $XB / $XM / $XT olarak formatla."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if abs(v) >= 1e12:  return f"${v/1e12:.2f}T"
        if abs(v) >= 1e9:   return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6:   return f"${v/1e6:.2f}M"
        return f"${v:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _upside_color(upside: float | None) -> str:
    if upside is None:
        return _C.GRAY
    if upside >= 15:
        return _C.GREEN
    if upside >= 0:
        return _C.YELLOW
    return _C.RED


def _rec_color(rec: str) -> str:
    mapping = {
        "STRONG BUY":  _C.GREEN,
        "BUY":         _C.GREEN,
        "HOLD":        _C.YELLOW,
        "SELL":        _C.RED,
        "STRONG SELL": _C.RED,
    }
    return mapping.get(rec.upper(), _C.GRAY)


def _bar(value: float, total: float, width: int = 20, color: str = _C.CYAN) -> str:
    """Basit ASCII progress bar."""
    if total == 0:
        return " " * width
    filled = int(round(value / total * width))
    filled = max(0, min(width, filled))
    return color + "█" * filled + _C.DIM + "░" * (width - filled) + _C.RESET


# ─── Ana sınıf ────────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Analyzer çıktısından tam analist raporu üretir.

    Parametreler
    ------------
    analysis : analyzer.Analyzer.analyze() sonucu
    """

    WIDTH = 72   # toplam karakter genişliği

    def __init__(self, analysis: dict):
        self.a    = analysis
        self.date = datetime.now().strftime("%d %B %Y")

    # ── Public API ────────────────────────────────────────────────────────────

    def print(self) -> None:
        """Renkli raporu terminale yazdır."""
        print(self._render(color=True))

    def to_text(self) -> str:
        """Düz metin olarak döndür (renk kodları olmadan)."""
        return self._render(color=False)

    def save(self, path: str | Path) -> Path:
        """Düz metin raporu dosyaya kaydet."""
        p = Path(path)
        p.write_text(self.to_text(), encoding="utf-8")
        logger.info("Report saved → %s", p.resolve())
        return p

    def to_dict(self) -> dict:
        """Makine okunabilir rapor sözlüğü döndür."""
        a = self.a
        comps = a.get("comps", {})
        return {
            "generated_at":      datetime.now().isoformat(),
            "ticker":            a.get("ticker"),
            "recommendation":    a.get("recommendation"),
            "current_price":     a.get("current_price"),
            "target_price":      a.get("target_price"),
            "upside_pct":        a.get("upside_pct"),
            "wacc_pct":          a.get("wacc", {}).get("wacc"),
            "dcf_scenarios": {
                name: {
                    "intrinsic_price": v.get("intrinsic_price"),
                    "enterprise_value": v.get("enterprise_value"),
                    "tv_as_pct_ev":    v.get("tv_as_pct_ev"),
                }
                for name, v in a.get("dcf", {}).items()
            },
            "weighted_dcf_price":    a.get("weighted_dcf_price"),
            "multiples_avg_price":   a.get("multiples", {}).get("multiples_avg_price"),
            "consensus_target":      a.get("multiples", {}).get("consensus_target"),
            "comps_fair_value":      comps.get("comps_fair_value"),
            "comps_peer_group":      comps.get("peer_group"),
            "comps_peers_used":      comps.get("peers_used", []),
        }

    # ── Render orchestrator ───────────────────────────────────────────────────

    def _render(self, color: bool) -> str:
        c = _C if color else _NoColor()
        lines: list[str] = []

        lines += self._section_header(c)
        lines += self._section_summary(c)
        lines += self._section_wacc(c)
        lines += self._section_dcf(c)
        lines += self._section_sensitivity(c)
        lines += self._section_multiples(c)
        lines += self._section_comps(c)
        lines += self._section_price_bridge(c)
        lines += self._section_risks(c)
        lines += self._section_footer(c)

        return "\n".join(lines)

    # ── Bölüm 0 – Başlık ─────────────────────────────────────────────────────

    def _section_header(self, c) -> list[str]:
        a    = self.a
        rec  = a.get("recommendation", "N/A")
        rec_c = _rec_color(rec) if hasattr(c, "GREEN") else ""

        w = self.WIDTH
        lines = [
            "",
            c.CYAN + c.BOLD + "═" * w + c.RESET,
            c.WHITE + c.BOLD +
            f"  EQUITY RESEARCH  │  {a.get('ticker', 'N/A')}  │  {self.date}".center(w)
            + c.RESET,
            c.CYAN + "═" * w + c.RESET,
            "",
            f"  {c.BOLD}Şirket   :{c.RESET}  {a.get('ticker', 'N/A')}",
            f"  {c.BOLD}Tarih    :{c.RESET}  {self.date}",
            f"  {c.BOLD}Öneri    :{c.RESET}  {rec_c}{c.BOLD}{rec}{c.RESET}",
            f"  {c.BOLD}Mevcut   :{c.RESET}  {_fmt(a.get('current_price'), prefix='$')}",
            f"  {c.BOLD}Hedef    :{c.RESET}  "
            + c.GREEN + c.BOLD + _fmt(a.get('target_price'), prefix='$') + c.RESET,
            f"  {c.BOLD}Upside   :{c.RESET}  "
            + _upside_color(a.get('upside_pct')) + c.BOLD
            + _fmt(a.get('upside_pct'), suffix='%') + c.RESET,
            "",
            c.CYAN + "─" * w + c.RESET,
        ]
        return lines

    # ── Bölüm 1 – Özet ───────────────────────────────────────────────────────

    def _section_summary(self, c) -> list[str]:
        a   = self.a
        m   = a.get("multiples", {})
        gq  = {}  # growth & quality şu an analysis'te yok, collector'dan gelir
        w   = self.WIDTH

        lines = [
            "",
            c.BOLD + c.BLUE + "  ▌ DEĞERLEME ÖZETİ" + c.RESET,
            "",
        ]

        rows = [
            ("DCF Ağırlıklı Fiyat",   _fmt(a.get("weighted_dcf_price"),       prefix="$")),
            ("Çarpan Ort. Fiyat",      _fmt(m.get("multiples_avg_price"),       prefix="$")),
            ("Konsensüs Hedef",        _fmt(m.get("consensus_target"),          prefix="$")),
            ("Baz FCF",                _fmt_b(a.get("base_fcf"))),
            ("NTM EPS Tahmini",        _fmt(m.get("ntm_eps"),                   prefix="$")),
            ("NTM Revenue Tahmini",    _fmt_b(m.get("ntm_revenue"))),
            ("Fwd P/E",               _fmt(m.get("fwd_pe"),                    suffix="x")),
            ("Trailing P/E",          _fmt(m.get("trailing_pe"),               suffix="x")),
            ("EV/EBITDA",             _fmt(m.get("ev_ebitda"),                 suffix="x")),
        ]

        for label, val in rows:
            lines.append(f"    {c.DIM}{label:<28}{c.RESET}  {c.WHITE}{val}{c.RESET}")

        return lines

    # ── Bölüm 2 – WACC ───────────────────────────────────────────────────────

    def _section_wacc(self, c) -> list[str]:
        w_data = self.a.get("wacc", {})
        w      = self.WIDTH

        lines = [
            "",
            c.BOLD + c.BLUE + "  ▌ WACC ANALİZİ" + c.RESET,
            "",
        ]

        rows = [
            ("WACC",                   _fmt(w_data.get("wacc"),    suffix="%")),
            ("Risk-Free Rate (^TNX)",  _fmt(w_data.get("rf"),      suffix="%")),
            ("Beta",                   _fmt(w_data.get("beta"),    decimals=2)),
            ("Özsermaye Maliyeti (Re)", _fmt(w_data.get("re"),     suffix="%")),
            ("Borç Maliyeti (Rd)",     _fmt(w_data.get("rd"),      suffix="%")),
            ("Vergi Oranı",            _fmt(w_data.get("tax_rate"), suffix="%")),
            ("Sermaye Yapısı (E/D)",
             f"{_fmt(w_data.get('e_weight'), suffix='%')} / "
             f"{_fmt(w_data.get('d_weight'), suffix='%')}"),
        ]

        for label, val in rows:
            lines.append(f"    {c.DIM}{label:<30}{c.RESET}  {c.YELLOW}{val}{c.RESET}")

        return lines

    # ── Bölüm 3 – DCF Senaryoları ─────────────────────────────────────────────

    def _section_dcf(self, c) -> list[str]:
        a        = self.a
        dcf      = a.get("dcf", {})
        scen     = a.get("scenarios", {})
        wacc_pct = a.get("wacc", {}).get("wacc", 0) or 0

        lines = [
            "",
            c.BOLD + c.BLUE + "  ▌ DCF DEĞERLEMESİ (2-Aşamalı FCF + Gordon Growth)" + c.RESET,
            "",
            f"    {c.DIM}{'Senaryo':<12} {'Stage1':>8} {'Stage2':>8} "
            f"{'Terminal':>10} {'İçsel Değer':>14} {'TV/EV':>8}{c.RESET}",
            f"    {'─'*60}",
        ]

        scenario_colors = {
            "bull": c.GREEN,
            "base": c.YELLOW,
            "bear": c.RED,
        }
        scenario_labels = {
            "bull": "🐂 Bull",
            "base": "⚖  Base",
            "bear": "🐻 Bear",
        }

        for name in ("bull", "base", "bear"):
            params = scen.get(name, {})
            res    = dcf.get(name, {})
            col    = scenario_colors.get(name, c.WHITE)
            label  = scenario_labels.get(name, name.title())

            intrinsic = res.get("intrinsic_price")
            tv_pct    = res.get("tv_as_pct_ev")

            line = (
                f"    {col}{label:<14}{c.RESET}"
                f"{_fmt(params.get('stage1_growth', 0)*100, suffix='%'):>9}"
                f"{_fmt(params.get('stage2_growth', 0)*100, suffix='%'):>9}"
                f"{_fmt(params.get('terminal_growth', 0)*100, suffix='%'):>11}"
                f"  {col}{c.BOLD}{_fmt(intrinsic, prefix='$'):>12}{c.RESET}"
                f"{_fmt(tv_pct, suffix='%'):>9}"
            )
            lines.append(line)

        lines += [
            f"    {'─'*60}",
            f"    {c.BOLD}{'Ağırlıklı Ortalama':>42}  "
            f"{c.GREEN}{_fmt(a.get('weighted_dcf_price'), prefix='$'):>12}{c.RESET}",
            "",
            f"    {c.DIM}Ağırlıklar: Bull %25 · Base %50 · Bear %25{c.RESET}",
        ]

        # FCF projeksiyonu (base senaryo)
        base_proj = dcf.get("base", {}).get("fcf_projections", [])
        if base_proj:
            lines += [
                "",
                f"    {c.DIM}FCF Projeksiyonu – Base Senaryo (Yıllık):{c.RESET}",
            ]
            max_fcf = max(abs(v) for v in base_proj if v) or 1
            for i, fcf_val in enumerate(base_proj, 1):
                stage   = "Stage1" if i <= 5 else "Stage2"
                bar_len = int(abs(fcf_val or 0) / max_fcf * 25)
                bar     = c.CYAN + "█" * bar_len + c.RESET
                lines.append(
                    f"    {c.DIM}Yıl {i:>2} [{stage}]{c.RESET}  "
                    f"{bar}  {c.WHITE}{_fmt_b(fcf_val)}{c.RESET}"
                )

        return lines

    # ── Bölüm 3b – Duyarlılık Matrisi ─────────────────────────────────────────

    def _section_sensitivity(self, c) -> list[str]:
        """
        WACC (sütun) × Terminal Growth Rate (satır) → İçsel Fiyat matrisi.
        Mevcut fiyatın altındaki hücreler kırmızı, üstündekiler yeşil gösterilir.
        """
        sens = self.a.get("sensitivity", {})
        if not sens or not sens.get("matrix"):
            return []

        wacc_axis = sens["wacc_axis"]    # % listesi  [8.5, 9.5, ...]
        gt_axis   = sens["gt_axis"]      # % listesi  [1.0, 1.5, ...]
        matrix    = sens["matrix"]       # satır=gt, sütun=wacc
        base_wacc = sens["base_wacc"]
        base_gt   = sens["base_gt"]
        cur       = self.a.get("current_price")

        lines = [
            "",
            c.BOLD + c.BLUE + "  ▌ DUYARLILIK MATRİSİ  (WACC × Terminal Büyüme → İçsel Fiyat/Hisse)" + c.RESET,
            "",
        ]

        # Başlık satırı (WACC ekseni)
        header = f"    {c.DIM}{'g↓ / WACC→':>10}{c.RESET}"
        for w in wacc_axis:
            marker = c.BOLD if w == base_wacc else c.DIM
            header += f"  {marker}{w:>5.1f}%{c.RESET}"
        lines.append(header)
        lines.append("    " + "─" * (10 + len(wacc_axis) * 9))

        for row_idx, gt_pct in enumerate(gt_axis):
            row_vals = matrix[row_idx] if row_idx < len(matrix) else []
            gt_marker = c.BOLD if gt_pct == base_gt else c.DIM
            row_str = f"    {gt_marker}{gt_pct:>8.1f}%{c.RESET}"

            for col_idx, price in enumerate(row_vals):
                w = wacc_axis[col_idx] if col_idx < len(wacc_axis) else None
                is_base = (w == base_wacc and gt_pct == base_gt)

                if price is None:
                    cell = f"  {'N/A':>7}"
                else:
                    # Renk: mevcut fiyata göre
                    if cur and float(price) >= float(cur) * 1.10:
                        cell_c = c.GREEN
                    elif cur and float(price) >= float(cur):
                        cell_c = c.YELLOW
                    else:
                        cell_c = c.RED

                    prefix = c.BOLD if is_base else ""
                    suffix = "◀" if is_base else " "
                    cell = f"  {prefix}{cell_c}${price:>6.0f}{c.RESET}{c.DIM}{suffix}{c.RESET}"

                row_str += cell
            lines.append(row_str)

        lines += [
            "    " + "─" * (10 + len(wacc_axis) * 9),
            f"    {c.DIM}Renk: {c.GREEN}Yeşil{c.RESET}{c.DIM} = mevcut fiyat+10% üstü  "
            f"{c.YELLOW}Sarı{c.RESET}{c.DIM} = üstü  "
            f"{c.RED}Kırmızı{c.RESET}{c.DIM} = altı   "
            f"◀ = baz senaryo{c.RESET}",
        ]
        return lines

    # ── Bölüm 4 – Çarpan Analizi ─────────────────────────────────────────────

    def _section_multiples(self, c) -> list[str]:
        m      = self.a.get("multiples", {})
        cur    = m.get("current_price")

        lines = [
            "",
            c.BOLD + c.BLUE + "  ▌ ÇARPAN ANALİZİ (Implied Price)" + c.RESET,
            "",
            f"    {c.DIM}{'Yöntem':<32} {'Fiyat':>12} {'Mevcut Fiyata Fark':>20}{c.RESET}",
            f"    {'─'*66}",
        ]

        methods = [
            ("Fwd P/E × NTM EPS",        m.get("pe_implied_price")),
            ("Hist. P/E × NTM EPS",       m.get("hist_pe_implied_price")),
            ("P/S × NTM Revenue/Hisse",   m.get("ps_implied_price")),
            ("EV/EBITDA Çarpanı",         m.get("ev_ebitda_implied_price")),
            ("Konsensüs Hedef",           m.get("consensus_target")),
        ]

        for label, price in methods:
            if price is None:
                continue
            diff = None
            diff_str = "N/A"
            diff_c = c.GRAY
            if cur and price:
                diff = (float(price) / float(cur) - 1) * 100
                diff_str = _fmt(diff, suffix="%", decimals=1)
                diff_c = c.GREEN if diff >= 0 else c.RED
            lines.append(
                f"    {c.DIM}{label:<32}{c.RESET}  "
                f"{c.WHITE}{_fmt(price, prefix='$'):>12}{c.RESET}  "
                f"{diff_c}{diff_str:>18}{c.RESET}"
            )

        lines += [
            f"    {'─'*66}",
            f"    {c.BOLD}{'Çarpan Ortalaması':<32}{c.RESET}  "
            f"{c.GREEN}{c.BOLD}{_fmt(m.get('multiples_avg_price'), prefix='$'):>12}{c.RESET}",
        ]

        return lines

    # ── Bölüm 4b – Comps Analizi ──────────────────────────────────────────────

    def _section_comps(self, c) -> list[str]:
        """Comparable Company Analysis — peer karşılaştırma tablosu."""
        comps = self.a.get("comps", {})
        if not comps or not comps.get("peers_used"):
            return []

        peer_group   = comps.get("peer_group", "N/A")
        peers_used   = comps.get("peers_used", [])
        peer_medians = comps.get("peer_medians", {})
        target_mults = comps.get("target_multiples", {})
        growth_prem  = comps.get("growth_premium", 0)
        implied      = comps.get("implied_prices", {})
        fair_value   = comps.get("comps_fair_value")
        peer_details = comps.get("peer_details", [])
        cur          = self.a.get("current_price")

        lines = [
            "",
            c.BOLD + c.BLUE + "  ▌ COMPS ANALİZİ (Comparable Company Analysis)" + c.RESET,
            "",
            f"    {c.DIM}Peer Grubu :{c.RESET}  {c.WHITE}{peer_group}{c.RESET}",
            f"    {c.DIM}Peer Sayısı:{c.RESET}  {c.WHITE}{len(peers_used)}{c.RESET}  "
            f"{c.DIM}({', '.join(peers_used)}){c.RESET}",
            "",
        ]

        # Peer detay tablosu
        if peer_details:
            lines.append(
                f"    {c.DIM}{'Peer':<8} {'Fwd P/E':>8} {'EV/EBITDA':>10} "
                f"{'EV/Rev':>8} {'P/FCF':>8} {'Büyüme':>8}{c.RESET}"
            )
            lines.append(f"    {'─'*54}")

            for pd_item in peer_details:
                fpe  = _fmt(pd_item.get("forward_pe"),  suffix="x", decimals=1) if pd_item.get("forward_pe") else "N/A"
                eveb = _fmt(pd_item.get("ev_ebitda"),    suffix="x", decimals=1) if pd_item.get("ev_ebitda")  else "N/A"
                evr  = _fmt(pd_item.get("ev_revenue"),   suffix="x", decimals=1) if pd_item.get("ev_revenue") else "N/A"
                pfcf = _fmt(pd_item.get("price_fcf"),    suffix="x", decimals=1) if pd_item.get("price_fcf")  else "N/A"
                rg   = _fmt((pd_item.get("rev_growth") or 0) * 100, suffix="%", decimals=1) if pd_item.get("rev_growth") else "N/A"
                lines.append(
                    f"    {c.DIM}{pd_item['ticker']:<8}{c.RESET}"
                    f"{fpe:>8} {eveb:>10} {evr:>8} {pfcf:>8} {rg:>8}"
                )

            lines.append(f"    {'─'*54}")

        # Medyan vs Hedef karşılaştırma
        lines += [
            "",
            f"    {c.DIM}{'Çarpan':<20} {'Peer Med.':>10} {'Hedef':>10} {'Fark':>10}{c.RESET}",
            f"    {'─'*52}",
        ]

        compare_keys = [
            ("forward_pe",  "Fwd P/E"),
            ("ev_ebitda",   "EV/EBITDA"),
            ("ev_revenue",  "EV/Revenue"),
            ("price_fcf",   "Price/FCF"),
            ("peg_ratio",   "PEG Ratio"),
        ]

        for key, label in compare_keys:
            pm = peer_medians.get(key)
            tm = target_mults.get(key)
            if pm is None and tm is None:
                continue

            pm_str = _fmt(pm, suffix="x", decimals=1) if pm else "N/A"
            tm_str = _fmt(tm, suffix="x", decimals=1) if tm else "N/A"

            diff_str = "N/A"
            diff_c   = c.GRAY
            if pm and tm:
                diff_pct = (float(tm) / float(pm) - 1) * 100
                diff_str = _fmt(diff_pct, suffix="%", decimals=1)
                # Düşük çarpan = ucuz = yeşil (P/E, EV/EBITDA gibi)
                diff_c = c.GREEN if diff_pct < 0 else c.RED

            lines.append(
                f"    {c.DIM}{label:<20}{c.RESET}"
                f"{pm_str:>10} {tm_str:>10} "
                f"{diff_c}{diff_str:>10}{c.RESET}"
            )

        lines.append(f"    {'─'*52}")

        # Büyüme primi
        prem_pct = (growth_prem or 0) * 100
        prem_c   = c.GREEN if prem_pct > 0 else (c.RED if prem_pct < 0 else c.GRAY)
        lines.append(
            f"    {c.DIM}Büyüme Primi:{c.RESET}  "
            f"{prem_c}{c.BOLD}{_fmt(prem_pct, suffix='%', decimals=1)}{c.RESET}"
        )

        # İma edilen fiyatlar
        if implied:
            lines += [
                "",
                f"    {c.DIM}{'Yöntem':<28} {'İma Edilen Fiyat':>16}{c.RESET}",
                f"    {'─'*46}",
            ]
            method_labels = {
                "pe_comps":         "P/E Comps",
                "ev_ebitda_comps":  "EV/EBITDA Comps",
                "ev_revenue_comps": "EV/Revenue Comps",
                "price_fcf_comps":  "Price/FCF Comps",
            }
            for mkey, mlabel in method_labels.items():
                price = implied.get(mkey)
                if price is None:
                    continue
                p_c = c.GREEN if (cur and float(price) >= float(cur)) else c.RED
                lines.append(
                    f"    {c.DIM}{mlabel:<28}{c.RESET}  "
                    f"{p_c}{_fmt(price, prefix='$'):>14}{c.RESET}"
                )
            lines.append(f"    {'─'*46}")

        # Comps fair value
        if fair_value:
            fv_c = c.GREEN if (cur and float(fair_value) >= float(cur)) else c.RED
            lines.append(
                f"    {c.BOLD}{'Comps Fair Value':<28}{c.RESET}  "
                f"{fv_c}{c.BOLD}{_fmt(fair_value, prefix='$'):>14}{c.RESET}"
            )

        return lines

    # ── Bölüm 5 – Hedef Fiyat Köprüsü ────────────────────────────────────────

    def _section_price_bridge(self, c) -> list[str]:
        a          = self.a
        cur        = a.get("current_price")
        dcf_p      = a.get("weighted_dcf_price")
        mul_p      = a.get("multiples", {}).get("multiples_avg_price")
        comps_p    = a.get("comps", {}).get("comps_fair_value")
        target     = a.get("target_price")
        upside     = a.get("upside_pct")
        rec        = a.get("recommendation", "N/A")

        # Dinamik ağırlık gösterimi
        dcf_w  = int(a.get("dcf_weight", 0.50) * 100)
        mul_w  = int(a.get("multiples_weight", 0.25) * 100)
        comp_w = int(a.get("comps_weight", 0.25) * 100)

        lines = [
            "",
            c.BOLD + c.BLUE + "  ▌ HEDEF FİYAT KÖPRÜSÜ" + c.RESET,
            "",
            f"    {c.DIM}{'Bileşen':<28} {'Ağırlık':>8} {'Fiyat':>12}{c.RESET}",
            f"    {'─'*50}",
            f"    {'DCF (Ağırlıklı)  ':<28}  {'%' + str(dcf_w):>6}  {_fmt(dcf_p, prefix='$'):>12}",
            f"    {'Çarpan Analizi   ':<28}  {'%' + str(mul_w):>6}  {_fmt(mul_p, prefix='$'):>12}",
            f"    {'Comps Analizi    ':<28}  {'%' + str(comp_w):>6}  {_fmt(comps_p, prefix='$'):>12}",
            f"    {'─'*50}",
            f"    {c.BOLD}{'12 Ay Hedef Fiyat':<28}{'':>8}  "
            f"{c.GREEN}{c.BOLD}{_fmt(target, prefix='$'):>12}{c.RESET}",
            "",
            f"    Mevcut Fiyat   :  {_fmt(cur, prefix='$')}",
            f"    Beklenen Getiri:  "
            + _upside_color(upside) + c.BOLD + _fmt(upside, suffix="%") + c.RESET,
            f"    Öneri          :  "
            + _rec_color(rec) + c.BOLD + rec + c.RESET,
        ]
        return lines

    # ── Bölüm 6 – Risk Faktörleri ─────────────────────────────────────────────

    def _section_risks(self, c) -> list[str]:
        a   = self.a
        upside = a.get("upside_pct")
        wacc   = a.get("wacc", {}).get("wacc", 0) or 0
        dcf    = a.get("dcf", {})

        bull_price = dcf.get("bull", {}).get("intrinsic_price")
        bear_price = dcf.get("bear", {}).get("intrinsic_price")

        lines = [
            "",
            c.BOLD + c.BLUE + "  ▌ RİSK & DUYARLILIK" + c.RESET,
            "",
        ]

        # Bull / Bear aralığı
        if bull_price and bear_price:
            lines += [
                f"    {c.DIM}Bull (içsel)  :{c.RESET}  "
                f"{c.GREEN}{_fmt(bull_price, prefix='$')}{c.RESET}",
                f"    {c.DIM}Bear (içsel)  :{c.RESET}  "
                f"{c.RED}{_fmt(bear_price, prefix='$')}{c.RESET}",
                f"    {c.DIM}Senaryo Aralığı:{c.RESET}  "
                f"{_fmt(bull_price - bear_price, prefix='$')} genişliği",
            ]

        # WACC duyarlılığı (±1%)
        lines += [
            "",
            f"    {c.DIM}WACC Duyarlılığı (±1pp):{c.RESET}",
        ]
        base_intrinsic = dcf.get("base", {}).get("intrinsic_price")
        base_ev        = dcf.get("base", {}).get("enterprise_value")
        wacc_dec       = wacc / 100.0
        if base_intrinsic and base_ev:
            # Yaklaşık: ΔP ≈ ΔEV/shares, ΔEV ≈ -EV × Δr / (WACC-g)
            g_t  = a.get("scenarios", {}).get("base", {}).get("terminal_growth", 0.025)
            denom = max(wacc_dec - g_t, 0.001)
            sens  = float(base_ev) * 0.01 / denom
            shares = dcf.get("base", {}).get("shares_outstanding") or 1
            per_share_sens = sens / float(shares)
            lines.append(
                f"    {c.DIM}  WACC -1pp  → yaklaşık {c.GREEN}"
                f"+${per_share_sens:.2f}/hisse{c.RESET}"
            )
            lines.append(
                f"    {c.DIM}  WACC +1pp  → yaklaşık {c.RED}"
                f"-${per_share_sens:.2f}/hisse{c.RESET}"
            )

        # Genel risk uyarıları
        lines += [
            "",
            f"    {c.YELLOW}⚠  Bu rapor yalnızca algoritmik analiz içermektedir.{c.RESET}",
            f"    {c.YELLOW}   Yatırım kararları için lisanslı danışmana başvurun.{c.RESET}",
        ]
        return lines

    # ── Bölüm 7 – Alt bilgi ──────────────────────────────────────────────────

    def _section_footer(self, c) -> list[str]:
        w = self.WIDTH
        return [
            "",
            c.CYAN + "─" * w + c.RESET,
            c.GRAY + c.DIM +
            f"  digital_analyst  •  Algorithmik Araştırma  •  {self.date}".center(w)
            + c.RESET,
            c.CYAN + "─" * w + c.RESET,
            "",
        ]


# ─── Renksiz fallback ─────────────────────────────────────────────────────────

class _NoColor:
    """Tüm renk kodlarını boş string döndüren dummy nesne."""
    def __getattr__(self, _):
        return ""


# ─── Kolaylık fonksiyonu ──────────────────────────────────────────────────────

def generate_report(ticker: str, save_path: str | Path | None = None) -> str:
    """
    Shortcut: veri topla → analiz et → rapor üret.

    Parametreler
    ------------
    ticker    : Hisse kodu (ör. "MSFT")
    save_path : Dosya yolu (None → sadece döndürür)

    Örnek
    -----
    >>> from report_generator import generate_report
    >>> text = generate_report("AAPL", "aapl_report.txt")
    """
    from analyzer import analyze_ticker
    result = analyze_ticker(ticker)
    rg     = ReportGenerator(result)
    rg.print()
    if save_path:
        rg.save(save_path)
    return rg.to_text()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ticker    = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    generate_report(ticker, save_path)
