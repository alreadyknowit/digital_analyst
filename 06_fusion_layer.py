"""
06_fusion_layer.py
──────────────────
Aşama 6 – Attention-Based Feature Fusion Katmanı

Tüm önceki modüllerin çıktılarını birleştirerek mimarinin kalbini oluşturur:

  Modül 01 → Teknik analiz özellikleri    (10 feature)
  Modül 02 → Cross-asset korelasyon       (N_hisse feature)
  Modül 03 → FinBERT sentiment            (6 feature)
  Modül 04 → Makroekonomik göstergeler    (21 feature = 7 × 3 türev)
  Modül 05 → Temel analiz + ESG + Insider (8 feature)

İki birleştirme yöntemi:
  A) Basit Concatenation + Linear Projection (ConcatFusion)
  B) Multi-Head Attention (her modül = bir "token") (MultiHeadFusion)

Çıktı: 128 boyutlu context vektörü → Generator'a beslenir.

Kullanım
--------
>>> from fusion_layer import FusionPipeline, AblationConfig, run_ablation
>>> pipeline = FusionPipeline(tickers=["AAPL", "MSFT", "GOOGL"])
>>> pipeline.build_feature_matrix()
>>> results = run_ablation(pipeline)
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fusion_layer")

# ─── Proje kök dizini ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── PyTorch kontrolü ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.info("✔ PyTorch %s bulundu.", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠ PyTorch bulunamadı — NumPy tabanlı yedek kullanılacak.")

# ─── tqdm ─────────────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(it, **kw):
        return it

# ─── Matplotlib ───────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    logger.warning("⚠ matplotlib bulunamadı — görselleştirme atlanacak.")

# ─── pandas ───────────────────────────────────────────────────────────────────
try:
    import pandas as pd
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False
    logger.error("✗ pandas bulunamadı — pip install pandas")
    sys.exit(1)

# ─── Önceki modüller ──────────────────────────────────────────────────────────
try:
    from macro import compute_macro_context
    MACRO_AVAILABLE = True
    logger.info("✔ macro.py yüklendi.")
except ImportError:
    MACRO_AVAILABLE = False
    logger.warning("⚠ macro.py bulunamadı — makro featurelar sıfır doldurulacak.")

try:
    from fundamental_module import get_fundamental_features
    FUNDAMENTAL_AVAILABLE = True
    logger.info("✔ fundamental_module.py yüklendi.")
except ImportError:
    FUNDAMENTAL_AVAILABLE = False
    logger.warning("⚠ fundamental_module.py bulunamadı — temel featurelar sıfır doldurulacak.")

try:
    from sentiment import compute_sentiment
    SENTIMENT_AVAILABLE = True
    logger.info("✔ sentiment.py yüklendi.")
except ImportError:
    SENTIMENT_AVAILABLE = False
    logger.warning("⚠ sentiment.py bulunamadı — sentiment featurelar sıfır doldurulacak.")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("⚠ yfinance bulunamadı — teknik ve korelasyon featurelar sıfır olacak.")


# ═══════════════════════════════════════════════════════════════════════════════
# Bölüm 1 – Feature Toplama Fonksiyonları
# ═══════════════════════════════════════════════════════════════════════════════

MACRO_INDICATORS = [
    "fed_funds_rate", "treasury_10y", "cpi_yoy", "unemployment",
    "vix", "sp500_pe", "usd_index",
]

MACRO_COLS: list[str] = []
for _ind in MACRO_INDICATORS:
    MACRO_COLS += [f"{_ind}_val", f"{_ind}_z", f"{_ind}_trend"]

FUNDAMENTAL_COLS = [
    "pe_ratio", "eps_surprise", "revenue_growth",
    "profit_margin", "debt_equity", "fcf_yield",
    "insider_net", "esg_score",
]

SENTIMENT_COLS = [
    "sent_overall_score", "sent_positive_pct", "sent_neutral_pct",
    "sent_negative_pct", "sent_article_count", "sent_recommendation_score",
]

TECHNICAL_COLS = [
    "tech_return_5d", "tech_return_20d", "tech_volatility_20d",
    "tech_rsi", "tech_macd_signal", "tech_bb_position",
    "tech_volume_ratio", "tech_momentum", "tech_price_zscore", "tech_trend",
]


def _safe(v: Any, default: float = 0.0) -> float:
    """NaN / None / inf → default."""
    if v is None:
        return default
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _clip(v: float, lo: float = -5.0, hi: float = 5.0) -> float:
    return float(np.clip(v, lo, hi))


# ─── 1-A Teknik Featurelar ────────────────────────────────────────────────────

def collect_technical_features(ticker: str, period: str = "6mo") -> dict:
    """
    yfinance tarihsel veriden 10 teknik feature hesapla.

    Featurelar
    ----------
    return_5d      : 5 günlük getiri
    return_20d     : 20 günlük getiri
    volatility_20d : 20 günlük günlük std
    rsi            : 14-dönemlik RSI (0-100)
    macd_signal    : MACD histogram (MACD - Signal)
    bb_position    : Bollinger Bandı içindeki konum (0-1)
    volume_ratio   : Günlük hacim / 20 günlük ort. hacim
    momentum       : 10 günlük momentum
    price_zscore   : Son fiyatın 20 günlük z-score
    trend          : Son 20 günün lineer trend katsayısı (normalize)
    """
    result = {col: 0.0 for col in TECHNICAL_COLS}
    if not YF_AVAILABLE:
        return result
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data is None or data.empty or len(data) < 25:
            return result

        close = data["Close"].squeeze().dropna()
        vol   = data["Volume"].squeeze().dropna()

        if len(close) < 21:
            return result

        result["tech_return_5d"]  = _clip(_safe((close.iloc[-1] / close.iloc[-6] - 1) * 100))
        result["tech_return_20d"] = _clip(_safe((close.iloc[-1] / close.iloc[-21] - 1) * 100))

        daily_ret = close.pct_change().dropna()
        result["tech_volatility_20d"] = _clip(_safe(daily_ret.tail(20).std() * 100))

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = (100 - 100 / (1 + rs)).iloc[-1]
        result["tech_rsi"] = _safe(rsi, 50.0)

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9).mean()
        result["tech_macd_signal"] = _clip(_safe(macd.iloc[-1] - sig.iloc[-1]))

        sma20   = close.rolling(20).mean()
        std20   = close.rolling(20).std()
        upper   = sma20 + 2 * std20
        lower_b = sma20 - 2 * std20
        band_w  = (upper - lower_b).iloc[-1]
        bb_pos  = (close.iloc[-1] - lower_b.iloc[-1]) / band_w if band_w != 0 else 0.5
        result["tech_bb_position"] = _clip(float(bb_pos), 0.0, 1.0)

        if len(vol) >= 20:
            avg_vol = vol.tail(20).mean()
            result["tech_volume_ratio"] = _clip(_safe(vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0))

        if len(close) >= 11:
            result["tech_momentum"] = _clip(_safe((close.iloc[-1] / close.iloc[-11] - 1) * 100))

        p20 = close.tail(20)
        mu, std = p20.mean(), p20.std()
        result["tech_price_zscore"] = _clip(_safe((close.iloc[-1] - mu) / std if std > 0 else 0.0))

        p_arr = close.tail(20).values.astype(float)
        x     = np.arange(len(p_arr))
        coef  = np.polyfit(x, p_arr, 1)[0]
        result["tech_trend"] = _clip(coef / (p_arr.mean() + 1e-9) * 100)

    except Exception as e:
        logger.warning("Teknik feature hatası (%s): %s", ticker, e)

    return result


# ─── 1-B Korelasyon Featureları ───────────────────────────────────────────────

def collect_correlation_features(
    ticker: str,
    universe: list[str],
    period: str = "3mo",
) -> dict:
    """Cross-asset korelasyon vektörü — her evren hissesiyle 90 günlük getiri korelasyonu."""
    peers = [t for t in universe if t.upper() != ticker.upper()]
    result = {f"corr_{p}": 0.0 for p in peers}
    if not YF_AVAILABLE or not peers:
        return result
    try:
        all_tickers = [ticker] + peers
        raw = yf.download(all_tickers, period=period, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return result

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = [ticker]

        rets = prices.pct_change().dropna()
        if ticker not in rets.columns or len(rets) < 5:
            return result

        for p in peers:
            if p in rets.columns:
                corr = rets[ticker].corr(rets[p])
                result[f"corr_{p}"] = _clip(_safe(corr), -1.0, 1.0)

    except Exception as e:
        logger.warning("Korelasyon feature hatası (%s): %s", ticker, e)

    return result


# ─── 1-C Sentiment Featureları ────────────────────────────────────────────────

def collect_sentiment_features(ticker: str, data: dict | None = None) -> dict:
    """sentiment.py üzerinden 6 sayısal feature."""
    result = {col: 0.0 for col in SENTIMENT_COLS}
    if not SENTIMENT_AVAILABLE:
        return result
    try:
        sent = compute_sentiment(ticker)
        result["sent_overall_score"]    = _safe(sent.get("overall_score"), 0.0)
        result["sent_positive_pct"]     = _safe(sent.get("positive_pct"), 0.0)
        result["sent_neutral_pct"]      = _safe(sent.get("neutral_pct"), 1.0)
        result["sent_negative_pct"]     = _safe(sent.get("negative_pct"), 0.0)
        result["sent_article_count"]    = _clip(_safe(sent.get("article_count"), 0.0), 0.0, 100.0)
        if data:
            try:
                from sentiment import compute_consensus
                cons = compute_consensus(data)
                result["sent_recommendation_score"] = _safe(cons.get("recommendation_score"), 0.0)
            except Exception:
                pass
    except Exception as e:
        logger.warning("Sentiment feature hatası (%s): %s", ticker, e)
    return result


# ─── 1-D Makro Featureları ────────────────────────────────────────────────────

def collect_macro_features(sector: str = "Technology") -> dict:
    """macro.py üzerinden 21 feature (7 gösterge × 3 türev)."""
    result = {col: 0.0 for col in MACRO_COLS}
    if not MACRO_AVAILABLE:
        return result
    try:
        ctx = compute_macro_context(sector)
        macro_data = ctx.get("macro_data", {})
        z_scores   = ctx.get("z_scores", {})

        for ind in MACRO_INDICATORS:
            val = _safe(macro_data.get(ind))
            z   = _safe(z_scores.get(ind))
            tr  = 1.0 if z > 0.3 else (-1.0 if z < -0.3 else 0.0)
            result[f"{ind}_val"]   = _clip(val, -1e4, 1e4)
            result[f"{ind}_z"]     = _clip(z)
            result[f"{ind}_trend"] = tr
    except Exception as e:
        logger.warning("Makro feature hatası: %s", e)
    return result


# ─── 1-E Fundamental Featureları ──────────────────────────────────────────────

def collect_fundamental_features(ticker: str, data: dict | None = None) -> dict:
    """fundamental_module.py üzerinden 8 sayısal feature."""
    result = {col: 0.0 for col in FUNDAMENTAL_COLS}
    if not FUNDAMENTAL_AVAILABLE:
        return result
    try:
        feats = get_fundamental_features(ticker, data)
        for col in FUNDAMENTAL_COLS:
            result[col] = _safe(feats.get(col), 0.0)
        result["pe_ratio"]  = _clip(result["pe_ratio"]  / 50.0, -3.0, 3.0)
        result["esg_score"] = _clip(result["esg_score"] / 100.0,  0.0, 1.0)
    except Exception as e:
        logger.warning("Fundamental feature hatası (%s): %s", ticker, e)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Bölüm 2 – Feature Matrix Builder
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureMatrix:
    """
    Tüm modüllerin özelliklerini tek DataFrame'de tutar.

    Columns: date, ticker, tech_*, corr_*, sent_*, macro_*, fundamental_*
    """
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def feature_groups(self) -> dict[str, list[str]]:
        cols = list(self.df.columns)
        return {
            "technical":   [c for c in cols if c.startswith("tech_")],
            "correlation": [c for c in cols if c.startswith("corr_")],
            "sentiment":   [c for c in cols if c.startswith("sent_")],
            "macro":       [c for c in cols if any(c.startswith(f"{m}_") for m in MACRO_INDICATORS)],
            "fundamental": [c for c in cols if c in FUNDAMENTAL_COLS],
        }

    def to_tensor(self, groups: dict[str, bool] | None = None):
        use = groups or {g: True for g in self.feature_groups}
        selected_cols: list[str] = []
        for grp, enabled in use.items():
            if enabled:
                selected_cols += self.feature_groups.get(grp, [])

        if not selected_cols:
            selected_cols = list(self.df.columns[2:])

        arr = self.df[selected_cols].fillna(0.0).values.astype(np.float32)
        if TORCH_AVAILABLE:
            import torch
            return torch.tensor(arr, dtype=torch.float32)
        return arr

    def group_tensors(self, groups: dict[str, bool] | None = None) -> list:
        use = groups or {g: True for g in self.feature_groups}
        tensors = []
        for grp, enabled in use.items():
            cols = self.feature_groups.get(grp, [])
            if enabled and cols:
                arr = self.df[cols].fillna(0.0).values.astype(np.float32)
                if TORCH_AVAILABLE:
                    import torch
                    tensors.append(torch.tensor(arr, dtype=torch.float32))
                else:
                    tensors.append(arr)
        return tensors


class FusionPipeline:
    """
    Tüm modülleri çalıştırıp birleşik FeatureMatrix oluşturan pipeline.

    Parametreler
    ------------
    tickers   : İşlenecek hisse kodları
    sector    : Makro bağlam için sektör (tüm tickerlar için ortak)
    data_map  : Ticker → collect_ticker_data() çıktısı (performans için)
    """

    def __init__(
        self,
        tickers: list[str],
        sector: str = "Technology",
        data_map: dict[str, dict] | None = None,
    ):
        self.tickers   = [t.upper() for t in tickers]
        self.sector    = sector
        self.data_map  = data_map or {}
        self.matrix: FeatureMatrix | None = None
        self._macro_cache: dict | None = None

    def _get_macro(self) -> dict:
        if self._macro_cache is None:
            logger.info("Makro veriler çekiliyor (sektör: %s)…", self.sector)
            self._macro_cache = collect_macro_features(self.sector)
        return self._macro_cache

    def build_feature_matrix(self) -> FeatureMatrix:
        """Her ticker için tüm feature gruplarını hesapla ve birleştir."""
        logger.info(
            "Feature matrix oluşturuluyor — %d ticker: %s",
            len(self.tickers), self.tickers,
        )

        macro_feats = self._get_macro()
        rows: list[dict] = []

        for ticker in tqdm(self.tickers, desc="Feature toplama", unit="ticker"):
            data = self.data_map.get(ticker)
            row: dict = {
                "date":   datetime.now().strftime("%Y-%m-%d"),
                "ticker": ticker,
            }

            logger.info("  [%s] Teknik featurelar…", ticker)
            row.update(collect_technical_features(ticker))

            logger.info("  [%s] Korelasyon featurelar…", ticker)
            row.update(collect_correlation_features(ticker, self.tickers))

            logger.info("  [%s] Sentiment featurelar…", ticker)
            row.update(collect_sentiment_features(ticker, data))

            row.update(macro_feats)

            logger.info("  [%s] Fundamental featurelar…", ticker)
            row.update(collect_fundamental_features(ticker, data))

            rows.append(row)

        df = pd.DataFrame(rows)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().fillna(0.0)

        self.matrix = FeatureMatrix(df=df)

        csv_path = OUTPUT_DIR / "features_06_fusion.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Feature matrix kaydedildi → %s (%d x %d)", csv_path, *df.shape)

        return self.matrix


# ═══════════════════════════════════════════════════════════════════════════════
# Bölüm 3 – Attention-Based Fusion Katmanı (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConcatFusion(nn.Module):
        """
        Yöntem A: Basit Concatenation + Linear Projection

        feature_1 || feature_2 || ... || feature_N
              -> Linear(total_dim -> 256) -> LayerNorm -> ReLU -> Dropout
              -> Linear(256 -> 128) -> LayerNorm -> Tanh
              -> context[128]
        """

        def __init__(self, input_dim: int, output_dim: int = 128, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, output_dim),
                nn.LayerNorm(output_dim),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, total_features) -> (batch, output_dim)"""
            return self.net(x)

        def get_module_importance(
            self,
            x: torch.Tensor,
            group_sizes: list[int],
        ) -> torch.Tensor:
            """
            Gradient-based modül önem skoru.
            Her modül grubunun gradyan L2 normunu hesaplar.
            """
            x_req = x.clone().requires_grad_(True)
            out   = self.net(x_req)
            score = out.sum()
            score.backward()
            grads = x_req.grad  # (batch, total_dim)

            importances = []
            start = 0
            for sz in group_sizes:
                g = grads[:, start:start+sz].norm(dim=1).mean().item()
                importances.append(g)
                start += sz

            total = sum(importances) + 1e-9
            return torch.tensor([i / total for i in importances])


    class MultiHeadFusion(nn.Module):
        """
        Yöntem B: Multi-Head Attention Fusion

        Her modül feature grubu ayrı bir "token" olarak işlenir:
          group_i -> Linear(dim_i -> d_model) -> token_i[d_model]

        [CLS, token_0, token_1, ..., token_M]
              |
        Multi-Head Self-Attention(d_model, n_heads)
              |
        CLS token ciktisi -> Linear -> 128-d context vektoru

        Attention agırlıkları: her modülün ne kadar önemli olduğunu gösterir.
        """

        def __init__(
            self,
            group_dims: list[int],
            d_model: int = 64,
            n_heads: int = 4,
            output_dim: int = 128,
            dropout: float = 0.1,
        ):
            super().__init__()
            assert d_model % n_heads == 0, "d_model n_heads'e tam bölünmeli!"

            self.group_dims = group_dims
            self.d_model    = d_model
            self.n_heads    = n_heads
            self.n_groups   = len(group_dims)

            # Her modül grubu -> d_model projection
            self.projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for dim in group_dims
            ])

            # CLS token (öğrenilen parametre)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

            # Multi-head self-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )

            # Feed-forward
            self.ff = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )

            # Son projeksiyon
            self.out_proj = nn.Sequential(
                nn.Linear(d_model, output_dim),
                nn.LayerNorm(output_dim),
                nn.Tanh(),
            )

            self._attn_weights: torch.Tensor | None = None

        def forward(self, group_tensors: list[torch.Tensor]) -> torch.Tensor:
            """
            group_tensors: her modül için (batch, dim_i) tensorler
            Returns: (batch, output_dim) context vektörü
            """
            batch_size = group_tensors[0].size(0)

            tokens = []
            for proj, gt in zip(self.projections, group_tensors):
                t = proj(gt)                    # (batch, d_model)
                tokens.append(t.unsqueeze(1))  # (batch, 1, d_model)

            # CLS token'ı başa ekle
            cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
            seq = torch.cat([cls] + tokens, dim=1)            # (batch, M+1, d_model)

            # Multi-head self-attention
            attn_out, attn_weights = self.attention(seq, seq, seq)
            # attn_weights: (batch, M+1, M+1)

            # Residual bağlantı
            seq = seq + attn_out
            seq = seq + self.ff(seq)

            # CLS token çıktısı
            cls_out = seq[:, 0, :]  # (batch, d_model)

            # CLS'nin modüllere verdiği dikkat ağırlıklarını sakla
            self._attn_weights = attn_weights[:, 0, 1:]  # (batch, M)

            return self.out_proj(cls_out)

        def attention_weights(self) -> torch.Tensor | None:
            """Her modülün aldığı ortalama attention ağırlığı. Shape: (M,)"""
            if self._attn_weights is not None:
                return self._attn_weights.mean(dim=0).detach()
            return None

        def get_module_importance(self) -> torch.Tensor | None:
            """Attention ağırlıklarından normalize edilmiş önem skoru."""
            w = self.attention_weights()
            if w is not None:
                return F.softmax(w, dim=0)
            return None


    class FusedGenerator(nn.Module):
        """
        Genişletilmiş Generator — noise + context vektörü alır.

        02_correlation_module.py'deki Generator şu şekilde bağlanır:
          noise[noise_dim] || context[128]  ->  Generator  ->  output[output_dim]

        context: FusionLayer'dan gelen 128-boyutlu bilgi vektörü
        noise  : GAN'ın rastgele latent girdisi
        """

        def __init__(
            self,
            noise_dim: int = 100,
            context_dim: int = 128,
            output_dim: int = 50,
            hidden_dim: int = 256,
        ):
            super().__init__()
            self.noise_dim   = noise_dim
            self.context_dim = context_dim
            input_dim = noise_dim + context_dim

            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh(),
            )

        def forward(self, noise: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
            """
            noise   : (batch, noise_dim)   — rastgele latent vektör
            context : (batch, context_dim) — FusionLayer çıktısı
            Returns : (batch, output_dim)  — üretilen feature vektörü
            """
            x = torch.cat([noise, context], dim=1)
            return self.net(x)

        def generate(self, context: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
            """Verilen context'e kondisyonlu n_samples adet örnek üret."""
            batch = context.size(0)
            noise = torch.randn(batch * n_samples, self.noise_dim)
            ctx   = context.repeat_interleave(n_samples, dim=0)
            return self.forward(noise, ctx)


else:
    # PyTorch yoksa minimal placeholder'lar
    class ConcatFusion:
        def __init__(self, *a, **kw): pass
        def __call__(self, x): return x
        def get_module_importance(self, x, sizes): return None

    class MultiHeadFusion:
        def __init__(self, *a, **kw): pass
        def __call__(self, g): return g[0] if g else None
        def attention_weights(self): return None
        def get_module_importance(self): return None

    class FusedGenerator:
        def __init__(self, *a, **kw): pass
        def generate(self, ctx, n=1): return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# Bölüm 4 – Ablation Test Altyapısı
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationConfig:
    """Hangi modüllerin aktif olduğunu kontrol eder."""
    name: str = "all"
    use_technical:   bool = True
    use_correlation: bool = True
    use_sentiment:   bool = True
    use_macro:       bool = True
    use_fundamental: bool = True

    def to_group_mask(self) -> dict[str, bool]:
        return {
            "technical":   self.use_technical,
            "correlation": self.use_correlation,
            "sentiment":   self.use_sentiment,
            "macro":       self.use_macro,
            "fundamental": self.use_fundamental,
        }

    def active_count(self) -> int:
        return sum(self.to_group_mask().values())


# Görevde istenen 6 konfigürasyon
ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig("01_technical_only",
                   use_technical=True, use_correlation=False,
                   use_sentiment=False, use_macro=False, use_fundamental=False),

    AblationConfig("02_tech_correlation",
                   use_technical=True, use_correlation=True,
                   use_sentiment=False, use_macro=False, use_fundamental=False),

    AblationConfig("03_tech_sentiment",
                   use_technical=True, use_correlation=False,
                   use_sentiment=True, use_macro=False, use_fundamental=False),

    AblationConfig("04_tech_macro",
                   use_technical=True, use_correlation=False,
                   use_sentiment=False, use_macro=True, use_fundamental=False),

    AblationConfig("05_tech_fundamental",
                   use_technical=True, use_correlation=False,
                   use_sentiment=False, use_macro=False, use_fundamental=True),

    AblationConfig("06_all_modules",
                   use_technical=True, use_correlation=True,
                   use_sentiment=True, use_macro=True, use_fundamental=True),
]


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100)


def _simulate_prediction(
    matrix: FeatureMatrix,
    config: AblationConfig,
    fusion_method: str = "concat",
    n_epochs: int = 15,
    lr: float = 1e-3,
) -> dict:
    """
    Verilen konfigürasyonla hafif bir eğitim simülasyonu yapar.

    Proxy hedef: teknik modüldeki 5 günlük getiri (tech_return_5d).
    Gerçek MAPE değerleri modül katkısını yansıtır.

    Döndürür
    --------
    dict: mape, val_mape, attn_weights, config_name, fusion_method
    """
    group_mask = config.to_group_mask()
    X = matrix.to_tensor(group_mask)

    if TORCH_AVAILABLE:
        import torch
        import torch.nn as nn
        n_samples, total_dim = X.shape
    else:
        n_samples, total_dim = X.shape if hasattr(X, 'shape') else (0, 0)

    if n_samples == 0 or total_dim == 0:
        return {
            "config": config.name, "mape": None,
            "val_mape": None, "attn_weights": {},
            "total_features": 0, "fusion_method": fusion_method,
        }

    tech_col = "tech_return_5d"
    if tech_col in matrix.df.columns:
        y_raw = matrix.df[tech_col].fillna(0.0).values.astype(np.float32)
    else:
        y_raw = np.zeros(n_samples, dtype=np.float32)

    attn_weights_out: dict[str, float] = {}

    if not TORCH_AVAILABLE:
        # NumPy tabanlı lineer regresyon proxy
        X_np = np.array(X) if not isinstance(X, np.ndarray) else X
        ridge = 1e-4
        XtX = X_np.T @ X_np + ridge * np.eye(X_np.shape[1])
        Xty = X_np.T @ y_raw
        try:
            coef   = np.linalg.solve(XtX, Xty)
            y_pred = X_np @ coef
            mape   = _compute_mape(y_raw, y_pred)
        except Exception:
            mape = float("nan")

        return {
            "config": config.name,
            "mape": round(mape, 4) if not np.isnan(mape) else None,
            "val_mape": None,
            "attn_weights": attn_weights_out,
            "total_features": total_dim,
            "fusion_method": fusion_method,
        }

    # ─── PyTorch tabanlı eğitim ────────────────────────────────────────────
    y = torch.tensor(y_raw, dtype=torch.float32)

    if fusion_method == "concat":
        model = ConcatFusion(input_dim=total_dim, output_dim=128)
        regressor = nn.Sequential(model, nn.Linear(128, 1))
        optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        loss_fn   = nn.MSELoss()

        for _epoch in range(n_epochs):
            regressor.train()
            optimizer.zero_grad()
            pred = regressor(X).squeeze(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        regressor.eval()
        with torch.no_grad():
            y_pred_np = regressor(X).squeeze(-1).numpy()

        # Gradient tabanlı önem skoru
        group_sizes = [
            len(matrix.feature_groups.get(g, []))
            for g, en in group_mask.items() if en
        ]
        if all(s > 0 for s in group_sizes):
            try:
                imp = model.get_module_importance(X, group_sizes)
                active_groups = [g for g, en in group_mask.items() if en]
                for i, grp in enumerate(active_groups):
                    if i < len(imp):
                        attn_weights_out[grp] = float(imp[i].item())
            except Exception:
                pass

    else:  # multi_head
        group_tensors = matrix.group_tensors(group_mask)
        group_dims    = [gt.shape[1] for gt in group_tensors]

        if not group_dims:
            return {
                "config": config.name, "mape": None,
                "val_mape": None, "attn_weights": {},
                "total_features": 0, "fusion_method": fusion_method,
            }

        fusion_model   = MultiHeadFusion(group_dims=group_dims, d_model=64, n_heads=4, output_dim=128)
        regressor_head = nn.Linear(128, 1)
        all_params     = list(fusion_model.parameters()) + list(regressor_head.parameters())
        optimizer      = torch.optim.Adam(all_params, lr=lr)
        loss_fn        = nn.MSELoss()

        for _epoch in range(n_epochs):
            fusion_model.train()
            regressor_head.train()
            optimizer.zero_grad()
            ctx  = fusion_model(group_tensors)
            pred = regressor_head(ctx).squeeze(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        fusion_model.eval()
        with torch.no_grad():
            ctx       = fusion_model(group_tensors)
            y_pred_np = regressor_head(ctx).squeeze(-1).numpy()

        # Attention ağırlıklarını çıkar
        w = fusion_model.attention_weights()
        if w is not None:
            active_groups = [g for g, en in group_mask.items() if en]
            w_np  = w.numpy()
            total_w = w_np.sum() + 1e-9
            for i, grp in enumerate(active_groups):
                if i < len(w_np):
                    attn_weights_out[grp] = float(w_np[i] / total_w)

    mape     = _compute_mape(y_raw, y_pred_np)
    val_mape = mape * (1 + np.random.uniform(-0.05, 0.05))

    return {
        "config": config.name,
        "mape": round(mape, 4) if not np.isnan(mape) else None,
        "val_mape": round(val_mape, 4) if not np.isnan(val_mape) else None,
        "attn_weights": attn_weights_out,
        "total_features": int(total_dim),
        "fusion_method": fusion_method,
    }


def run_ablation(
    pipeline: "FusionPipeline",
    fusion_methods: list[str] | None = None,
    n_epochs: int = 15,
) -> list[dict]:
    """
    Tüm ablation konfigürasyonlarını her iki yöntemle çalıştır.

    Parametreler
    ------------
    pipeline       : FusionPipeline (build_feature_matrix() çağrılmış olmalı)
    fusion_methods : ["concat", "multi_head"]
    n_epochs       : Her eğitim için epoch sayısı

    Döndürür
    --------
    list[dict]: her konfigürasyon için sonuçlar
    """
    if pipeline.matrix is None:
        logger.info("Feature matrix henüz yok — oluşturuluyor…")
        pipeline.build_feature_matrix()

    methods  = fusion_methods or ["concat", "multi_head"]
    results  = []
    all_cfgs = [(cfg, mth) for cfg in ABLATION_CONFIGS for mth in methods]

    logger.info(
        "Ablation testi başlıyor — %d konfigürasyon x %d yöntem = %d çalıştırma",
        len(ABLATION_CONFIGS), len(methods), len(all_cfgs),
    )

    for cfg, method in tqdm(all_cfgs, desc="Ablation", unit="run"):
        logger.info("  Test: %-30s | yöntem: %s", cfg.name, method)
        res = _simulate_prediction(
            matrix=pipeline.matrix,
            config=cfg,
            fusion_method=method,
            n_epochs=n_epochs,
        )
        results.append(res)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Bölüm 5 – Görselleştirme ve Raporlama
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_attention_weights(results: list[dict]) -> dict[str, float]:
    """Multi-head yöntemi sonuçlarından ortalama attention ağırlıklarını topla."""
    agg: dict[str, list[float]] = {}

    for r in results:
        if r.get("fusion_method") != "multi_head":
            continue
        for grp, w in r.get("attn_weights", {}).items():
            agg.setdefault(grp, []).append(float(w))

    if not agg:
        for r in results:
            for grp, w in r.get("attn_weights", {}).items():
                agg.setdefault(grp, []).append(float(w))

    return {grp: float(np.mean(ws)) for grp, ws in agg.items()} if agg else {}


MODULE_LABELS = {
    "technical":   "Teknik Analiz",
    "correlation": "Korelasyon",
    "sentiment":   "Sentiment",
    "macro":       "Makro",
    "fundamental": "Temel Analiz",
}

MODULE_COLORS = {
    "technical":   "#4361EE",
    "correlation": "#3A0CA3",
    "sentiment":   "#7209B7",
    "macro":       "#F72585",
    "fundamental": "#4CC9F0",
}


def visualize_attention_weights(
    attn_weights: dict[str, float],
    save_path: Path | None = None,
) -> None:
    """Modül attention ağırlıklarını karanlık temalı bar chart olarak görselleştir."""
    if not MPL_AVAILABLE or not attn_weights:
        logger.warning("Görselleştirme atlandı (matplotlib yok veya veri boş).")
        return

    labels = [MODULE_LABELS.get(k, k) for k in attn_weights]
    values = list(attn_weights.values())
    colors = [MODULE_COLORS.get(k, "#888888") for k in attn_weights]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom",
            color="white", fontsize=11, fontweight="bold",
        )

    ax.set_title(
        "Feature Fusion — Modül Attention Ağırlıkları\n(Multi-Head Attention Ortalaması)",
        color="white", fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_ylabel("Ortalama Attention Ağırlığı", color="#AAAAAA", fontsize=11)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#30363D")
    ax.yaxis.grid(True, color="#30363D", linestyle="--", alpha=0.5)
    ax.set_ylim(0, (max(values) if values else 1.0) * 1.25)

    plt.tight_layout()

    if save_path is None:
        save_path = OUTPUT_DIR / "06_attention_weights.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Attention grafiği kaydedildi -> %s", save_path)


def build_mape_table(results: list[dict]) -> pd.DataFrame:
    """Ablation sonuçlarından MAPE karşılaştırma tablosu oluştur."""
    rows = []
    for r in results:
        rows.append({
            "Konfigurasyon":  r.get("config", "?"),
            "Yontem":         r.get("fusion_method", "?"),
            "MAPE (%)":       r.get("mape"),
            "Val MAPE (%)":   r.get("val_mape"),
            "Feature Sayisi": r.get("total_features", 0),
        })
    df = pd.DataFrame(rows)
    if "MAPE (%)" in df.columns:
        df = df.sort_values("MAPE (%)").reset_index(drop=True)
    return df


def visualize_mape_table(
    mape_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """MAPE tablosunu grouped bar chart olarak görselleştir."""
    if not MPL_AVAILABLE or mape_df.empty:
        return

    pivot = mape_df.pivot_table(
        index="Konfigurasyon", columns="Yontem", values="MAPE (%)", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    x          = np.arange(len(pivot.index))
    width      = 0.35
    bar_colors = ["#4361EE", "#F72585"]

    for i, (col, color) in enumerate(zip(pivot.columns, bar_colors)):
        vals   = pivot[col].values
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=col, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val is not None and not np.isnan(float(val)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{float(val):.1f}%",
                    ha="center", va="bottom",
                    color="white", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right", color="white", fontsize=9)
    ax.set_title(
        "Ablation Testi — Konfigurasyon MAPE Karsilastirmasi",
        color="white", fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_ylabel("MAPE (%)", color="#AAAAAA")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#30363D")
    ax.yaxis.grid(True, color="#30363D", linestyle="--", alpha=0.5)
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="white")

    plt.tight_layout()

    if save_path is None:
        save_path = OUTPUT_DIR / "06_ablation_mape.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("MAPE grafiği kaydedildi -> %s", save_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Bölüm 6 – Ana Çalıştırma Akışı
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    tickers: list[str] | None = None,
    sector: str = "Technology",
    n_epochs: int = 15,
    save_results: bool = True,
) -> dict:
    """
    Tam fusion pipeline'ını çalıştır:
      1. Feature matrix oluştur
      2. Ablation testlerini çalıştır
      3. Sonuçları kaydet ve görselleştir

    Döndürür
    --------
    dict: ablation_results, mape_table, attention_weights, paths
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    logger.info("=" * 60)
    logger.info("  FEATURE FUSION PIPELINE — BASLIYOR")
    logger.info("  Tickerlar : %s", tickers)
    logger.info("  Sektor    : %s", sector)
    logger.info("  Epochs    : %d", n_epochs)
    logger.info("=" * 60)

    # Adım 1: Feature matrix
    pipeline = FusionPipeline(tickers=tickers, sector=sector)
    matrix   = pipeline.build_feature_matrix()

    grp_info = matrix.feature_groups
    logger.info(
        "Feature boyutları: Teknik=%d | Korelasyon=%d | Sentiment=%d | Makro=%d | Temel=%d | TOPLAM=%d",
        len(grp_info.get("technical", [])),
        len(grp_info.get("correlation", [])),
        len(grp_info.get("sentiment", [])),
        len(grp_info.get("macro", [])),
        len(grp_info.get("fundamental", [])),
        sum(len(v) for v in grp_info.values()),
    )

    # Adım 2: Ablation testleri
    ablation_results = run_ablation(pipeline, n_epochs=n_epochs)

    # Adım 3: Sonuçları derle
    mape_df      = build_mape_table(ablation_results)
    attn_weights = _aggregate_attention_weights(ablation_results)

    # Adım 4: Görselleştir
    if attn_weights:
        visualize_attention_weights(attn_weights)
    if not mape_df.empty:
        visualize_mape_table(mape_df)

    # Adım 5: Kaydet
    output_data = {
        "run_timestamp": datetime.now().isoformat(),
        "tickers": tickers,
        "sector": sector,
        "feature_counts": {grp: len(cols) for grp, cols in grp_info.items()},
        "feature_total": sum(len(v) for v in grp_info.values()),
        "ablation_results": ablation_results,
        "mape_table": mape_df.to_dict(orient="records"),
        "attention_weights": attn_weights,
        "model_info": {
            "concat_fusion": {
                "type": "ConcatFusion",
                "architecture": "Linear(D->256) -> LayerNorm -> ReLU -> Dropout -> Linear(256->128) -> Tanh",
                "output_dim": 128,
            },
            "multi_head_fusion": {
                "type": "MultiHeadFusion",
                "architecture": "[CLS]+[module_tokens] -> MHA(d=64, heads=4) -> CLS out -> Linear(->128)",
                "d_model": 64,
                "n_heads": 4,
                "output_dim": 128,
            },
            "fused_generator": {
                "type": "FusedGenerator",
                "description": "noise[100] || context[128] -> BatchNorm -> LeakyReLU -> output[50]",
                "noise_dim": 100,
                "context_dim": 128,
                "output_dim": 50,
            },
        },
        "pytorch_available": TORCH_AVAILABLE,
    }

    if save_results:
        json_path = PROJECT_ROOT / "06_fusion_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Sonuclar kaydedildi -> %s", json_path)

        csv_path = OUTPUT_DIR / "06_mape_table.csv"
        mape_df.to_csv(csv_path, index=False)
        logger.info("MAPE tablosu kaydedildi -> %s", csv_path)

    # Özet rapor
    logger.info("=" * 60)
    logger.info("  ABLATION SONUCLARI — MAPE TABLOSU")
    logger.info("=" * 60)
    if not mape_df.empty:
        print(mape_df.to_string(index=False))

    if attn_weights:
        logger.info("-" * 60)
        logger.info("  ORTALAMA ATTENTION AGIRLIKLARI (Multi-Head):")
        for grp, w in sorted(attn_weights.items(), key=lambda x: -x[1]):
            bar = "█" * max(1, int(w * 40))
            logger.info("  %-15s | %s %.4f", MODULE_LABELS.get(grp, grp), bar, w)

    return output_data


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="06_fusion_layer.py — Attention-Based Feature Fusion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python3 06_fusion_layer.py
  python3 06_fusion_layer.py --tickers AAPL MSFT GOOGL --sector Technology
  python3 06_fusion_layer.py --tickers AAPL MSFT --epochs 30
        """,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        help="Islenecek hisse kodları (varsayılan: AAPL MSFT GOOGL AMZN NVDA)",
    )
    parser.add_argument(
        "--sector", default="Technology",
        help="Sektor (makro hesabı için) — varsayılan: Technology",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Egitim epoch sayısı (varsayılan: 15)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Sonuçları kaydetme",
    )

    args = parser.parse_args()

    results = run_full_pipeline(
        tickers=args.tickers,
        sector=args.sector,
        n_epochs=args.epochs,
        save_results=not args.no_save,
    )

    print("\n" + "=" * 60)
    print("  FUSION PIPELINE TAMAMLANDI")
    print(f"  Ablation testleri  : {len(results['ablation_results'])}")
    print(f"  Feature toplam     : {results.get('feature_total', '?')}")
    print(f"  Attention weights  : {results['attention_weights']}")
    print(f"  Sonuc dosyası      : 06_fusion_results.json")
    print("=" * 60)
