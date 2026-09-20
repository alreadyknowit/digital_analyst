"""
sentiment.py
────────────
Aşama 4 – Analist Konsensüsü ve Duygu Analizi Modülü

data_collector.py çıktısındaki analist beklentilerini ve yfinance haber
başlıklarını kullanarak:
  1. Analist konsensüs skoru (güçlü al → güçlü sat aralığı)
  2. Haber duygu analizi (FinBERT veya anahtar-kelime tabanlı yedek)
  3. Birleşik piyasa algısı skoru

üretir.

Kullanım
--------
>>> from data_collector import collect_ticker_data
>>> from sentiment import compute_market_perception
>>> data = collect_ticker_data("MSFT")
>>> result = compute_market_perception(data, "MSFT")
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

logger = logging.getLogger("sentiment")


# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

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
    """Değeri 2 ondalık basamağa yuvarla; None/NaN ise None döndür."""
    v = _safe(value)
    return round(float(v), 2) if v is not None else None


# ─── FinBERT Model Önbelleği ──────────────────────────────────────────────────

_MODEL_CACHE: dict[str, Any] = {}


def _load_finbert() -> tuple | None:
    """
    FinBERT modelini ve tokenizer'ını yükle (ilk çağrıda).
    transformers/torch yoksa None döndürür.
    """
    if "finbert" in _MODEL_CACHE:
        return _MODEL_CACHE["finbert"]

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch  # noqa: F401

        logger.info("FinBERT modeli yükleniyor…")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()
        result = (tokenizer, model)
        _MODEL_CACHE["finbert"] = result
        logger.info("FinBERT modeli başarıyla yüklendi.")
        return result
    except ImportError:
        logger.warning(
            "transformers/torch kurulu değil — anahtar-kelime tabanlı yedek "
            "duygu analizi kullanılacak"
        )
        _MODEL_CACHE["finbert"] = None
        return None
    except Exception as e:
        logger.error("FinBERT yüklenirken hata: %s", e)
        _MODEL_CACHE["finbert"] = None
        return None


# ─── Anahtar Kelime Tabanlı Yedek Duygu Analizi ──────────────────────────────

POSITIVE_KEYWORDS = [
    "surge", "jump", "gain", "rise", "bull", "upgrade", "beat", "record",
    "high", "growth", "strong", "profit", "rally", "boom", "soar", "outperform",
]

NEGATIVE_KEYWORDS = [
    "fall", "drop", "decline", "bear", "downgrade", "miss", "loss", "low",
    "weak", "crash", "plunge", "cut", "warning", "risk", "fear", "sell",
]


def _keyword_sentiment(title: str) -> dict:
    """
    Basit anahtar-kelime tabanlı duygu analizi.
    Her başlık için pozitif/negatif/nötr olasılık döndürür.
    """
    lower = title.lower()
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in lower)
    total = pos_count + neg_count

    if total == 0:
        return {"positive": 0.1, "negative": 0.1, "neutral": 0.8}

    pos_ratio = pos_count / total
    neg_ratio = neg_count / total

    # Ağırlıklı dağılım: anahtar kelime oranına göre
    positive = pos_ratio * 0.7
    negative = neg_ratio * 0.7
    neutral = 1.0 - positive - negative

    return {"positive": positive, "negative": negative, "neutral": neutral}


# ─── Bölüm A: Analist Konsensüsü ─────────────────────────────────────────────

def compute_consensus(data: dict) -> dict:
    """
    data_collector.py çıktısındaki analist beklentilerini kullanarak
    konsensüs skoru, öneri etiketi ve revizyon trendi hesaplar.

    Parametreler
    ------------
    data : collect_ticker_data() sonucu

    Döndürür
    --------
    dict : target_mean, target_low, target_high, upside_pct,
           recommendation_score, recommendation_label, analyst_count,
           revision_trend
    """
    try:
        ae = data.get("analyst_expectations", {})
        vm = data.get("valuation_multiples", {})

        # ── Fiyat hedefleri ──────────────────────────────────────────────
        target_mean = _safe(ae.get("price_target_mean"), 0)
        target_low = _safe(ae.get("price_target_low"), 0)
        target_high = _safe(ae.get("price_target_high"), 0)

        # ── Dağılım (Strong Buy → Strong Sell) ──────────────────────────
        strong_buy = int(_safe(ae.get("strong_buy"), 0))
        buy = int(_safe(ae.get("buy"), 0))
        hold = int(_safe(ae.get("hold"), 0))
        sell = int(_safe(ae.get("sell"), 0))
        strong_sell = int(_safe(ae.get("strong_sell"), 0))

        total = strong_buy + buy + hold + sell + strong_sell
        num_analyst = _safe(ae.get("num_analyst_opinions"), total)

        # ── Konsensüs skoru [-2, +2] ────────────────────────────────────
        if total > 0:
            recommendation_score = (
                strong_buy * 2 + buy * 1 + hold * 0
                + sell * (-1) + strong_sell * (-2)
            ) / total
        else:
            recommendation_score = 0.0

        # ── Öneri etiketi ────────────────────────────────────────────────
        if recommendation_score >= 1.5:
            recommendation_label = "Strong Buy"
        elif recommendation_score >= 0.5:
            recommendation_label = "Buy"
        elif recommendation_score >= -0.5:
            recommendation_label = "Hold"
        elif recommendation_score >= -1.5:
            recommendation_label = "Sell"
        else:
            recommendation_label = "Strong Sell"

        # ── Upside (mevcut fiyat → hedef fiyat) ─────────────────────────
        current_price = _safe(vm.get("current_price"))
        upside_pct = None
        if current_price and target_mean and float(current_price) > 0:
            upside_pct = _r2(
                (float(target_mean) / float(current_price) - 1) * 100
            )

        # ── Revizyon trendi (son 90 gün) ─────────────────────────────────
        revision_trend = _get_revision_trend(data.get("ticker", ""))

        logger.info(
            "Konsensüs: skor=%.2f (%s) | hedef=$%.2f | analist=%d | trend=%s",
            recommendation_score, recommendation_label,
            float(target_mean) if target_mean else 0,
            num_analyst or 0, revision_trend,
        )

        return {
            "target_mean": _r2(target_mean),
            "target_low": _r2(target_low),
            "target_high": _r2(target_high),
            "upside_pct": upside_pct,
            "recommendation_score": _r2(recommendation_score),
            "recommendation_label": recommendation_label,
            "analyst_count": int(num_analyst) if num_analyst else 0,
            "revision_trend": revision_trend,
        }

    except Exception as e:
        logger.error("Konsensüs hesaplaması başarısız: %s", e)
        return {
            "target_mean": None,
            "target_low": None,
            "target_high": None,
            "upside_pct": None,
            "recommendation_score": 0.0,
            "recommendation_label": "Hold",
            "analyst_count": 0,
            "revision_trend": "stable",
        }


def _get_revision_trend(ticker: str) -> str:
    """
    Son 90 günde analist yükseltme/düşürme sayılarını karşılaştırarak
    revizyon trendini belirle.

    Döndürür: "upgrading" | "stable" | "downgrading"
    """
    try:
        if not ticker:
            return "stable"

        tk = yf.Ticker(ticker)
        ud = tk.upgrades_downgrades

        if ud is None or ud.empty:
            return "stable"

        # Son 90 günü filtrele
        cutoff = datetime.now() - timedelta(days=90)

        # Index tarih formatına göre filtrele
        if hasattr(ud.index, 'tz'):
            cutoff_ts = cutoff
        else:
            cutoff_ts = cutoff

        try:
            recent = ud.loc[ud.index >= str(cutoff.date())]
        except Exception:
            recent = ud.tail(20)  # Yedek: son 20 kayıt

        if recent.empty:
            return "stable"

        # "Action" sütununda upgrade/downgrade say
        actions = recent.get("Action", recent.get("action", None))
        if actions is None:
            return "stable"

        actions_lower = actions.astype(str).str.lower()
        upgrades = actions_lower.str.contains("up|upgrade|initiated|reiterat", na=False).sum()
        downgrades = actions_lower.str.contains("down|downgrade", na=False).sum()

        if upgrades > downgrades * 1.5:
            return "upgrading"
        elif downgrades > upgrades * 1.5:
            return "downgrading"
        else:
            return "stable"

    except Exception as e:
        logger.warning("Revizyon trendi alınamadı (%s): %s", ticker, e)
        return "stable"


# ─── Bölüm B: Duygu Analizi ──────────────────────────────────────────────────

def compute_sentiment(ticker: str) -> dict:
    """
    yfinance haber başlıklarını çekerek FinBERT (veya anahtar-kelime)
    tabanlı duygu analizi yapar.

    Parametreler
    ------------
    ticker : Hisse senedi kodu (ör: "MSFT")

    Döndürür
    --------
    dict : overall_score, overall_label, positive_pct, neutral_pct,
           negative_pct, article_count, headlines_analyzed, method
    """
    try:
        # ── Haberleri çek ────────────────────────────────────────────────
        tk = yf.Ticker(ticker)
        news_raw = tk.news

        if not news_raw:
            logger.warning("Haber bulunamadı: %s", ticker)
            return _empty_sentiment("finbert")

        # ── Son 30 günü filtrele ─────────────────────────────────────────
        cutoff = datetime.now() - timedelta(days=30)
        articles = []
        for item in news_raw:
            # yfinance news formatı: 'content' → 'title', 'providerPublishTime'
            # veya düz dict olarak 'title', 'providerPublishTime'
            title = None
            pub_time = None

            if isinstance(item, dict):
                # Yeni format: item["content"]["title"] vb.
                content = item.get("content", item)
                if isinstance(content, dict):
                    title = content.get("title")
                    pub_time = content.get("providerPublishTime")
                    if pub_time is None:
                        pub_date_str = content.get("pubDate")
                        if pub_date_str:
                            try:
                                pub_time = datetime.fromisoformat(
                                    pub_date_str.replace("Z", "+00:00")
                                ).timestamp()
                            except Exception:
                                pub_time = None
                else:
                    title = item.get("title")
                    pub_time = item.get("providerPublishTime")

            if not title:
                continue

            # Tarih filtresi
            if pub_time is not None:
                try:
                    article_dt = datetime.fromtimestamp(float(pub_time))
                    if article_dt < cutoff:
                        continue
                except (TypeError, ValueError, OSError):
                    pass  # Tarih ayrıştırılamadı → dahil et

            articles.append(title)

        if not articles:
            logger.warning("Son 30 günde haber bulunamadı: %s", ticker)
            return _empty_sentiment("finbert")

        logger.info("%s için %d haber başlığı bulundu.", ticker, len(articles))

        # ── Duygu analizi yöntemi seç ────────────────────────────────────
        finbert = _load_finbert()

        if finbert is not None:
            return _analyze_with_finbert(articles, finbert)
        else:
            return _analyze_with_keywords(articles)

    except Exception as e:
        logger.error("Duygu analizi başarısız (%s): %s", ticker, e)
        return _empty_sentiment("keyword")


def _empty_sentiment(method: str) -> dict:
    """Boş/varsayılan duygu analizi sonucu döndür."""
    return {
        "overall_score": 0.0,
        "overall_label": "Neutral",
        "positive_pct": 0.0,
        "neutral_pct": 100.0,
        "negative_pct": 0.0,
        "article_count": 0,
        "headlines_analyzed": [],
        "method": method,
    }


def _analyze_with_finbert(articles: list[str], finbert_pair: tuple) -> dict:
    """
    FinBERT modeli ile haber başlıklarının duygu analizini yap.

    FinBERT çıkış etiketleri: [positive, negative, neutral]
    """
    import torch

    tokenizer, model = finbert_pair

    all_pos: list[float] = []
    all_neg: list[float] = []
    all_neu: list[float] = []
    headline_results: list[dict] = []

    for title in articles:
        try:
            with torch.no_grad():
                inputs = tokenizer(
                    title,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # FinBERT sırası: positive=0, negative=1, neutral=2
                pos_p = float(probs[0][0])
                neg_p = float(probs[0][1])
                neu_p = float(probs[0][2])

            all_pos.append(pos_p)
            all_neg.append(neg_p)
            all_neu.append(neu_p)

            score = pos_p - neg_p  # [-1, +1]
            if score > 0.15:
                label = "Positive"
            elif score < -0.15:
                label = "Negative"
            else:
                label = "Neutral"

            headline_results.append({
                "title": title,
                "sentiment": label,
                "score": _r2(score),
            })

        except Exception as e:
            logger.warning("Başlık analiz edilemedi: '%s' — %s", title[:50], e)
            continue

    if not all_pos:
        return _empty_sentiment("finbert")

    # ── Ortalama olasılıklar (yüzde) ────────────────────────────────────
    avg_pos = np.mean(all_pos)
    avg_neg = np.mean(all_neg)
    avg_neu = np.mean(all_neu)

    overall_score = float(np.mean([p - n for p, n in zip(all_pos, all_neg)]))

    if overall_score > 0.15:
        overall_label = "Positive"
    elif overall_score < -0.15:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    # En uç 10 başlığı seç (mutlak skora göre)
    headline_results.sort(key=lambda x: abs(x["score"]), reverse=True)
    top_headlines = headline_results[:10]

    logger.info(
        "FinBERT sonuç: skor=%.2f (%s) | pos=%.1f%% neg=%.1f%% neu=%.1f%% | %d başlık",
        overall_score, overall_label,
        avg_pos * 100, avg_neg * 100, avg_neu * 100,
        len(all_pos),
    )

    return {
        "overall_score": _r2(overall_score),
        "overall_label": overall_label,
        "positive_pct": _r2(avg_pos * 100),
        "neutral_pct": _r2(avg_neu * 100),
        "negative_pct": _r2(avg_neg * 100),
        "article_count": len(all_pos),
        "headlines_analyzed": top_headlines,
        "method": "finbert",
    }


def _analyze_with_keywords(articles: list[str]) -> dict:
    """
    FinBERT mevcut değilse anahtar-kelime tabanlı yedek duygu analizi.
    """
    all_pos: list[float] = []
    all_neg: list[float] = []
    all_neu: list[float] = []
    headline_results: list[dict] = []

    for title in articles:
        try:
            scores = _keyword_sentiment(title)
            pos_p = scores["positive"]
            neg_p = scores["negative"]
            neu_p = scores["neutral"]

            all_pos.append(pos_p)
            all_neg.append(neg_p)
            all_neu.append(neu_p)

            score = pos_p - neg_p
            if score > 0.15:
                label = "Positive"
            elif score < -0.15:
                label = "Negative"
            else:
                label = "Neutral"

            headline_results.append({
                "title": title,
                "sentiment": label,
                "score": _r2(score),
            })

        except Exception as e:
            logger.warning("Başlık analiz edilemedi: '%s' — %s", title[:50], e)
            continue

    if not all_pos:
        return _empty_sentiment("keyword")

    # ── Ortalama olasılıklar (yüzde) ────────────────────────────────────
    avg_pos = np.mean(all_pos)
    avg_neg = np.mean(all_neg)
    avg_neu = np.mean(all_neu)

    overall_score = float(np.mean([p - n for p, n in zip(all_pos, all_neg)]))

    if overall_score > 0.15:
        overall_label = "Positive"
    elif overall_score < -0.15:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    # En uç 10 başlığı seç (mutlak skora göre)
    headline_results.sort(key=lambda x: abs(x["score"]), reverse=True)
    top_headlines = headline_results[:10]

    logger.info(
        "Anahtar-kelime sonuç: skor=%.2f (%s) | pos=%.1f%% neg=%.1f%% neu=%.1f%% | %d başlık",
        overall_score, overall_label,
        avg_pos * 100, avg_neg * 100, avg_neu * 100,
        len(all_pos),
    )

    return {
        "overall_score": _r2(overall_score),
        "overall_label": overall_label,
        "positive_pct": _r2(avg_pos * 100),
        "neutral_pct": _r2(avg_neu * 100),
        "negative_pct": _r2(avg_neg * 100),
        "article_count": len(all_pos),
        "headlines_analyzed": top_headlines,
        "method": "keyword",
    }


# ─── Bölüm C: Birleşik Piyasa Algısı ─────────────────────────────────────────

def compute_market_perception(data: dict, ticker: str) -> dict:
    """
    Analist konsensüsü ve duygu analizini birleştirerek
    piyasa algısı skoru üretir.

    Ağırlıklar: %60 konsensüs + %40 duygu

    Parametreler
    ------------
    data   : collect_ticker_data() sonucu
    ticker : Hisse senedi kodu (ör: "MSFT")

    Döndürür
    --------
    dict : consensus, sentiment, market_perception_score,
           market_perception_label
    """
    try:
        logger.info("▶ Piyasa algısı hesaplanıyor: %s …", ticker)

        # ── 1. Konsensüs ─────────────────────────────────────────────────
        consensus = compute_consensus(data)

        # ── 2. Duygu analizi ─────────────────────────────────────────────
        sentiment = compute_sentiment(ticker)

        # ── 3. Birleşik skor ─────────────────────────────────────────────
        #   Konsensüs skoru [-2, +2] → normalize [-1, +1]
        consensus_normalized = (consensus.get("recommendation_score") or 0.0) / 2.0
        sentiment_score = sentiment.get("overall_score") or 0.0

        market_score = 0.60 * consensus_normalized + 0.40 * sentiment_score

        # ── 4. Etiket ────────────────────────────────────────────────────
        if market_score > 0.3:
            market_label = "Bullish"
        elif market_score > 0.1:
            market_label = "Slightly Bullish"
        elif market_score > -0.1:
            market_label = "Neutral"
        elif market_score > -0.3:
            market_label = "Slightly Bearish"
        else:
            market_label = "Bearish"

        logger.info(
            "✔ Piyasa algısı: skor=%.2f (%s) | konsensüs=%.2f | duygu=%.2f",
            market_score, market_label,
            consensus_normalized, sentiment_score,
        )

        return {
            "consensus": consensus,
            "sentiment": sentiment,
            "market_perception_score": _r2(market_score),
            "market_perception_label": market_label,
        }

    except Exception as e:
        logger.error("Piyasa algısı hesaplaması başarısız (%s): %s", ticker, e)
        return {
            "consensus": compute_consensus(data),
            "sentiment": _empty_sentiment("keyword"),
            "market_perception_score": 0.0,
            "market_perception_label": "Neutral",
        }


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"

    from data_collector import collect_ticker_data

    data = collect_ticker_data(ticker)
    result = compute_market_perception(data, ticker)
    print(json.dumps(result, indent=2, default=str))
