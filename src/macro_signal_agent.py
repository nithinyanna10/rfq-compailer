import argparse
import datetime as dt
from typing import List, Tuple

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
import plotly.graph_objects as go

from .config import PROVIDER, OLLAMA_BASE_URL, MODEL_NAME


def fetch_headlines(sources: List[str], limit: int = 30) -> List[Tuple[str, str]]:
    feeds = []
    if "fed" in sources:
        feeds.append("https://www.federalreserve.gov/feeds/press_all.xml")
    if "reuters" in sources:
        feeds.append("https://feeds.reuters.com/reuters/businessNews")

    items: List[Tuple[str, str]] = []
    for url in feeds:
        d = feedparser.parse(url)
        for e in d.entries[:limit]:
            title = getattr(e, "title", "").strip()
            summary = getattr(e, "summary", "").strip()
            if title:
                items.append((title, summary))
    return items[:limit]


def score_sentiment(headlines: List[str]) -> List[float]:
    clf = pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
    preds = clf(headlines)
    scores: List[float] = []
    for p in preds:
        label = p["label"].lower()
        score = float(p["score"]) if "score" in p else 0.0
        if label == "positive":
            scores.append(+score)
        elif label == "negative":
            scores.append(-score)
        else:
            scores.append(0.0)
    return scores


def market_mood_index(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return float(np.mean(scores))


def load_yields(days: int = 120) -> pd.DataFrame:
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=days + 10)
    tn = yf.download("^TNX", start=start, end=end, progress=False)
    tn = tn.reset_index()[["Date", "Close"]]
    tn.rename(columns={"Close": "TNX"}, inplace=True)
    tn["TNX"] = tn["TNX"].astype(float)
    tn.sort_values("Date", inplace=True)
    tn["ret"] = tn["TNX"].pct_change()
    return tn.dropna()


def train_predict_direction(yields_df: pd.DataFrame, today_mood: float) -> Tuple[str, float]:
    df = yields_df.copy().tail(90)
    # Features: last 3 returns and today mood (as exogenous feature)
    df["ret1"] = df["ret"].shift(1)
    df["ret2"] = df["ret"].shift(2)
    df["ret3"] = df["ret"].shift(3)
    df["target"] = np.sign(df["ret"].shift(-1)).fillna(0)
    df.dropna(inplace=True)

    X = df[["ret1", "ret2", "ret3"]].values
    y = df["target"].values

    # Train simple logistic regression on Up(>0) vs Down(<=0)
    y_bin = (y > 0).astype(int)
    model = SkPipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X, y_bin)

    x_today = np.array([[df["ret"].iloc[-1], df["ret"].iloc[-2], df["ret"].iloc[-3]]])
    proba = float(model.predict_proba(x_today)[0, 1])

    # Map to Up/Down/Flat with a small band
    if proba > 0.55:
        direction = "Up"
    elif proba < 0.45:
        direction = "Down"
    else:
        direction = "Flat"
    return direction, proba


def plot_sentiment_vs_yield(yields_df: pd.DataFrame, mood: float, out_html: str) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yields_df["Date"], y=yields_df["TNX"], name="10Y Yield (^TNX)"))
    # Show today mood as a horizontal marker at last date
    if not yields_df.empty:
        last_date = yields_df["Date"].iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last_date], y=[yields_df["TNX"].iloc[-1]],
                mode="markers+text",
                text=[f"Mood: {mood:+.2f}"],
                name="Market Mood",
            )
        )
    fig.update_layout(title="News Sentiment vs 10Y Yield", xaxis_title="Date", yaxis_title="^TNX")
    fig.write_html(out_html)
    return out_html


def gemma_commentary(headlines: List[str], direction: str, proba: float, mood: float) -> str:
    if PROVIDER.lower() != "ollama":
        return "(LLM commentary requires PROVIDER=ollama with a local Gemma model)"
    from langchain_community.chat_models import ChatOllama
    chat = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.2)
    system = (
        "You are a macro strategist. Explain briefly why Treasury yields might move given \n"
        "today's headlines and sentiment. Reference the Fed, POTUS, and commodities if relevant."
    )
    user = (
        f"Predicted direction: {direction} (p={proba:.2f}), Mood={mood:+.2f}.\n"
        f"Top headlines:\n- " + "\n- ".join(headlines[:8])
    )
    msg = chat.invoke([("system", system), ("user", user)])
    return getattr(msg, "content", str(msg))


def main():
    ap = argparse.ArgumentParser(description="Macro Predictive Agent")
    ap.add_argument("--sources", default="fed,reuters", help="Comma-separated sources: fed,reuters")
    ap.add_argument("--days", type=int, default=120, help="Yield lookback days")
    ap.add_argument("--out", default="./outputs/macro_sentiment.html", help="Plot output HTML path")
    args = ap.parse_args()

    srcs = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    pairs = fetch_headlines(srcs, limit=30)
    titles = [t for t, _ in pairs]
    scores = score_sentiment(titles) if titles else []
    mood = market_mood_index(scores)

    yields_df = load_yields(days=args.days)
    direction, proba = train_predict_direction(yields_df, mood)

    out_path = plot_sentiment_vs_yield(yields_df, mood, args.out)

    commentary = gemma_commentary(titles, direction, proba, mood)

    print(f"Market Mood Index: {mood:+.3f}")
    print(f"Predicted next-day 10Y direction: {direction} (p={proba:.2f})")
    print(f"Plot saved to: {out_path}")
    print("\nGemma commentary:\n" + commentary)


if __name__ == "__main__":
    main()


