import os
import time
import math
import requests
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv

# load .env if present (for local dev)
load_dotenv()

# CONFIG
COINGECKO_API_KEY = os.getenv("CG-ZvfhjFxxNae1TMGX4arCG3om")  # set this in environment
DEFAULT_VS_CURRENCY = "usd"
DEFAULT_DAYS = 30  # history window in days to train on
HORIZON_HOURS = 1  # horizon to predict (1 hour ahead default)
MODEL_REFRESH_EVERY_SEC = 60 * 60  # refresh model hourly (simple periodic retrain)

if not COINGECKO_API_KEY:
    print("Warning: COINGECKO_API_KEY not set. Set it in env variable COINGECKO_API_KEY for pro endpoints.")

app = FastAPI(title="IA-Cripto-Analyze API")

# Simple request model
class AnalyzeRequest(BaseModel):
    coin_id: str = "bitcoin"   # CoinGecko coin id: "bitcoin", "ethereum", "solana", etc.
    vs_currency: str = DEFAULT_VS_CURRENCY
    horizon_hours: Optional[int] = HORIZON_HOURS

# Store model and metadata globally
MODEL_STORE = {
    "pipeline": None,
    "feature_names": [],
    "last_trained": None,
    "train_stats": {}
}

# ---------- Utility: fetch market data from CoinGecko ----------
def fetch_market_chart(coin_id: str, vs_currency: str = "usd", days: int = 30, interval: str = "hourly"):
    """
    Uses CoinGecko /coins/{id}/market_chart endpoint.
    If you have a Pro key, it will be passed in header 'x-cg-pro-api-key'.
    Returns DataFrame with columns: time(ms), price
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": interval}
    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-pro-api-key"] = COINGECKO_API_KEY

    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"CoinGecko fetch failed: {r.status_code} {r.text}")
    data = r.json()
    # prices: [ [timestamp, price], ... ]
    prices = data.get("prices", [])
    if not prices:
        raise HTTPException(status_code=502, detail="No price data from CoinGecko")
    df = pd.DataFrame(prices, columns=["time", "price"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time").sort_index()
    return df

# ---------- Feature engineering ----------
def compute_features(df: pd.DataFrame):
    """
    df must have index=time and column price
    Returns dataframe with engineered features for modelling
    """
    df = df.copy()
    df["logret_1"] = np.log(df["price"]).diff()  # log return 1 period
    df["ret_1"] = df["price"].pct_change()
    # rolling windows (in hours if data hourly)
    windows = [3, 6, 12, 24]  # hours
    for w in windows:
        df[f"sma_{w}"] = df["price"].rolling(window=w, min_periods=1).mean()
        df[f"std_{w}"] = df["logret_1"].rolling(window=w, min_periods=1).std()
        df[f"max_{w}"] = df["price"].rolling(window=w, min_periods=1).max()
        df[f"min_{w}"] = df["price"].rolling(window=w, min_periods=1).min()
    # momentum
    df["mom_12"] = df["price"] / df["price"].shift(12) - 1
    # simple RSI implementation on close log returns
    def rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=period, min_periods=1).mean()
        ma_down = down.rolling(window=period, min_periods=1).mean()
        rs = ma_up / (ma_down + 1e-9)
        return 100 - (100 / (1 + rs))
    df["rsi_14"] = rsi(df["price"], period=14)
    df = df.dropna().copy()
    return df

# ---------- Prepare training set, labels ----------
def prepare_training(df: pd.DataFrame, horizon_hours: int = 1):
    """
    Creates features X and binary target y:
    y = 1 if price at t+horizon > price at t (positive return), else 0
    """
    df = df.copy()
    df["future_price"] = df["price"].shift(-horizon_hours)
    df["future_ret"] = df["future_price"] / df["price"] - 1
    df["target"] = (df["future_ret"] > 0).astype(int)
    df = df.dropna().copy()
    feature_cols = [c for c in df.columns if c not in ["future_price", "future_ret", "target"]]
    X = df[feature_cols]
    y = df["target"]
    return X, y, feature_cols, df

# ---------- Train or refresh model ----------
def train_model_for_coin(coin_id: str, vs_currency: str = "usd", days: int = DEFAULT_DAYS, horizon_hours: int = HORIZON_HOURS):
    try:
        raw = fetch_market_chart(coin_id=coin_id, vs_currency=vs_currency, days=days, interval="hourly")
        feat = compute_features(raw)
        X, y, feat_cols, full = prepare_training(feat, horizon_hours=horizon_hours)

        if len(X) < 50:
            raise ValueError("Not enough data to train model. Need more history or larger days.")

        # train-test split (time-series style: no shuffle)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ])
        pipeline.fit(X_train, y_train)
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)

        MODEL_STORE["pipeline"] = pipeline
        MODEL_STORE["feature_names"] = feat_cols
        MODEL_STORE["last_trained"] = time.time()
        MODEL_STORE["train_stats"] = {
            "train_score": float(train_score),
            "test_score": float(test_score),
            "n_rows": int(len(X)),
            "coin_id": coin_id,
            "horizon_hours": horizon_hours
        }
        # Also compute some win/loss stats for simple Kelly fraction estimate
        full["pred_proba"] = pipeline.predict_proba(X)[:, 1]
        avg_pos_ret = full.loc[full["target"] == 1, "future_ret"].mean() or 0.0
        avg_neg_ret = -full.loc[full["target"] == 0, "future_ret"].mean() or 1e-6
        MODEL_STORE["train_stats"].update({
            "avg_pos_ret": float(avg_pos_ret),
            "avg_neg_ret": float(avg_neg_ret)
        })
        return MODEL_STORE["train_stats"]
    except Exception as e:
        traceback.print_exc()
        raise

# ---------- Helper: Kelly fraction ----------
def kelly_fraction(p: float, avg_win: float, avg_loss: float, cap: float = 0.2):
    """
    p: probability of win (0..1)
    avg_win: average fractional gain when win (e.g., 0.02 for +2%)
    avg_loss: average fractional loss when losing (positive number, e.g., 0.015 for -1.5%)
    returns fraction f of bankroll (capped)
    Kelly (edge b = avg_win/avg_loss):
      f = (p*(b+1) - 1)/b
    We guard for extremes and cap returned value to [0,cap]
    """
    if avg_loss <= 0:
        return min(cap, 0.01)
    b = (avg_win / avg_loss) if avg_loss>0 else 1e-6
    denom = b if b != 0 else 1e-6
    f = (p * (b + 1) - 1) / denom
    if math.isnan(f) or f <= 0:
        return 0.0
    # smooth and cap
    f = max(0.0, f)
    return min(f, cap)

# ---------- Analyze endpoint ----------
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Example POST body:
    { "coin_id": "bitcoin", "vs_currency": "usd", "horizon_hours": 1 }
    Returns: decision, confidence, explanation, size_pct, stop_loss, take_profit, price
    """
    coin_id = req.coin_id
    vs = req.vs_currency
    horizon = req.horizon_hours or HORIZON_HOURS

    # retrain model if not present or if horizon changed
    try:
        need_train = MODEL_STORE["pipeline"] is None or MODEL_STORE["train_stats"].get("coin_id") != coin_id or MODEL_STORE["train_stats"].get("horizon_hours") != horizon
        if need_train:
            train_model_for_coin(coin_id=coin_id, vs_currency=vs, days=DEFAULT_DAYS, horizon_hours=horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    pipeline = MODEL_STORE["pipeline"]
    feat_names = MODEL_STORE["feature_names"]

    # fetch fresh most recent data (a bit more history to compute features)
    try:
        raw = fetch_market_chart(coin_id=coin_id, vs_currency=vs, days=7, interval="hourly")
        feat = compute_features(raw)
        X_live = feat[feat_names].iloc[-1:]
        price_now = float(feat["price"].iloc[-1])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch recent data: {e}")

    # predict
    proba = pipeline.predict_proba(X_live)[0][1]  # prob of price up
    pred_label = "UP" if proba >= 0.5 else "DOWN"
    confidence = float(proba if proba>=0.5 else 1-proba)

    # feature importance explanation (top 3 features)
    rf = pipeline.named_steps["rf"]
    importances = rf.feature_importances_
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    top_feats = feat_imp[:3]
    top_explanations = []
    for f, imp in top_feats:
        val = float(X_live[f].iloc[0])
        top_explanations.append({"feature": f, "importance": float(imp), "value": round(val, 6)})

    # estimate kelly fraction using training stats (fallback to heuristic)
    avg_win = MODEL_STORE["train_stats"].get("avg_pos_ret", 0.01)
    avg_loss = MODEL_STORE["train_stats"].get("avg_neg_ret", 0.01)
    p = proba
    kelly = kelly_fraction(p, avg_win, avg_loss, cap=0.15)  # cap 15% max suggested
    # safety caps and min allocation
    suggested_pct = round(float(kelly), 4)

    # compute stop / take profit heuristics using volatility
    vol_24 = float(feat["std_24"].iloc[-1]) if "std_24" in feat.columns else 0.01
    # stop: a few stds below/above price
    if pred_label == "UP":
        stop_loss = price_now * (1 - max(0.005, vol_24 * 2))
        take_profit = price_now * (1 + max(0.01, vol_24 * 3))
    else:
        stop_loss = price_now * (1 + max(0.005, vol_24 * 2))  # short: stop above
        take_profit = price_now * (1 - max(0.01, vol_24 * 3))

    # Compose textual explanation
    reason_lines = [
        f"Modelo baseline (RandomForest) previsão para {horizon}h: {pred_label} com confiança {round(confidence,3)}.",
        f"Top features: " + ", ".join([f"{t['feature']} (val={t['value']}, imp={round(t['importance'],3)})" for t in top_explanations]) + ".",
        f"Histórico: treino (rows={MODEL_STORE['train_stats'].get('n_rows')}, test_score={round(MODEL_STORE['train_stats'].get('test_score',0),3)})",
        f"Alocação sugerida (Kelly-based, cap 15%): {round(suggested_pct*100,3)}% do portfólio."
    ]
    explanation = " ".join(reason_lines)

    response = {
        "coin_id": coin_id,
        "price": round(price_now, 8),
        "prediction": pred_label,
        "probability_up": round(proba, 4),
        "confidence": round(confidence, 4),
        "suggested_allocation_pct": round(suggested_pct * 100, 3),
        "stop_loss": round(stop_loss, 8),
        "take_profit": round(take_profit, 8),
        "top_features": top_explanations,
        "explanation": explanation,
        "train_stats": MODEL_STORE["train_stats"]
    }
    return response

# Simple health endpoint
@app.get("/health")
def health():
    return {"ok": True, "model_trained": MODEL_STORE["pipeline"] is not None, "last_trained": MODEL_STORE["last_trained"]}

# Optionally, retrain on demand
@app.post("/retrain")
def retrain(coin_id: str = "bitcoin", vs_currency: str = DEFAULT_VS_CURRENCY, days: int = DEFAULT_DAYS, horizon_hours: int = HORIZON_HOURS):
    stats = train_model_for_coin(coin_id=coin_id, vs_currency=vs_currency, days=days, horizon_hours=horizon_hours)
    return {"status": "trained", "stats": stats}


# To run:
# uvicorn main:app --host 0.0.0.0 --port 8000
