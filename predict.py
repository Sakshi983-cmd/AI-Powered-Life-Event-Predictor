import pandas as pd
from joblib import load
from pathlib import Path
from src.preprocess import clean_and_features, make_weekly_labels

MODEL_PATH = Path("model.pkl")

def explain_top_reasons(model, feature_cols, x_row, top_k=3):
    coefs = getattr(model, "coef_", None)
    if coefs is None: return []
    contrib = coefs[0] * x_row.values
    pairs = list(zip(feature_cols, contrib))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return [f"{name} ({'high' if val>0 else 'low'})" for name,val in pairs[:top_k]]

def predict_proba_from_daily(df_daily: pd.DataFrame) -> dict:
    bundle = load(MODEL_PATH)
    model, feature_cols = bundle['model'], bundle['features']
    df = clean_and_features(df_daily)
    weekly = make_weekly_labels(df)
    if weekly.empty:
        return {"error": "Not enough data"}
    x = weekly[feature_cols].iloc[[-1]]
    proba = float(model.predict_proba(x)[0,1])
    reasons = explain_top_reasons(model, feature_cols, x.iloc[0], top_k=3)
    return {"probability": proba, "reasons": reasons}
