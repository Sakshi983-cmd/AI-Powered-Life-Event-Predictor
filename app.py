import streamlit as st
import pandas as pd
from pathlib import Path
from src.preprocess import clean_and_features, make_weekly_labels
from src.predict import predict_proba_from_daily
from joblib import load
from sklearn.linear_model import LogisticRegression
import altair as alt

st.set_page_config(page_title="AI Life Event Predictor", page_icon="🧠", layout="centered")

st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🧠 AI-Powered Burnout Risk Predictor</h1>", unsafe_allow_html=True)
st.write("Upload your daily wellness CSV (or use the sample) to estimate **next-30-days burnout risk**.")

# Load data
sample_path = Path("data/sample_data.csv")
uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No file uploaded — using demo data/sample_data.csv")
    df = pd.read_csv(sample_path)

# Features
df_feat = clean_and_features(df)
weekly = make_weekly_labels(df_feat)

st.subheader("📊 Weekly Features (last 5)")
st.dataframe(weekly.tail(5))

# Train/load model
model_path = Path("model.pkl")
if model_path.exists():
    bundle = load(model_path)
    model = bundle["model"]
    features = bundle["features"]
else:
    st.warning("No trained model found — training demo model.")
    from sklearn.model_selection import train_test_split
    feature_cols = ['avg_sleep_7d','sleep_debt','avg_steps_7d','avg_stress_7d',
                    'avg_mood_7d','avg_calorie_balance_7d','resting_hr']
    features = feature_cols
    if len(weekly) >= 6 and weekly['burnout_label'].nunique() == 2:
        X, y = weekly[feature_cols], weekly['burnout_label']
        model = LogisticRegression(max_iter=200).fit(X, y)
    else:
        model = None
        st.error("Not enough data to train a model.")

# Prediction
if model is not None and not weekly.empty:
    from src.predict import explain_top_reasons
    x = weekly[features].iloc[[-1]]
    proba = float(model.predict_proba(x)[0,1])
    reasons = explain_top_reasons(model, features, x.iloc[0], top_k=3)

    st.subheader("🔮 Prediction")
    st.progress(proba)
    st.metric("Burnout Risk", f"{proba*100:.1f}%")
    st.write("**Top 3 reasons:**")
    for r in reasons:
        st.write(f"- {r}")

    # Chart
    st.subheader("📉 Last 14 days Sleep vs Stress")
    chart = alt.Chart(df_feat.tail(14)).mark_line(point=True).encode(
        x='date:T',
        y=alt.Y('sleep_hours:Q', axis=alt.Axis(title='Sleep (hrs)')),
        color=alt.value("blue")
    ) + alt.Chart(df_feat.tail(14)).mark_line(point=True, color="red").encode(
        x='date:T',
        y=alt.Y('stress_1to5:Q', axis=alt.Axis(title='Stress (1-5)'))
    )
    st.altair_chart(chart, use_container_width=True)

    # Suggestions
    st.subheader("💡 Suggestions")
    suggestions = []
    if proba > 0.7:
        suggestions.append("😴 Sleep 7–8 hrs daily, 20 min walk, reduce screen time.")
    if (df_feat['calorie_balance'].tail(7).mean() < -500):
        suggestions.append("🍎 Eat more protein & balance meals (calorie deficit too high).")
    if (df_feat['stress_1to5'].tail(7).mean() > 3):
        suggestions.append("🧘 Try meditation, short breaks, weekend rest.")

    if suggestions:
        for s in suggestions:
            st.success("• " + s)
    else:
        st.info("✅ You're on track — keep it up!")

    # Download
    st.download_button(
        "⬇️ Download weekly features CSV",
        weekly.to_csv(index=False).encode("utf-8"),
        file_name="weekly_features.csv",
        mime="text/csv"
    )
