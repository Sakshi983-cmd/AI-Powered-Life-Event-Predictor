#APP- https://ai-powered-life-event-predictor-me2gcnrjqu93kdnso5hjav.streamlit.app/
# AI-Powered-Life-Event-Predictor
# 🧠 AI-Powered Life Event Predictor

This project predicts **Burnout Risk** (next 30 days) from your daily wellness data such as **steps, sleep, calories, stress, and mood**.

## 🚀 Features
- Upload CSV or use the provided `data/sample_data.csv`.
- Cleans data & generates features (sleep debt, rolling averages).
- Logistic Regression model for weekly burnout prediction.
- Shows **Burnout Risk %**, **Top 3 Reasons**, and **Suggestions**.
- Download weekly features as CSV.

## 🛠️ How to Run (Local)
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
