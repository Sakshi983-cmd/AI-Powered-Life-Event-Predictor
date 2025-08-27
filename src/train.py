import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from pathlib import Path
from src.preprocess import clean_and_features, make_weekly_labels

DATA_PATH = Path("data/sample_data.csv")
MODEL_PATH = Path("model.pkl")

def main():
    df = pd.read_csv(DATA_PATH)
    df = clean_and_features(df)
    weekly = make_weekly_labels(df)
    feature_cols = ['avg_sleep_7d','sleep_debt','avg_steps_7d','avg_stress_7d',
                    'avg_mood_7d','avg_calorie_balance_7d','resting_hr']
    X, y = weekly[feature_cols], weekly['burnout_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200).fit(X_train, y_train)
    print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
    dump({'model': clf, 'features': feature_cols}, MODEL_PATH)

if __name__ == "__main__":
    main()
