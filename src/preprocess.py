import pandas as pd

IDEAL_SLEEP = 8.0

def clean_and_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    numeric_cols = ['steps','sleep_hours','resting_hr','calories_in','calories_out','mood_1to5','stress_1to5']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction='both')

    df['sleep_debt'] = IDEAL_SLEEP - df['sleep_hours']
    df['calorie_balance'] = df['calories_in'] - df['calories_out']
    df['avg_sleep_7d'] = df['sleep_hours'].rolling(7, min_periods=1).mean()
    df['avg_steps_7d'] = df['steps'].rolling(7, min_periods=1).mean()
    df['avg_stress_7d'] = df['stress_1to5'].rolling(7, min_periods=1).mean()
    df['avg_mood_7d'] = df['mood_1to5'].rolling(7, min_periods=1).mean()
    df['avg_calorie_balance_7d'] = df['calorie_balance'].rolling(7, min_periods=1).mean()
    return df

def make_weekly_labels(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
    grouped = df.groupby('week')
    label = grouped.apply(lambda g: int(((g['sleep_hours'] < 5) & (g['stress_1to5'] > 3)).any()))
    weekly = grouped.agg({
        'avg_sleep_7d':'last',
        'sleep_debt':'mean',
        'avg_steps_7d':'last',
        'avg_stress_7d':'last',
        'avg_mood_7d':'last',
        'avg_calorie_balance_7d':'last',
        'resting_hr':'mean'
    }).reset_index()
    weekly = weekly.merge(label.rename('burnout_label'), left_on='week', right_index=True)
    return weekly.dropna().reset_index(drop=True)
