import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset.xlsx"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
FEATURE_COLUMNS = ["hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos","is_weekend","day_of_year","trend_days"]

def load_data():
    raw = pd.read_excel(DATA, sheet_name="Yearly Demand Profile")
    raw = raw.rename(columns={raw.columns[0]:"Year", raw.columns[1]:"Date", raw.columns[2]:"Demand_MW"})
    raw = raw[pd.to_numeric(raw["Year"], errors="coerce").notna()].copy()
    raw["Year"] = raw["Year"].astype(int)
    raw["Demand_MW"] = pd.to_numeric(raw["Demand_MW"], errors="coerce")
    raw = raw.dropna(subset=["Date","Demand_MW"])
    # The source has dates without a year; combine with the Year column.
    raw["timestamp"] = pd.to_datetime(raw["Year"].astype(str) + "-" + raw["Date"].astype(str), format="%Y-%d-%b %I%p", errors="coerce")
    # Fallback parser for strings such as 01-Jan 1pm.
    bad = raw["timestamp"].isna()
    if bad.any():
        raw.loc[bad, "timestamp"] = pd.to_datetime(raw.loc[bad, "Date"].astype(str) + " " + raw.loc[bad,"Year"].astype(str), format="%d-%b %Y", errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
    return raw

def features(df):
    dt = df["timestamp"]
    dow = dt.dt.weekday
    return pd.DataFrame({
        "hour_sin": np.sin(2*np.pi*dt.dt.hour/24),
        "hour_cos": np.cos(2*np.pi*dt.dt.hour/24),
        "month_sin": np.sin(2*np.pi*dt.dt.month/12),
        "month_cos": np.cos(2*np.pi*dt.dt.month/12),
        "dow_sin": np.sin(2*np.pi*dow/7),
        "dow_cos": np.cos(2*np.pi*dow/7),
        "is_weekend": (dow >= 5).astype(int),
        "day_of_year": dt.dt.dayofyear,
        "trend_days": (dt - dt.min()).dt.total_seconds()/86400,
    })

def main():
    df = load_data()
    X = features(df)
    y = df["Demand_MW"].astype(float)
    # Chronological holdout: last 20% of available observations.
    cut = int(len(df)*0.8)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, max_features=1.0, min_samples_leaf=2)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "model": "Random Forest Regressor",
        "test_observations": int(len(y_test)),
        "mae_mw": round(float(mean_absolute_error(y_test,pred)),2),
        "rmse_mw": round(float(np.sqrt(mean_squared_error(y_test,pred))),2),
        "r2": round(float(r2_score(y_test,pred)),4),
    }
    # Refit on all observed data for deployment.
    model.fit(X, y)
    joblib.dump(model, MODELS/"demand_model.pkl", compress=3)
    with open(MODELS/"metrics.json","w") as f: json.dump(metrics,f,indent=2)
    importance = {k: round(float(v),4) for k,v in zip(FEATURE_COLUMNS, model.feature_importances_)}
    with open(MODELS/"feature_importance.json","w") as f: json.dump(importance,f,indent=2)
    summary = {
        "observations": int(len(df)),
        "dataset_start": df.timestamp.min().strftime("%Y-%m-%d"),
        "dataset_end": df.timestamp.max().strftime("%Y-%m-%d"),
        "years": sorted(df.Year.unique().tolist()),
        "min_demand_mw": round(float(y.min()),2),
        "max_demand_mw": round(float(y.max()),2),
    }
    with open(MODELS/"dataset_summary.json","w") as f: json.dump(summary,f,indent=2)
    print(json.dumps(metrics, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
