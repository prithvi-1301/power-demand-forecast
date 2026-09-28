import json, os
from datetime import datetime, date
from pathlib import Path
import joblib, numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
MODEL = joblib.load(ROOT/"models"/"demand_model.pkl")
with open(ROOT/"models"/"metrics.json") as f: METRICS=json.load(f)
with open(ROOT/"models"/"feature_importance.json") as f: FEATURE_IMPORTANCE=json.load(f)
with open(ROOT/"models"/"dataset_summary.json") as f: SUMMARY=json.load(f)
FEATURE_COLUMNS=["hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos","is_weekend","day_of_year","trend_days"]
DATASET_START=datetime.strptime(SUMMARY["dataset_start"],"%Y-%m-%d")

def build_features(target_date: date, hour: int):
    dt=datetime(target_date.year,target_date.month,target_date.day,hour)
    dow=dt.weekday()
    row={"hour_sin":np.sin(2*np.pi*hour/24),"hour_cos":np.cos(2*np.pi*hour/24),"month_sin":np.sin(2*np.pi*dt.month/12),"month_cos":np.cos(2*np.pi*dt.month/12),"dow_sin":np.sin(2*np.pi*dow/7),"dow_cos":np.cos(2*np.pi*dow/7),"is_weekend":int(dow>=5),"day_of_year":dt.timetuple().tm_yday,"trend_days":(dt-DATASET_START).total_seconds()/86400}
    return pd.DataFrame([[row[c] for c in FEATURE_COLUMNS]],columns=FEATURE_COLUMNS)

def predict_for(target_date, hour): return float(MODEL.predict(build_features(target_date,hour))[0])

@app.route('/')
def index(): return render_template('index.html')
@app.route('/api/health')
def health(): return jsonify({"status":"ok","model":"Random Forest Regressor"})
@app.route('/api/summary')
def summary(): return jsonify({"dataset":SUMMARY,"metrics":METRICS,"feature_importance":FEATURE_IMPORTANCE})
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        payload=request.get_json(force=True)
        target_date=datetime.strptime(payload['date'],'%Y-%m-%d').date()
        hour=int(payload['hour'])
        if not 0<=hour<=23: raise ValueError('hour must be between 0 and 23')
        value=predict_for(target_date,hour)
        name=target_date.strftime('%A')
        return jsonify({"predicted_mw":round(value,1),"date":target_date.isoformat(),"hour":hour,"day_of_week":name,"is_weekend":name in ('Saturday','Sunday')})
    except (KeyError,ValueError,TypeError) as e:
        return jsonify({"error":f"Invalid input: {e}"}),400

@app.route('/api/profile', methods=['POST'])
def profile():
    try:
        payload=request.get_json(force=True); target_date=datetime.strptime(payload['date'],'%Y-%m-%d').date()
        points=[]
        for hour in range(24): points.append({"hour":hour,"predicted_mw":round(predict_for(target_date,hour),1)})
        return jsonify({"date":target_date.isoformat(),"points":points})
    except (KeyError,ValueError,TypeError) as e: return jsonify({"error":f"Invalid input: {e}"}),400

if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)),debug=False)
