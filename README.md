# Power Demand Forecast

A deploy-ready Flask website for predicting hourly electricity demand in MW from a supplied Excel dataset.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/train_model.py
python app.py
```

Open http://127.0.0.1:5000

## Deploy on Render

1. Create a GitHub repository and upload this entire project folder.
2. In Render, choose **New → Web Service** and connect the GitHub repo.
3. Use Python 3, build command `pip install -r requirements.txt && python src/train_model.py`, and start command `gunicorn app:app`.
4. Choose the Free plan for a demo/class project and deploy.
5. Render will provide a public `https://...onrender.com` URL.

The included `render.yaml` can also be used as the service blueprint.

## Important dataset note

The supplied source contains 8,760 hourly observations for 2017 and 2,160 observations for Jan–Mar 2026. It is not a continuous 2017–2026 time series. The model therefore learns from the observations that actually exist in the source file.
