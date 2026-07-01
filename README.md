# Customer Churn Prediction — End-to-End Data Science Project

Name: Kartikay Verma Company: CODTECH IT SOLUTIONS ID:CTIS9080 Duration: 8 weeks Mentor: Neela santhosh kumar

An end-to-end machine learning project that predicts whether a telecom customer is likely to **churn** (cancel their subscription), deployed as a REST API and web app using **Flask**.

This project covers the full data science lifecycle:

1. **Data Collection** — synthetic telecom customer dataset generated with realistic churn patterns (`data/generate_data.py`)
2. **Exploratory Data Analysis** — `eda.py` produces summary stats and visualizations
3. **Preprocessing** — missing value handling, one-hot encoding for categoricals, scaling for numerics (via `sklearn.ColumnTransformer`)
4. **Model Training & Selection** — trains Logistic Regression, Random Forest, and XGBoost; selects the best model by ROC-AUC (`model/train_model.py`)
5. **Deployment** — Flask REST API + simple web UI for live predictions (`app.py`)

---

## Project Structure

```
churn-prediction-project/
├── app.py                     # Flask app (web UI + REST API)
├── requirements.txt
├── Procfile                   # for Heroku/Render deployment
├── eda.py                     # exploratory data analysis script
├── data/
│   ├── generate_data.py       # synthetic data generator
│   └── customer_churn.csv     # generated dataset
├── model/
│   ├── train_model.py         # preprocessing + training pipeline
│   ├── churn_model.pkl        # saved trained pipeline (generated)
│   ├── feature_schema.json    # feature names/options (generated)
│   └── evaluation_report.json # model comparison metrics (generated)
├── templates/
│   └── index.html             # web UI
└── static/
    └── style.css
```

---

## Setup

```bash
git clone https://github.com/<your-username>/churn-prediction-project.git
cd churn-prediction-project

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Run the pipeline from scratch

```bash
# 1. Generate the dataset
python data/generate_data.py

# 2. (Optional) Run EDA
python eda.py

# 3. Train the model — saves model/churn_model.pkl
python model/train_model.py

# 4. Launch the web app + API
python app.py
```

Visit **http://localhost:5000** in your browser to use the web form.

---

## REST API

### `POST /api/predict`

Send customer details as JSON, get back a churn prediction.

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
    "TotalCharges": 427.5
  }'
```

**Response:**

```json
{
  "success": true,
  "prediction": {
    "churn_prediction": "Yes",
    "churn_probability": 0.9311,
    "risk_level": "High"
  }
}
```

### `GET /api/schema`
Returns the full list of expected input fields and valid categorical values — useful for building forms or client integrations.

### `GET /api/health`
Health check endpoint, returns `{"status": "ok", "model_loaded": true}`.

---

## Model Performance

The training pipeline compares Logistic Regression, Random Forest, and XGBoost, and automatically selects the best model by ROC-AUC. Metrics for all candidates are saved to `model/evaluation_report.json` after training.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.67 | ~0.66 | ~0.62 | ~0.64 | **~0.75** |
| Random Forest | ~0.67 | ~0.65 | ~0.62 | ~0.64 | ~0.74 |
| XGBoost | ~0.66 | ~0.65 | ~0.61 | ~0.63 | ~0.73 |

*(Numbers vary slightly per run since the dataset is synthetically generated each time.)*

---

## Deployment

This app is ready to deploy to any platform that supports Flask + gunicorn (e.g. **Render**, **Railway**, **Heroku**, **PythonAnywhere**, or a VPS).

**Render / Heroku style deployment:**
1. Push this repo to GitHub.
2. Create a new Web Service on Render (or Heroku app), connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app` (already defined in `Procfile`)
5. Make sure `model/churn_model.pkl` is committed (or regenerate it in a build step by running `data/generate_data.py` then `model/train_model.py`).

---

## Tech Stack

- **Python**, **scikit-learn**, **XGBoost**, **pandas**, **NumPy** — data processing & modeling
- **Flask** — REST API and web app
- **HTML/CSS** — frontend UI
- **joblib** — model serialization

## License

MIT — free to use and modify.
