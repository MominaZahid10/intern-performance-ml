# 🎓 Intern Performance Prediction Model

An ML-powered system that predicts intern performance using **Random Forest** and **XGBoost** regression models, with an interactive web dashboard for visualization and live predictions.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)

## ✨ Features

- **Dual Model Training** — Random Forest & XGBoost with cross-validation
- **Interactive Dashboard** — Beautiful dark-themed UI with Chart.js visualizations
- **Live Predictions** — Enter intern metrics and get real-time ML predictions via API
- **Intern Classification** — Automatically categorize interns as Excellent / Good / Needs Support / At Risk
- **Feature Importance** — Visual comparison of which factors drive performance

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generate data → train models → export results)
python run_pipeline.py

# Start the web server
python app.py
```

Then open **http://localhost:5000** in your browser.

## 📁 Project Structure

```
├── app.py                  # Flask web server + prediction API
├── generate_data.py        # Synthetic data generator (500 interns)
├── train_model.py          # Model training, evaluation & plots
├── run_pipeline.py         # Full pipeline runner
├── dashboard.html          # Interactive web dashboard
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── data/                   # Generated intern dataset
├── models/                 # Trained model files (.pkl)
└── outputs/                # Plots & dashboard data JSON
```

## 🔌 API

**POST** `/api/predict`

```json
{
  "weeks_in_program": 12,
  "task_completion_rate": 75,
  "avg_feedback_rating": 3.5,
  "attendance_rate": 85,
  "tasks_completed": 20,
  "avg_completion_time_hours": 10,
  "mentor_sessions_attended": 5
}
```

**Response:**
```json
{
  "score": 72.3,
  "category": "Good",
  "method": "ml_model"
}
```

## 🛠 Tech Stack

- **ML**: scikit-learn, XGBoost, pandas, NumPy
- **Viz**: matplotlib, seaborn, Chart.js
- **Web**: Flask, Gunicorn
- **Deploy**: Render
# intern-performance-ml
