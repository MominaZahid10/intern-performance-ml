"""
Intern Performance Model — Flask Application
==============================================
Serves the dashboard and provides a prediction API endpoint.
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory, send_file

# ── App Setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)

# ── Load Model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
model = None

FEATURE_ORDER = [
    "weeks_in_program",
    "task_completion_rate",
    "avg_feedback_rating",
    "attendance_rate",
    "tasks_completed",
    "avg_completion_time_hours",
    "mentor_sessions_attended",
]


def load_model():
    """Load the trained model from disk."""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Loaded model from {MODEL_PATH}")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH} — predictions will use fallback formula")


def categorize(score):
    """Classify a performance score into a category."""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Needs Support"
    else:
        return "At Risk"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    return send_file(os.path.join(BASE_DIR, "dashboard.html"))


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    """Serve files from the outputs directory."""
    return send_from_directory(os.path.join(BASE_DIR, "outputs"), filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict intern performance score.

    Expects JSON body with keys:
        weeks_in_program, task_completion_rate, avg_feedback_rating,
        attendance_rate, tasks_completed, avg_completion_time_hours,
        mentor_sessions_attended
    """
    try:
        data = request.get_json(force=True)

        # Build feature vector in the correct order
        features = [float(data.get(f, 0)) for f in FEATURE_ORDER]

        if model is not None:
            # Use the trained ML model
            prediction = model.predict(np.array([features]))[0]
            score = round(float(np.clip(prediction, 0, 100)), 1)
            method = "ml_model"
        else:
            # Fallback: use the same weighted formula as the data generator
            score = round(float(np.clip(
                0.30 * features[1]    # task_completion_rate
                + 8.0 * features[2]   # avg_feedback_rating
                + 0.15 * features[3]  # attendance_rate
                + 0.4 * features[4]   # tasks_completed
                - 0.5 * features[5]   # avg_completion_time_hours
                + 1.0 * features[6],  # mentor_sessions_attended
                0, 100
            )), 1)
            method = "formula_fallback"

        category = categorize(score)

        return jsonify({
            "score": score,
            "category": category,
            "method": method,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
    })


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # When run via gunicorn
    load_model()
