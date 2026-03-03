"""
Intern Performance Model Trainer
==================================
Trains Random Forest and XGBoost regression models, evaluates them,
generates comparison plots, and saves the best model.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings("ignore")

# -- Paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "intern_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

FEATURE_COLS = [
    "weeks_in_program",
    "task_completion_rate",
    "avg_feedback_rating",
    "attendance_rate",
    "tasks_completed",
    "avg_completion_time_hours",
    "mentor_sessions_attended",
]
TARGET_COL = "performance_score"


# -- Plotting Style ----------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "font.family": "sans-serif",
    "font.size": 11,
})


def load_data():
    """Load and validate the intern dataset."""
    df = pd.read_csv(DATA_FILE)
    print(f"[OK] Loaded {len(df)} records from {DATA_FILE}")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Target:   {TARGET_COL}")
    return df


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train both models and return results dict."""
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"  Training: {name}")
        print(f"{'-' * 50}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            model, np.vstack([X_train, X_test]),
            np.concatenate([y_train, y_test]),
            cv=5, scoring="r2"
        )

        results[name] = {
            "model": model,
            "predictions": y_pred,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std": round(cv_scores.std(), 4),
            "feature_importance": dict(zip(
                FEATURE_COLS,
                [round(float(x), 4) for x in model.feature_importances_]
            )),
        }

        print(f"  MAE:     {mae:.3f}")
        print(f"  RMSE:    {rmse:.3f}")
        print(f"  R2:      {r2:.4f}")
        print(f"  CV R2:   {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return results


def generate_plots(results, y_test):
    """Generate comparison and diagnostic plots."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    colors = {"Random Forest": "#38bdf8", "XGBoost": "#a78bfa"}

    # -- 1. Feature Importance Comparison -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (name, res) in zip(axes, results.items()):
        imp = res["feature_importance"]
        sorted_features = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_features)
        short_names = [f.replace("_", "\n") for f in features]
        bars = ax.barh(range(len(features)), values, color=colors[name], alpha=0.85, height=0.6)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(f"{name} - Feature Importance", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, color="#94a3b8")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved feature_importance.png")

    # -- 2. Actual vs Predicted ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, res) in zip(axes, results.items()):
        ax.scatter(y_test, res["predictions"], alpha=0.5, s=30, color=colors[name], edgecolors="white", linewidth=0.3)
        lims = [min(y_test.min(), res["predictions"].min()) - 2,
                max(y_test.max(), res["predictions"].max()) + 2]
        ax.plot(lims, lims, "--", color="#f87171", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Performance Score")
        ax.set_ylabel("Predicted Performance Score")
        ax.set_title(f"{name} - Actual vs Predicted\nR2 = {res['r2']:.4f}", fontsize=12, fontweight="bold")
        ax.legend(framealpha=0.3)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved actual_vs_predicted.png")

    # -- 3. Residuals Distribution ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, res) in zip(axes, results.items()):
        residuals = y_test - res["predictions"]
        ax.hist(residuals, bins=30, color=colors[name], alpha=0.7, edgecolor="#1e293b")
        ax.axvline(0, color="#f87171", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} - Residual Distribution", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "residuals.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved residuals.png")

    # -- 4. Model Metrics Comparison Bar Chart -------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_names = ["MAE", "RMSE", "R2", "CV R2"]
    x = np.arange(len(metrics_names))
    width = 0.3
    for i, (name, res) in enumerate(results.items()):
        vals = [res["mae"], res["rmse"], res["r2"], res["cv_r2_mean"]]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[name], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=9, color="#e2e8f0")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_names)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved metrics_comparison.png")


def classify_interns(df, model, feature_cols):
    """Classify all interns into performance categories."""
    X_all = df[feature_cols].values
    df = df.copy()
    df["predicted_score"] = model.predict(X_all).round(1)

    def categorize(score):
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Needs Support"
        else:
            return "At Risk"

    df["predicted_category"] = df["predicted_score"].apply(categorize)
    return df


def save_dashboard_data(results, df_classified):
    """Export data for the HTML dashboard."""
    dashboard = {
        "models": {},
        "features": FEATURE_COLS,
        "interns": [],
    }

    for name, res in results.items():
        dashboard["models"][name] = {
            "mae": res["mae"],
            "rmse": res["rmse"],
            "r2": res["r2"],
            "cv_r2_mean": res["cv_r2_mean"],
            "cv_r2_std": res["cv_r2_std"],
            "feature_importance": res["feature_importance"],
        }

    # Top 50 interns for the dashboard table
    for _, row in df_classified.head(50).iterrows():
        dashboard["interns"].append({
            "id": row["intern_id"],
            "name": row["name"],
            "department": row["department"],
            "task_completion_rate": float(row["task_completion_rate"]),
            "avg_feedback_rating": float(row["avg_feedback_rating"]),
            "attendance_rate": float(row["attendance_rate"]),
            "actual_score": float(row["performance_score"]),
            "predicted_score": float(row["predicted_score"]),
            "category": row["predicted_category"],
        })

    # Category distribution for all interns
    cat_counts = df_classified["predicted_category"].value_counts().to_dict()
    dashboard["category_distribution"] = cat_counts

    output_path = os.path.join(OUTPUTS_DIR, "dashboard_data.json")
    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"[OK] Saved dashboard_data.json")
    return output_path


def main():
    print("=" * 60)
    print("  Intern Performance Model Trainer")
    print("=" * 60)

    # Load data
    df = load_data()
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # Train & evaluate
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Determine best model
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = results[best_name]["model"]
    print(f"\n{'=' * 60}")
    print(f"  [BEST] Best Model: {best_name} (R2 = {results[best_name]['r2']:.4f})")
    print(f"{'=' * 60}")

    # Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"[OK] Saved best model to {model_path}")

    # Also save the other model
    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(res["model"], os.path.join(MODELS_DIR, f"{safe_name}.pkl"))

    # Generate plots
    print(f"\n-- Generating Plots {'-' * 39}")
    generate_plots(results, y_test)

    # Classify all interns
    print(f"\n-- Intern Classification {'-' * 35}")
    df_classified = classify_interns(df, best_model, FEATURE_COLS)
    cat_dist = df_classified["predicted_category"].value_counts().sort_index()
    for cat, count in cat_dist.items():
        print(f"  {cat}: {count}")

    # Save dashboard data
    save_dashboard_data(results, df_classified)

    print(f"\n{'=' * 60}")
    print(f"  [OK] All done! Check outputs/ for plots and dashboard data.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
