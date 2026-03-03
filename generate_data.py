"""
Intern Performance Data Generator
==================================
Generates synthetic intern performance data with realistic correlations
between features and the target performance score.
"""

import os
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_INTERNS = 500
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "intern_data.csv")

# Intern name pools for realistic records
FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
    "Cameron", "Dakota", "Jamie", "Skyler", "Drew", "Harper", "Reese", "Sage",
    "Emery", "Finley", "Rowan", "Hayden", "Parker", "Sawyer", "Blake", "Charlie",
    "Elliott", "Frankie", "Jesse", "Kerry", "Lane", "Micah", "Noel", "Pat",
    "Robin", "Sam", "Terry", "Val", "Wren", "Amara", "Ravi", "Priya",
    "Yuki", "Liam", "Sophia", "Ethan", "Olivia", "Noah", "Ava", "Lucas",
    "Mia", "Mason"
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts"
]
DEPARTMENTS = [
    "Engineering", "Data Science", "Product", "Design", "Marketing",
    "Finance", "Operations", "HR", "Legal", "Sales"
]


def generate_intern_data(n: int = NUM_INTERNS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic intern performance data with realistic correlations."""
    rng = np.random.default_rng(seed)

    # ── Core features ──────────────────────────────────────────────────────
    # Base aptitude (latent variable driving correlations)
    aptitude = rng.beta(a=2.5, b=2.5, size=n)  # centered around 0.5

    # Task completion rate (%) — higher aptitude → higher completion
    task_completion_rate = np.clip(
        aptitude * 80 + rng.normal(15, 8, n), 10, 100
    ).round(1)

    # Average feedback rating (1-5) — correlated with aptitude
    avg_feedback_rating = np.clip(
        aptitude * 3 + rng.normal(1.5, 0.5, n), 1.0, 5.0
    ).round(2)

    # Attendance rate (%) — moderately correlated
    attendance_rate = np.clip(
        aptitude * 40 + rng.normal(55, 10, n), 40, 100
    ).round(1)

    # Tasks completed (count) — depends on weeks and aptitude
    weeks_in_program = rng.integers(4, 25, size=n)
    tasks_completed = np.clip(
        (aptitude * 4 + rng.normal(2, 1, n)) * weeks_in_program / 4,
        1, None
    ).astype(int)

    # Average completion time (hours) — lower is better, inversely correlated
    avg_completion_time_hours = np.clip(
        (1 - aptitude) * 20 + rng.normal(8, 4, n), 1, 40
    ).round(1)

    # Mentor sessions attended — correlated with engagement
    mentor_sessions_attended = np.clip(
        aptitude * 8 + rng.normal(3, 2, n), 0, 20
    ).astype(int)

    # ── Target variable ────────────────────────────────────────────────────
    # Performance score = weighted combination + noise
    performance_score = np.clip(
        0.30 * task_completion_rate
        + 8.0 * avg_feedback_rating
        + 0.15 * attendance_rate
        + 0.4 * tasks_completed
        - 0.5 * avg_completion_time_hours
        + 1.0 * mentor_sessions_attended
        + rng.normal(0, 4, n),
        0, 100
    ).round(1)

    # ── Metadata ───────────────────────────────────────────────────────────
    intern_ids = [f"INT-{i+1001:04d}" for i in range(n)]
    names = [
        f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n)
    ]
    departments = rng.choice(DEPARTMENTS, size=n).tolist()

    # ── Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "intern_id": intern_ids,
        "name": names,
        "department": departments,
        "weeks_in_program": weeks_in_program,
        "task_completion_rate": task_completion_rate,
        "avg_feedback_rating": avg_feedback_rating,
        "attendance_rate": attendance_rate,
        "tasks_completed": tasks_completed,
        "avg_completion_time_hours": avg_completion_time_hours,
        "mentor_sessions_attended": mentor_sessions_attended,
        "performance_score": performance_score,
    })

    return df


def main():
    print("=" * 60)
    print("  Intern Performance Data Generator")
    print("=" * 60)

    df = generate_intern_data()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n[OK] Generated {len(df)} intern records")
    print(f"[OK] Saved to: {OUTPUT_FILE}")
    print(f"\n-- Dataset Summary {'-' * 40}")
    print(df.describe().round(2).to_string())
    print(f"\n-- First 5 Records {'-' * 40}")
    print(df.head().to_string())
    print(f"\n-- Performance Distribution {'-' * 31}")
    bins = [0, 40, 60, 80, 100]
    labels = ["At Risk (0-40)", "Needs Support (40-60)", "Good (60-80)", "Excellent (80-100)"]
    df["category"] = pd.cut(df["performance_score"], bins=bins, labels=labels, include_lowest=True)
    print(df["category"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
