from flask import Flask, request, jsonify, render_template
import pandas as pd
from joblib import load
import os

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "saved_models/best_model_Random_Forest.joblib"
BEST_THRESHOLD = 0.6  # SAME as notebook
PR_AUC = 0.892
F1_SCORE = 0.8903

FEATURE_COLUMNS = [
    "mean_cons",
    "std_cons",
    "cv",
    "zero_ratio",
    "max_diff",
    "event_rate"
]

# =====================================================
# LOAD MODEL
# =====================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load(MODEL_PATH)

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Electricity Theft Detection API Running",
        "threshold": BEST_THRESHOLD
    })

@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON received"}), 400

    try:
        input_df = pd.DataFrame(
            [[float(data[col]) for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS
        )
    except:
        return jsonify({"error": "Invalid values"}), 400

    prob = model.predict_proba(input_df)[0][1]

    if prob >= BEST_THRESHOLD:
        risk_level = "Theft Detected"
        risk_color = "#e74c3c"
    else:
        risk_level = "Normal Usage"
        risk_color = "#27ae60"

    return jsonify({
        "risk_level": risk_level,
        "risk_color": risk_color,
        "probability_percent": round(prob * 100, 2),
        "threshold_percent": round(BEST_THRESHOLD * 100, 2),
        "pr_auc": 0.892,
        "f1_score": 0.8903
    })

if __name__ == "__main__":
    app.run(debug=True)