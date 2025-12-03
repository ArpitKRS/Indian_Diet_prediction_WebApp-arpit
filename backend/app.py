from flask import Flask, request, jsonify, render_template, redirect
import numpy as np
import pickle
from flask_cors import CORS
import os

# ---------------------------
# PATH CONFIG
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "frontend"),
    static_folder=os.path.join(BASE_DIR, "frontend"),
    static_url_path=""
)
CORS(app)

# ---------------------------
# LOAD MODEL + ENCODERS + META
# ---------------------------
model_path = os.path.join(os.path.dirname(__file__), "diet_model.pkl")
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
label_encoders = model_data["label_encoders"]    # dict: column -> LabelEncoder
target_encoder = model_data["target_encoder"]
feature_names = model_data["feature_names"]      # list in correct order

# ---------------------------
# HELPERS
# ---------------------------
def normalize_key(s: str) -> str:
    """Normalize a string to compare feature names (lowercase + alnum only)."""
    if s is None:
        return ""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def safe_label_encode(encoder, value):
    """
    Safely convert a single categorical value to its integer-encoded form.
    If value is unseen, fallback to encoder.classes_[0].
    """
    classes = getattr(encoder, "classes_", None)
    if classes is None:
        # Not a LabelEncoder (unexpected) — attempt transform and let it raise
        try:
            return int(encoder.transform([value])[0])
        except Exception:
            raise

    # Build mapping once (string keys)
    mapping = {str(c): idx for idx, c in enumerate(classes)}
    key = str(value)
    if key in mapping:
        return mapping[key]
    # fallback: use first class
    default_key = str(classes[0])
    return mapping[default_key]

# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_form")
def predict_form():
    # redirect to the static frontend page so UI is correct
    return redirect("/prediction.html")

@app.route("/result")
def result_page():
    name = request.args.get("name", "")
    state = request.args.get("state", "")
    diet = request.args.get("diet", "")

    from food_map import FOOD_MAP
    foods = FOOD_MAP.get(state, {}).get(diet, [])

    return render_template(
        "result.html",
        name=name,
        user_state=state,
        predicted_diet=diet,
        foods=foods
    )

# ---------------------------
# PREDICTION API
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON body received"}), 400

        # Extract inputs (as sent by the frontend)
        username = data.get("username", "")
        state = data.get("state", "")
        weight = float(data.get("weight", 0))
        bp_value = float(data.get("bp", 0))
        glucose = float(data.get("glucose", 0))
        heartrate = float(data.get("heartRate", 0))
        stamina = data.get("stamina", "")
        disease = data.get("disease", "")
        severity = data.get("severity", "")
        daily_calories = float(data.get("dailyCalories", 0))

        # --------------------------
        # Option A requirement: map "None" -> "Mild"
        # --------------------------
        if severity == "None":
            severity = "Mild"

        # --------------------------
        # Build raw input dict from frontend names
        # (these keys will be normalized/matched to feature_names)
        # --------------------------
        raw_inputs = {
            "Weight_kg": weight,
            "Blood_Pressure": bp_value,
            "Glucose_Level": glucose,
            "Heart_Rate": heartrate,
            "Stamina": stamina,
            "Disease": disease,
            "Severity": severity,
            "Daily_Calorie_Intake": daily_calories,

            # also include alternate keys that your frontend used previously
            "Weight": weight,
            "BP": bp_value,
            "Glucose": glucose,
            "HeartRate": heartrate,
            "Stamina_Level": stamina,
            "Severity_Level": severity,
            "DailyCalories": daily_calories,
        }

        # --------------------------
        # Normalize keys and prepare mapping for fuzzy matching
        # --------------------------
        normalized_map = {normalize_key(k): k for k in raw_inputs.keys()}

        # --------------------------
        # Build final_inputs aligned with feature_names
        # --------------------------
        final_inputs = {}
        missing_features = []
        for feat in feature_names:
            if feat in raw_inputs:
                final_inputs[feat] = raw_inputs[feat]
                continue

            # try normalized matching
            norm = normalize_key(feat)
            mapped_key = normalized_map.get(norm)
            if mapped_key is not None:
                final_inputs[feat] = raw_inputs[mapped_key]
            else:
                missing_features.append(feat)

        if missing_features:
            # helpful diagnostic for debugging feature name mismatches
            return jsonify({
                "status": "error",
                "message": "Missing feature(s) required by model",
                "missing_features": missing_features,
                "provided_keys": list(raw_inputs.keys())
            }), 400

        # --------------------------
        # LABEL ENCODING (safe)
        # For every label encoder we have, convert the final_inputs value to integer index.
        # If unseen value encountered, safe_label_encode will fallback to encoder.classes_[0].
        # --------------------------
        for col, encoder in label_encoders.items():
            # ensure column exists in final_inputs
            if col not in final_inputs:
                # try normalized match
                norm = normalize_key(col)
                # look for a key in final_inputs with same normalized form
                found = None
                for k in final_inputs.keys():
                    if normalize_key(k) == norm:
                        found = k
                        break
                if found is None:
                    return jsonify({
                        "status": "error",
                        "message": f"Label encoder expects column '{col}' but it is not present in inputs.",
                        "available_input_columns": list(final_inputs.keys())
                    }), 400
                # else use that found key
                value_to_encode = final_inputs[found]
            else:
                value_to_encode = final_inputs[col]

            # safe encode
            try:
                encoded_val = safe_label_encode(encoder, value_to_encode)
            except Exception as ex:
                return jsonify({
                    "status": "error",
                    "message": f"Failed to label-encode column '{col}': {str(ex)}",
                    "value": value_to_encode
                }), 500

            final_inputs[col] = encoded_val

        # --------------------------
        # Create input vector in model's feature order
        # --------------------------
        try:
            input_vector = np.array([[final_inputs[col] for col in feature_names]], dtype=float)
        except KeyError as ke:
            return jsonify({
                "status": "error",
                "message": "Feature ordering error, missing column",
                "detail": str(ke),
                "feature_names": feature_names,
                "final_inputs_keys": list(final_inputs.keys())
            }), 500

        # --------------------------
        # Scale & Predict
        # --------------------------
        input_scaled = scaler.transform(input_vector)
        encoded_pred = model.predict(input_scaled)[0]
        predicted_diet = target_encoder.inverse_transform([encoded_pred])[0]

        # --------------------------
        # Food suggestions
        # --------------------------
        from food_map import FOOD_MAP
        foods = FOOD_MAP.get(state, {}).get(predicted_diet, [])

        return jsonify({
            "status": "success",
            "diet": predicted_diet,
            "foods": foods
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
