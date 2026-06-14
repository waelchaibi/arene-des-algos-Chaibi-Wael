import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

CHEMIN_MODELE = os.path.join(os.path.dirname(__file__), "modele.joblib")
NB_FEATURES = 30

app = Flask(__name__)
payload = joblib.load(CHEMIN_MODELE)
modele = payload["modele"]
scaler = payload["scaler"]
labels = payload.get("labels", {0: "maligne", 1: "benigne"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "JSON attendu avec la clé 'features'"}), 400

    features = data["features"]
    if not isinstance(features, list):
        return jsonify({"error": "'features' doit être une liste de nombres"}), 400
    if len(features) != NB_FEATURES:
        return jsonify({"error": f"'features' doit contenir {NB_FEATURES} valeurs"}), 400

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify({"error": "Toutes les features doivent être numériques"}), 400

    if np.isnan(X).any():
        return jsonify({"error": "Features invalides (NaN détecté)"}), 400

    X_scaled = scaler.transform(X)
    pred = int(modele.predict(X_scaled)[0])
    proba = float(modele.predict_proba(X_scaled)[0][pred])

    return jsonify({
        "prediction": pred,
        "proba": round(proba, 4),
        "label": labels.get(pred, str(pred)),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
