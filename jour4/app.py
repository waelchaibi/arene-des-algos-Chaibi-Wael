import os
import joblib
import numpy as np
import streamlit as st

CHEMIN_MODELE = os.path.join(os.path.dirname(__file__), "modele.joblib")


def lancer_webapp():
    payload = joblib.load(CHEMIN_MODELE)
    modele = payload["modele"]
    scaler = payload["scaler"]
    feature_names = payload.get("feature_names", [f"feature_{i}" for i in range(30)])
    feature_mins = np.array(payload.get("feature_mins", [0] * len(feature_names)))
    feature_maxs = np.array(payload.get("feature_maxs", [1] * len(feature_names)))
    labels = payload.get("labels", {0: "maligne", 1: "benigne"})

    st.title("Prédiction tumeur — Breast Cancer")
    st.caption("Modèle Random Forest — dépistage bénin / malin")

    valeurs = {}
    cols = st.columns(2)
    for i, name in enumerate(feature_names):
        with cols[i % 2]:
            default = float((feature_mins[i] + feature_maxs[i]) / 2)
            valeurs[name] = st.number_input(name, value=default, format="%.4f", key=name)

    if st.button("Prédire"):
        features = []
        hors_plage = []
        for i, name in enumerate(feature_names):
            v = valeurs[name]
            if v < feature_mins[i] or v > feature_maxs[i]:
                hors_plage.append(name)
            features.append(v)

        if hors_plage:
            st.warning(
                f"Valeurs hors plage d'entraînement : {', '.join(hors_plage[:5])}"
                + (" ..." if len(hors_plage) > 5 else "")
                + " — la prédiction peut être peu fiable."
            )

        X = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = int(modele.predict(X_scaled)[0])
        proba = float(modele.predict_proba(X_scaled)[0][pred])

        st.metric("Prédiction", labels.get(pred, str(pred)))
        st.progress(proba)
        st.write(f"Probabilité : {proba:.0%}")


if __name__ == "__main__":
    lancer_webapp()
