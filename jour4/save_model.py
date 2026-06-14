import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def sauvegarder_modele(modele, scaler, chemin="modele.joblib", extra=None):
    payload = {"modele": modele, "scaler": scaler}
    if extra:
        payload.update(extra)
    joblib.dump(payload, chemin)
    print(f"Modèle sauvegardé : {chemin}")


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    modele = RandomForestClassifier(n_estimators=200, random_state=42)
    modele.fit(X_train_s, y_train)

    sauvegarder_modele(
        modele,
        scaler,
        chemin="modele.joblib",
        extra={
            "feature_names": list(data.feature_names),
            "feature_mins": X_train.min(axis=0).tolist(),
            "feature_maxs": X_train.max(axis=0).tolist(),
            "labels": {0: "maligne", 1: "benigne"},
        },
    )
