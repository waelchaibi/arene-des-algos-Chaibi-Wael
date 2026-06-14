# Jour 4 — Évaluation rigoureuse et mise en production

Dataset : **Breast Cancer** (`load_breast_cancer`, 569 patients, 30 mesures, binaire).

## Plan de l'après-midi

| Phase | Objectif | Fichier |
|-------|----------|---------|
| 0 | Mise en route, imports | `evaluation.ipynb` |
| 1 | Split train / validation / test (stratify) | `evaluation.ipynb` |
| 2 | Bootstrap et stabilité du score | `evaluation.ipynb` |
| 3 | Validation croisée k-fold | `evaluation.ipynb` |
| 4 | Métriques métier (recall, coût FN/FP) | `evaluation.ipynb` |
| 5 | Sérialisation joblib + API Flask | `evaluation.ipynb`, `save_model.py`, `api.py` |
| 6 | WebApp Streamlit | `app.py` |
| 7 | Arbitrage final : RF vs GB vs PMC Keras | `evaluation.ipynb` |

## Fichiers livrables

| Fichier | Rôle |
|---------|------|
| `evaluation.ipynb` | Phases 0 à 7 |
| `save_model.py` | Entraînement + sauvegarde `modele.joblib` |
| `api.py` | API Flask `/predict` |
| `app.py` | WebApp Streamlit |
| `modele.joblib` | Généré après Phase 5 |

## Contexte matin (PMC Keras)

Architecture retenue sur breast cancer :

- `Dense(16, relu)` → `Dense(8, relu)` → `Dense(1, sigmoid)`
- Optimiseur Adam, loss `binary_crossentropy`
- Données normalisées avec `StandardScaler`

## Installation TensorFlow (Windows)

Si `%pip install tensorflow` échoue avec une erreur sur `C:\Python311\Scripts\` :

```bash
python -m pip install --user tensorflow
```

Puis **redémarre le kernel Jupyter** et relance la cellule 0.

Les Phases 0-6 fonctionnent sans TensorFlow. Seule la Phase 7 (PMC Keras) en a besoin.

## Lancement

```bash
# Notebook
jupyter notebook evaluation.ipynb

# API (après entraînement et sauvegarde)
python api.py

# WebApp
streamlit run app.py
```

## Métrique métier retenue

**Dépistage médical** : rater une tumeur maligne (faux négatif) est dramatique → **recall** et **coût métier** (FN × 10, FP × 1) priment sur l'accuracy seule.
