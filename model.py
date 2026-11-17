"""Training and prediction logic for the Obesity Level Predictor.

Mirrors the pipeline in "ML Project.ipynb":
label-encode categorical columns -> StandardScaler -> 80/20 split -> Random Forest.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "ObesityDataSet_raw_and_data_sinthetic.csv"
TARGET = "NObeyesdad"

# Order used to show results from lightest to heaviest category.
LABEL_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def train(df: pd.DataFrame) -> dict:
    """Fit encoders, scaler and Random Forest. Returns everything the app needs."""
    feature_cols = [c for c in df.columns if c != TARGET]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    encoders = {c: LabelEncoder().fit(df[c]) for c in cat_cols}
    target_encoder = LabelEncoder().fit(df[TARGET])

    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = encoders[c].transform(X[c])
    y = target_encoder.transform(df[TARGET])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()

    return {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "target_encoder": target_encoder,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "accuracy": accuracy,
        "importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict(bundle: dict, values: dict) -> tuple[str, pd.Series]:
    """Predict the category for one person. `values` maps feature name -> raw value."""
    row = pd.DataFrame([values])[bundle["feature_cols"]].copy()
    for c in bundle["cat_cols"]:
        row[c] = bundle["encoders"][c].transform(row[c])
    scaled = bundle["scaler"].transform(row.values.astype(float))
    proba = bundle["model"].predict_proba(scaled)[0]
    classes = bundle["target_encoder"].inverse_transform(bundle["model"].classes_)
    probs = pd.Series(proba, index=classes)
    probs = probs.reindex([c for c in LABEL_ORDER if c in probs.index]).fillna(0.0)
    return probs.idxmax(), probs
