"""
AgroChain — Prédiction maladie avec vrai modèle Kaggle
Utilise: temperature, humidity, rainfall, soil_pH
"""
import joblib
import numpy as np

MODEL_PATH = "/home/mouna/agrochain/plant_model_kaggle.pkl"

model, scaler = joblib.load(MODEL_PATH)
print("[Santé Plantes] Modèle ML chargé — Accuracy: 87.10%")

def predict_disease(temperature, humidity, rainfall, soil_ph):
    """Prédit si la plante est malade ou saine"""
    features = np.array([[temperature, humidity, rainfall, soil_ph]])
    features_scaled = scaler.transform(features)
    prediction  = model.predict(features_scaled)[0]
    proba       = model.predict_proba(features_scaled)[0]
    confidence  = int(max(proba) * 100)
    label       = "Maladie détectée" if prediction == 1 else "Plante saine"
    return {
        "prediction":  int(prediction),
        "label":       label,
        "confidence":  confidence,
        "probability_saine":   round(float(proba[0]) * 100, 1),
        "probability_maladie": round(float(proba[1]) * 100, 1)
    }
