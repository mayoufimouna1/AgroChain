"""
AgroChain — Entraînement ML sur vrai dataset Kaggle
Dataset: Plant Disease Classification
Features: temperature, humidity, rainfall, soil_pH
Target: disease_present (0=sain, 1=maladie)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 50)
print("AgroChain — Entraînement ML")
print("Dataset: Plant Disease Classification (Kaggle)")
print("=" * 50)

# ── Chargement dataset ──
df = pd.read_csv("/home/mouna/plant_data/plant_disease_dataset.csv")
print(f"\n[DATA] Dataset chargé: {len(df)} échantillons")
print(f"[DATA] Colonnes: {list(df.columns)}")
print(f"[DATA] Distribution:\n{df['disease_present'].value_counts()}")

# ── Features et target ──
X = df[["temperature", "humidity", "rainfall", "soil_pH"]].values
y = df["disease_present"].values

# ── Split train/test 80/20 ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

# ── Normalisation ──
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Entraînement Random Forest ──
print("\n[ML] Entraînement Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# ── Évaluation ──
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n[RÉSULTATS]")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"\nRapport détaillé:")
print(classification_report(y_test, y_pred,
      target_names=["Saine", "Maladie"]))

# ── Feature importance ──
features = ["temperature", "humidity", "rainfall", "soil_pH"]
importances = model.feature_importances_
print("\n[IMPORTANCE DES FEATURES]")
for f, imp in sorted(zip(features, importances),
                     key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 50)
    print(f"  {f:15s}: {imp:.3f} {bar}")

# ── Sauvegarde ──
joblib.dump((model, scaler), "/home/mouna/agrochain/plant_model_kaggle.pkl")
print(f"\n[OK] Modèle sauvegardé: plant_model_kaggle.pkl")
print("=" * 50)
