"""
AgroChain Plant Detector — Version Optimisée
Thread séparé pour capture + analyse IA
"""
import cv2
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import threading

MODEL_PATH = "/home/mouna/agrochain/plant_model.pkl"
SNAPSHOT_PATH = "/home/mouna/agrochain/static/snapshot.jpg"

# ═══════════════════════════════════════
# MODÈLE ML
# ═══════════════════════════════════════
def create_ml_model():
    X_train = np.array([
        [0.70, 0.05, 0.02, 150, 45],
        [0.65, 0.08, 0.03, 140, 42],
        [0.75, 0.04, 0.01, 160, 50],
        [0.68, 0.06, 0.02, 145, 44],
        [0.40, 0.30, 0.10, 120, 35],
        [0.35, 0.35, 0.12, 115, 33],
        [0.45, 0.28, 0.08, 125, 37],
        [0.38, 0.32, 0.11, 118, 34],
        [0.20, 0.15, 0.45, 100, 25],
        [0.18, 0.12, 0.50, 95,  23],
        [0.22, 0.18, 0.42, 105, 27],
        [0.15, 0.10, 0.55, 90,  20],
        [0.05, 0.05, 0.70, 80,  15],
        [0.03, 0.03, 0.75, 75,  12],
    ])
    y_train = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y_train)
    joblib.dump((model, scaler), MODEL_PATH)
    print("[ML] ✅ Modèle entraîné")
    return model, scaler

def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        model, scaler = joblib.load(MODEL_PATH)
        print("[ML] ✅ Modèle chargé")
        return model, scaler
    return create_ml_model()

LABELS = {0: "Saine", 1: "Stress hydrique", 2: "Maladie détectée", 3: "Plante sèche"}

# ═══════════════════════════════════════
# THREAD CAPTURE (rapide)
# ═══════════════════════════════════════
latest_frame = None
frame_lock = threading.Lock()

def capture_thread():
    """Thread dédié uniquement à la capture — très rapide"""
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[CAMERA] ❌ Caméra non trouvée")
        return

    print("[IA] ✅ Thread capture démarré")
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        time.sleep(0.05)  # 20 FPS max

# ═══════════════════════════════════════
# THREAD ANALYSE IA (lent — séparé)
# ═══════════════════════════════════════
def analyze_plant_ai(shared_state=None):
    """Analyse IA dans un thread séparé"""
    global latest_frame

    # Lancer thread capture en arrière-plan
    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    # Charger modèle ML
    model, scaler = load_or_create_model()

    print("[IA] ✅ Analyse IA démarrée")
    time.sleep(2)  # Attendre que la caméra démarre

    while True:
        # Récupérer dernière frame
        with frame_lock:
            if latest_frame is None:
                time.sleep(1)
                continue
            frame = latest_frame.copy()

        try:
            # ── Analyse HSV ──
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            total = frame.shape[0] * frame.shape[1]

            green_mask  = cv2.inRange(hsv, (35, 40, 40),  (85, 255, 255))
            yellow_mask = cv2.inRange(hsv, (20, 80, 80),  (35, 255, 255))
            brown_mask  = cv2.inRange(hsv, (10, 50, 20),  (20, 200, 120))

            green_ratio  = cv2.countNonZero(green_mask)  / total
            yellow_ratio = cv2.countNonZero(yellow_mask) / total
            brown_ratio  = cv2.countNonZero(brown_mask)  / total

            health  = min(100, int(green_ratio  * 400))
            stress  = min(100, int(yellow_ratio * 500))
            disease = min(100, int(brown_ratio  * 600))
            anomaly = min(100, max(0, 100 - health))

            # ── ML Random Forest ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            contrast   = float(np.std(gray))
            features = np.array([[green_ratio, yellow_ratio,
                                   brown_ratio, brightness, contrast]])
            features_scaled = scaler.transform(features)
            prediction  = model.predict(features_scaled)[0]
            proba       = model.predict_proba(features_scaled)[0]
            confidence  = int(max(proba) * 100)
            label       = LABELS[prediction]

            # Score composite
            composite = int(health*0.4 + (100-stress)*0.3 + (100-disease)*0.3)

            # Mise à jour état
            if shared_state is not None:
                shared_state["ai_health"]     = composite
                shared_state["ai_stress"]     = stress
                shared_state["ai_disease"]    = disease
                shared_state["ai_anomaly"]    = anomaly
                shared_state["ai_label"]      = label
                shared_state["ai_confidence"] = confidence
                shared_state["ai_ml_class"]   = int(prediction)

            # Sauvegarder snapshot compressé
            cv2.imwrite(SNAPSHOT_PATH, frame,
                       [cv2.IMWRITE_JPEG_QUALITY, 70])

            print(f"[IA] 🌿 Santé:{composite}% | "
                  f"Stress:{stress}% | Maladie:{disease}% | "
                  f"ML:{label}({confidence}%)")

        except Exception as e:
            print(f"[IA] Erreur: {e}")

        time.sleep(5)  # Analyse toutes les 5s
