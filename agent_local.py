"""
AgroChain Agent IA — Version Améliorée
Analyse intelligente multi-capteurs + ML
8 scénarios + réponses langage naturel
"""
from datetime import datetime

# ═══════════════════════════════════════
# SEUILS INTELLIGENTS
# ═══════════════════════════════════════
SEUILS = {
    "temp_critique":   38.0,
    "temp_warning":    33.0,
    "temp_froide":     10.0,
    "sol_sec":         25.0,
    "sol_sature":      80.0,
    "gaz_critique":    500,
    "gaz_warning":     400,
    "pluie_seuil":     30,
    "reservoir_vide":  5,
    "maladie_seuil":   40,
    "stress_seuil":    50,
}

def analyze_and_decide(state):
    t        = float(state.get("t", 0))
    h        = float(state.get("h", 0))
    s        = float(state.get("s", 0))
    g        = float(state.get("g", 0))
    flame    = bool(state.get("flame", False))
    distance = float(state.get("distance", 0))
    pump     = bool(state.get("pump", False))
    l        = float(state.get("l", 0))
    f        = float(state.get("f", 0))
    health   = float(state.get("ai_health", 0))
    disease  = float(state.get("ai_disease", 0))
    stress   = float(state.get("ai_stress", 0))
    label    = state.get("ai_label", "Inconnu")
    confidence = float(state.get("ai_confidence", 0))

    action = "NO_ACTION"
    alert  = "AUCUNE"
    message = ""
    recommendation = ""
    analysis = ""

    # ══ Scénario 1 : FLAMME ══
    if flame:
        action         = "NO_ACTION"
        alert          = "CRITIQUE"
        message        = "🔥 FLAMME DÉTECTÉE ! Urgence absolue — évacuez immédiatement !"
        recommendation = "Coupez l'alimentation. Appelez les secours. Arrêt total du système."
        analysis       = "Capteur IR détecte une source de chaleur intense. Risque incendie maximum."

    # ══ Scénario 2 : GAZ CRITIQUE ══
    elif g > SEUILS["gaz_critique"]:
        action         = "NO_ACTION"
        alert          = "CRITIQUE"
        message        = f"⚠️ DANGER GAZ — {g:.0f} ppm ! Aérez immédiatement !"
        recommendation = "Ne pas allumer d'équipement. Identifier la source de gaz."
        analysis       = f"MQ135: {g:.0f} ppm dépasse le seuil critique de {SEUILS['gaz_critique']} ppm."

    # ══ Scénario 3 : GAZ WARNING ══
    elif g > SEUILS["gaz_warning"]:
        action         = "NO_ACTION"
        alert          = "AVERTISSEMENT"
        message        = f"💨 Qualité air dégradée — {g:.0f} ppm. Surveillance requise."
        recommendation = "Augmentez la ventilation. Vérifiez engrais et pesticides."
        analysis       = f"MQ135: {g:.0f} ppm proche du seuil d'alerte de {SEUILS['gaz_warning']} ppm."

    # ══ Scénario 4 : TEMPÉRATURE CRITIQUE ══
    elif t > SEUILS["temp_critique"]:
        action         = "PUMP_ON"
        alert          = "CRITIQUE"
        message        = f"🌡️ CHALEUR CRITIQUE {t:.1f}°C — Irrigation d'urgence activée !"
        recommendation = "Installez des filets d'ombrage. Arrosez abondamment."
        analysis       = f"Température {t:.1f}°C dépasse le seuil critique de {SEUILS['temp_critique']}°C."

    # ══ Scénario 5 : SOL SEC ══
    elif s < SEUILS["sol_sec"] and not flame:
        if distance > SEUILS["reservoir_vide"]:
            action         = "PUMP_ON"
            alert          = "INFO"
            message        = f"🌱 Sol trop sec ({s:.0f}%) — Irrigation automatique activée !"
            recommendation = f"Arrosage 10-15 min recommandé. Temp: {t:.1f}°C."
            analysis       = f"Sol {s:.0f}% < seuil {SEUILS['sol_sec']}%. Réservoir OK ({distance:.0f} cm)."
        else:
            action         = "NO_ACTION"
            alert          = "AVERTISSEMENT"
            message        = f"🌱 Sol sec ({s:.0f}%) mais réservoir vide ! Remplissage urgent."
            recommendation = "Remplissez le réservoir immédiatement."
            analysis       = f"Sol critique {s:.0f}%. Réservoir insuffisant ({distance:.0f} cm)."

    # ══ Scénario 6 : SOL SATURÉ ══
    elif s > SEUILS["sol_sature"]:
        action         = "PUMP_OFF"
        alert          = "AUCUNE"
        message        = f"💧 Sol bien hydraté ({s:.0f}%) — Pompe arrêtée."
        recommendation = "Prochaine irrigation dans 2-3h selon évaporation."
        analysis       = f"Sol {s:.0f}% > seuil {SEUILS['sol_sature']}%. Pompe arrêtée automatiquement."

    # ══ Scénario 7 : MALADIE DÉTECTÉE PAR ML ══
    elif disease > SEUILS["maladie_seuil"] or stress > SEUILS["stress_seuil"]:
        action         = "NO_ACTION"
        alert          = "AVERTISSEMENT"
        message        = f"🌿 {label} détecté (confiance: {confidence:.0f}%) — Inspection requise !"
        recommendation = "Inspectez visuellement. Traitements phytosanitaires recommandés."
        analysis       = f"ML Random Forest: {label} avec {confidence:.0f}% confiance. Maladie: {disease:.0f}%, Stress: {stress:.0f}%."

    # ══ Scénario 8 : TEMPÉRATURE FROIDE ══
    elif t < SEUILS["temp_froide"]:
        action         = "NO_ACTION"
        alert          = "AVERTISSEMENT"
        message        = f"🥶 Température basse {t:.1f}°C — Risque gel des cultures !"
        recommendation = "Couvrez les cultures sensibles. Réduisez l'irrigation."
        analysis       = f"Température {t:.1f}°C sous le seuil minimal de {SEUILS['temp_froide']}°C."

    # ══ Scénario 9 : CONDITIONS OPTIMALES ══
    else:
        action         = "NO_ACTION"
        alert          = "AUCUNE"
        message        = f"✅ Conditions optimales — Système en parfait état."
        recommendation = f"Sol: {s:.0f}% | Temp: {t:.1f}°C | Gaz: {g:.0f}ppm | Santé: {health:.0f}%"
        analysis       = f"Tous les paramètres dans les plages normales. ML: {label} ({confidence:.0f}%)."

    # ══ RAPPORT ══
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    report = f"""📊 RAPPORT AGROCHAIN — {now}
{'━'*40}
🌡️  Température    : {t:.1f}°C
💧  Humidité air   : {h:.1f}%
🌱  Humidité sol   : {s:.0f}%
💨  Qualité air    : {g:.0f} ppm
🔥  Flamme         : {'⚠️ OUI' if flame else '✓ Non'}
📏  Réservoir      : {distance:.0f} cm
🌊  Débit eau      : {f:.2f} L/min
💧  Pompe          : {'🟢 ON' if pump else '⚫ OFF'}
🌿  Santé plantes  : {health:.0f}%
🦠  Maladie        : {disease:.0f}%
😰  Stress         : {stress:.0f}%
🤖  ML Diagnostic  : {label} ({confidence:.0f}%)
{'━'*40}
🤖  Décision       : {action}
🚨  Alerte         : {alert}
📋  {message}"""

    return {
        "action":         action,
        "alert":          alert,
        "message":        message,
        "recommendation": recommendation,
        "analysis":       analysis,
        "report_line":    report
    }

# ═══════════════════════════════════════
# RÉPONSES AUX QUESTIONS
# ═══════════════════════════════════════
def answer_question(question, state):
    t        = float(state.get("t", 0))
    h        = float(state.get("h", 0))
    s        = float(state.get("s", 0))
    g        = float(state.get("g", 0))
    flame    = bool(state.get("flame", False))
    pump     = bool(state.get("pump", False))
    f        = float(state.get("f", 0))
    distance = float(state.get("distance", 0))
    health   = float(state.get("ai_health", 0))
    disease  = float(state.get("ai_disease", 0))
    stress   = float(state.get("ai_stress", 0))
    label    = state.get("ai_label", "Inconnu")
    confidence = float(state.get("ai_confidence", 0))

    # Extraire vraie question du prompt système
    if 'Capteurs:' in question and 'JSON' in question:
        import re
        match = re.search(r'Question: "(.+?)"', question)
        if match:
            question = match.group(1)
        else:
            return analyze_and_decide(state)

    q = question.lower()

    # Température
    if any(w in q for w in ["temp", "chaud", "froid", "chaleur", "degre"]):
        msg = f"🌡️ Température actuelle : {t:.1f}°C, Humidité air : {h:.1f}%. "
        if t > 38:   msg += "⚠️ CRITIQUE ! Irrigation d'urgence recommandée."
        elif t > 33: msg += "Température élevée, surveillez le sol."
        elif t < 10: msg += "⚠️ Risque de gel ! Protégez les cultures."
        else:        msg += "Température normale et favorable aux cultures."

    # Sol
    elif any(w in q for w in ["sol", "terre", "humide", "sec", "arros", "irrigation"]):
        msg = f"🌱 Humidité du sol : {s:.0f}%. "
        if s < 25:   msg += "⚠️ Sol trop sec ! Irrigation nécessaire."
        elif s > 80: msg += "Sol saturé. Pas d'arrosage nécessaire."
        else:        msg += "Humidité correcte. Surveillance dans 1-2h."

    # Pompe
    elif any(w in q for w in ["pompe", "pump", "eau", "debit", "reservoir"]):
        msg = f"💧 Pompe : {'🟢 EN MARCHE' if pump else '⚫ ARRÊTÉE'}. "
        msg += f"Débit : {f:.2f} L/min. Réservoir : {distance:.0f} cm."

    # Gaz
    elif any(w in q for w in ["gaz", "air", "pollution", "qualite", "ppm"]):
        msg = f"💨 Qualité de l'air : {g:.0f} ppm. "
        if g > 500:  msg += "🚨 DANGER ! Niveau critique."
        elif g > 400: msg += "⚠️ Niveau élevé. Aérez la zone."
        else:         msg += "Air de bonne qualité. ✓"

    # Flamme
    elif any(w in q for w in ["feu", "flamme", "incendie", "fum", "brule"]):
        msg = f"🔥 Capteur flamme : {'⚠️ FLAMME DÉTECTÉE !' if flame else '✓ Aucune flamme.'}"

    # Plantes / IA
    elif any(w in q for w in ["plante", "sante", "maladie", "feuille", "ia", "camera"]):
        msg = f"🌿 Diagnostic ML : {label} (confiance {confidence:.0f}%). "
        msg += f"Santé : {health:.0f}% | Stress : {stress:.0f}% | Maladie : {disease:.0f}%. "
        if disease > 40:  msg += "⚠️ Inspection recommandée !"
        elif health > 70: msg += "Plantes en bonne santé ✓"
        else:             msg += "Surveillance recommandée."

    # Rapport complet
    elif any(w in q for w in ["rapport", "etat", "bilan", "tout", "complet", "resume"]):
        return analyze_and_decide(state)

    # Question générale
    else:
        msg = (f"🤖 État actuel — Temp: {t:.1f}°C | Sol: {s:.0f}% | "
               f"Gaz: {g:.0f}ppm | Pompe: {'ON' if pump else 'OFF'} | "
               f"Plantes: {label}. "
               f"Posez une question sur la température, le sol, la pompe, les plantes ou la qualité de l'air !")

    return {
        "action":         "NO_ACTION",
        "alert":          "AUCUNE",
        "message":        msg,
        "recommendation": "Posez une autre question ou demandez un rapport complet.",
        "analysis":       f"Question traitée: '{question}'",
        "report_line":    msg
    }
