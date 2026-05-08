"""
AgroChain AI Agent — Powered by Claude API
Analyzes sensor data in real-time and takes autonomous decisions
"""

import anthropic
import json
import time
import threading
import paho.mqtt.client as mqtt
from datetime import datetime

# ── Config ──
MQTT_BROKER = "192.168.1.3"
MQTT_PORT   = 1883
PUMP_TOPIC  = "smartcity/garden/pump/control"
ALERT_TOPIC = "smartcity/alerts"

# ── Shared sensor state ──
sensor_state = {
    "temperature": 0,
    "humidity": 0,
    "soil": 0,
    "gas": 0,
    "flame": False,
    "rain": False,
    "distance": 0,
    "flow": 0,
    "ldr": 0,
    "pump": False,
    "ai_health": 0,
    "ai_stress": 0,
    "ai_disease": 0,
}

agent_log = []   # history of agent decisions
last_decision_time = 0
DECISION_INTERVAL = 30  # seconds between agent decisions

# ─────────────────────────────────────────
# MQTT callbacks
# ─────────────────────────────────────────
def on_message(client, userdata, msg):
    global sensor_state
    try:
        topic = msg.topic
        data  = json.loads(msg.payload.decode())

        if "climate"   in topic: sensor_state.update({"temperature": data.get("temperature", 0), "humidity": data.get("humidity", 0)})
        elif "soil"    in topic: sensor_state["soil"]     = data.get("value", 0)
        elif "gas"     in topic: sensor_state["gas"]      = data.get("ppm", 0)
        elif "flame"   in topic: sensor_state["flame"]    = data.get("detected", False)
        elif "rain"    in topic: sensor_state["rain"]     = data.get("detected", False)
        elif "distance"in topic: sensor_state["distance"] = data.get("cm", 0)
        elif "waterflow"in topic:sensor_state["flow"]     = data.get("lpm", 0)
        elif "light"   in topic: sensor_state["ldr"]      = data.get("value", 0)
        elif "pump"    in topic: sensor_state["pump"]     = data.get("state") == "ON"
    except Exception as e:
        print(f"[MQTT] Parse error: {e}")

def setup_mqtt():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    topics = [
        "smartcity/garden/climate",
        "smartcity/garden/soil",
        "smartcity/garden/waterflow",
        "smartcity/garden/pump",
        "smartcity/environment/gas",
        "smartcity/environment/rain",
        "smartcity/environment/light",
        "smartcity/security/flame",
        "smartcity/security/distance",
    ]
    for t in topics:
        client.subscribe(t)

    client.loop_start()
    return client

# ─────────────────────────────────────────
# Claude AI Agent
# ─────────────────────────────────────────
def run_agent(mqtt_client):
    """
    Sends sensor data to Claude and executes its decisions.
    """
    global last_decision_time, sensor_state, agent_log

    claude = anthropic.Anthropic()

    system_prompt = """You are AgroChain AI Agent — an autonomous agricultural IoT controller.

You receive real-time sensor data from a smart farm and must make intelligent decisions.

Your available actions (respond ONLY with valid JSON):
{
  "decision": "PUMP_ON" | "PUMP_OFF" | "NO_ACTION",
  "alert_level": "NONE" | "INFO" | "WARNING" | "CRITICAL",
  "alert_message": "short message in French for the farmer",
  "reasoning": "brief explanation of your decision in French",
  "recommendation": "one practical advice for the farmer in French"
}

Decision rules to follow:
- PUMP_ON if: soil < 30% AND rain = false AND reservoir > 10cm AND no flame
- PUMP_OFF if: soil > 70% OR rain = true OR flame detected OR reservoir < 5cm
- CRITICAL alert if: flame detected OR gas > 500ppm OR temperature > 40°C
- WARNING alert if: soil < 25% OR temperature > 35°C OR gas > 300ppm OR reservoir < 15cm
- INFO alert if: pump state changed OR disease score > 50%

Always respond ONLY with the JSON object, no extra text."""

    while True:
        time.sleep(5)
        now = time.time()

        if now - last_decision_time < DECISION_INTERVAL:
            continue

        last_decision_time = now

        # Build sensor summary for Claude
        s = sensor_state
        user_message = f"""Données capteurs en temps réel — {datetime.now().strftime('%H:%M:%S')}:

- Température: {s['temperature']:.1f}°C
- Humidité air: {s['humidity']:.0f}%
- Humidité sol: {s['soil']:.0f}%
- Gaz MQ135: {s['gas']:.0f} ppm
- Flamme détectée: {'OUI ⚠️' if s['flame'] else 'Non'}
- Pluie détectée: {'OUI' if s['rain'] else 'Non'}
- Niveau réservoir: {s['distance']:.0f} cm
- Débit eau: {s['flow']:.2f} L/min
- Luminosité: {s['ldr']:.0f}
- Pompe actuelle: {'ON' if s['pump'] else 'OFF'}
- Santé plantes (IA): {s['ai_health']:.0f}%
- Stress hydrique (IA): {s['ai_stress']:.0f}%
- Maladie détectée (IA): {s['ai_disease']:.0f}%

Prends une décision autonome basée sur ces données."""

        try:
            print(f"\n[Agent] 🤖 Analyse en cours...")

            response = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )

            raw = response.content[0].text.strip()
            # Clean JSON if wrapped in markdown
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            decision = json.loads(raw.strip())

            print(f"[Agent] Decision: {decision['decision']}")
            print(f"[Agent] Alert: {decision['alert_level']} — {decision['alert_message']}")
            print(f"[Agent] Reasoning: {decision['reasoning']}")

            # ── Execute pump decision ──
            if decision["decision"] == "PUMP_ON" and not sensor_state["pump"]:
                mqtt_client.publish(PUMP_TOPIC, json.dumps({"command": "ON", "source": "AI_AGENT"}))
                print("[Agent] ✅ Pompe démarrée automatiquement")

            elif decision["decision"] == "PUMP_OFF" and sensor_state["pump"]:
                mqtt_client.publish(PUMP_TOPIC, json.dumps({"command": "OFF", "source": "AI_AGENT"}))
                print("[Agent] ⛔ Pompe arrêtée automatiquement")

            # ── Publish alert ──
            if decision["alert_level"] != "NONE":
                alert_payload = {
                    "level": decision["alert_level"],
                    "message": decision["alert_message"],
                    "source": "AI_AGENT",
                    "timestamp": datetime.now().isoformat()
                }
                mqtt_client.publish(ALERT_TOPIC, json.dumps(alert_payload))

            # ── Log decision ──
            log_entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "decision": decision["decision"],
                "alert": decision["alert_level"],
                "message": decision["alert_message"],
                "reasoning": decision["reasoning"],
                "recommendation": decision.get("recommendation", ""),
                "sensors": {
                    "temp": s["temperature"],
                    "soil": s["soil"],
                    "gas": s["gas"],
                    "flame": s["flame"],
                    "pump": s["pump"],
                }
            }
            agent_log.insert(0, log_entry)
            if len(agent_log) > 50:
                agent_log.pop()

        except json.JSONDecodeError as e:
            print(f"[Agent] ❌ JSON parse error: {e} — raw: {raw[:100]}")
        except Exception as e:
            print(f"[Agent] ❌ Error: {e}")

# ─────────────────────────────────────────
# Flask route to expose agent log
# (add this to your existing server.py)
# ─────────────────────────────────────────
def get_agent_log():
    """Call this from server.py to expose /agent endpoint"""
    return agent_log

# ─────────────────────────────────────────
# Standalone run (or import into server.py)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🌿 AgroChain AI Agent starting...")
    print("📡 Connecting to MQTT broker...")

    mqtt_client = setup_mqtt()
    time.sleep(2)
    print("✅ MQTT connected — listening to sensors")
    print("🤖 Claude AI Agent active — decision every 30s\n")

    # Run agent in main thread
    try:
        run_agent(mqtt_client)
    except KeyboardInterrupt:
        print("\n[Agent] Stopped.")
        mqtt_client.loop_stop()
