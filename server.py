# ═══════════════════════════════════════════════════════════
#  AgroChain Server — Raspberry Pi 4
#  MQTT → WebSocket → Dashboard
# ═══════════════════════════════════════════════════════════
from plant_detector import analyze_plant_ai
import json
import time
import hashlib
import threading
import requests
from alert_email import check_and_alert
from disease_predictor import predict_disease
from datetime import datetime
from flask import Flask, send_from_directory, request, send_file, jsonify
from flask_sock import Sock
import paho.mqtt.client as mqtt

app  = Flask(__name__)
sock = Sock(app)

FABRIC_URL = "http://192.168.1.169:8080/invoke"

state = {
    "t": 0, "h": 0, "l": 0, "f": 0,
    "s": 0, "g": 0, "flame": False,
    "pump": False, "distance": 0,
    "ai_health": 0, "ai_stress": 0,
    "ai_disease": 0, "ai_anomaly": 0,
    "ai_label": "Analyse...", "ai_confidence": 0, "ai_ml_class": 0,
    "ai_kaggle_label": "---", "ai_kaggle_confidence": 0,
    "last_block": "—"
}

connected_ws_clients = set()
blockchain = []

# ═══════════════════════════════════════════
#  HYPERLEDGER
# ═══════════════════════════════════════════
def record_to_hyperledger(block):
    try:
        payload = {
            "id":          str(block["index"]),
            "temperature": state.get("t", 0),
            "soil":        state.get("s", 0),
            "gas":         state.get("g", 0),
            "pump":        state.get("pump", False),
            "hash":        block["hash"]
        }
        res = requests.post(FABRIC_URL, json=payload, timeout=5)
        print(f"[HLF] ✅ Bloc #{block['index']} → Hyperledger: {res.status_code}")
    except Exception as e:
        print(f"[HLF] ⚠️ Erreur: {e}")

# ═══════════════════════════════════════════
#  BLOCKCHAIN
# ═══════════════════════════════════════════
def make_block(data):
    prev_hash = blockchain[-1]["hash"] if blockchain else "0" * 16
    payload   = json.dumps(data, sort_keys=True) + prev_hash + str(time.time())
    block_hash = "0x" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    block = {
        "index":     len(blockchain) + 1,
        "timestamp": datetime.now().isoformat(),
        "data":      {k: data[k] for k in ["t","h","l","f","s","g","flame","pump"]},
        "prev_hash": prev_hash,
        "hash":      block_hash
    }
    blockchain.append(block)

    try:
        with open("blockchain.json", "w") as f:
            json.dump(blockchain, f, indent=2)
    except Exception as e:
        print(f"[BLOCKCHAIN] Save error: {e}")

    print(f"[BLOCKCHAIN] Block #{block['index']} → {block_hash}")

    # Send to Hyperledger in background
    threading.Thread(target=record_to_hyperledger, args=(block,), daemon=True).start()

    return block

# ═══════════════════════════════════════════
#  MQTT CLIENT
# ═══════════════════════════════════════════
MQTT_BROKER = "localhost"
MQTT_PORT   = 1883

TOPIC_MAP = {
    "smartcity/garden/soil":        lambda d: state.update({"s": d.get("value", 0)}),
    "smartcity/garden/climate":     lambda d: state.update({"t": d.get("temperature", 0), "h": d.get("humidity", 0)}),
    "smartcity/environment/gas":    lambda d: state.update({"g": d.get("value", 0)}),
    "smartcity/environment/rain":   lambda d: None,
    "smartcity/environment/light":  lambda d: state.update({"l": d.get("value", 0)}),
    "smartcity/security/flame":     lambda d: state.update({"flame": d.get("detected", False)}),
    "smartcity/security/distance":  lambda d: state.update({"distance": d.get("distance_cm", 0)}),
    "smartcity/garden/waterflow":   lambda d: state.update({"f": d.get("rate_lpm", 0)}),
    "smartcity/garden/pump":        lambda d: state.update({"pump": d == "ON" or d.get("status") == "ON"}),
    "smartcity/alerts":             lambda d: print(f"[ALERT] {d}"),
}

block_counter = 0

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[MQTT] Connected!")
        for topic in TOPIC_MAP:
            client.subscribe(topic)
    else:
        print(f"[MQTT] Failed rc={rc}")

def on_mqtt_message(client, userdata, msg):
    global block_counter, connected_ws_clients
    try:
        topic   = msg.topic
        payload = msg.payload.decode()
        try:
            data = json.loads(payload)
            if not isinstance(data, dict):
                data = {"value": data}
        except:
            data = {"value": payload}

        state["last_mqtt"] = __import__("time").time()
        if topic in TOPIC_MAP:
            TOPIC_MAP[topic](data)

        broadcast(json.dumps(state))
        check_and_alert(state)

        block_counter += 1
        if block_counter >= 10:
            block_counter = 0
            block = make_block(state)
            state["last_block"] = block["hash"]
            broadcast(json.dumps(state))

    except Exception as e:
        print(f"[MQTT] Error: {e}")

def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_mqtt_connect
    client.on_message = on_mqtt_message
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            print(f"[MQTT] Reconnecting... ({e})")
            time.sleep(5)

# ═══════════════════════════════════════════
#  WEBSOCKET
# ═══════════════════════════════════════════
@sock.route("/ws")
def websocket_handler(ws):
    connected_ws_clients.add(ws)
    print(f"[WS] Client connected — {len(connected_ws_clients)} total")
    try:
        ws.send(json.dumps(state))
        while True:
            ws.receive(timeout=30)
    except Exception:
        pass
    finally:
        connected_ws_clients.discard(ws)

def broadcast(message):
    global connected_ws_clients
    dead = set()
    for client in connected_ws_clients:
        try:
            client.send(message)
        except:
            dead.add(client)
    connected_ws_clients -= dead

# ═══════════════════════════════════════════
#  HTTP ROUTES
# ═══════════════════════════════════════════
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/')
def index():
    return send_from_directory("static", "login.html")

@app.route("/state")
def get_state():
    return json.dumps(state)

@app.route("/blockchain")
def get_chain():
    return json.dumps(blockchain)

@app.route("/snapshot")
def snapshot():
    try:
        from flask import Response
        import io
        with open("/home/mouna/agrochain/static/snapshot.jpg", "rb") as f:
            data = f.read()
        response = Response(data, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-cache, no-store"
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    except Exception as e:
        print(f"[SNAPSHOT] Error: {e}")
        return "No snapshot yet", 404

@app.route("/agent/ask", methods=["POST"])
def agent_ask():
    from agent_local import analyze_and_decide, answer_question
    data = request.get_json() or {}
    question = data.get("prompt", "")
    if question:
        result = answer_question(question, state)
    else:
        result = analyze_and_decide(state)
    return jsonify(result), 200


@app.route("/system_status")
def system_status():
    import subprocess, os, time, cv2
    # CPU
    try:
        result = subprocess.check_output(['top','-bn1']).decode()
        lines = result.split(chr(10))
        cpu_line = [l for l in lines if 'Cpu' in l][0]
        cpu_usage = cpu_line.split()[1]
    except:
        cpu_usage = '0'
    # RAM
    try:
        mem = subprocess.check_output(['free','-m']).decode()
        mem_line = mem.split(chr(10))[1].split()
        ram_used = mem_line[2]
        ram_total = mem_line[1]
        ram_pct = int(int(ram_used)/int(ram_total)*100)
    except:
        ram_used, ram_total, ram_pct = 0, 4096, 0
    # Pi temp
    try:
        tmp = subprocess.check_output(['vcgencmd','measure_temp']).decode().strip()
        pi_temp = tmp.replace("temp=","").replace("'C","")
    except:
        pi_temp = '0'
    # Camera
    try:
        import os
        snapshot = '/home/mouna/agrochain/static/snapshot.jpg'
        cam_ok = os.path.exists('/dev/video0') and os.path.exists(snapshot)
    except:
        cam_ok = False
    # ESP32
    esp_ok = (time.time() - state.get("last_mqtt", 0)) < 30
    return jsonify({
        "cpu": cpu_usage,
        "ram_used": ram_used,
        "ram_total": ram_total,
        "ram_pct": ram_pct,
        "pi_temp": pi_temp,
        "camera": cam_ok,
        "esp32": esp_ok,
        "mqtt_broker": True,
        "hyperledger": True,
        "flask": True
    })

@app.route("/pump/on", methods=["POST"])
def pump_on():
    state["pump"] = True
    broadcast(json.dumps(state))
    return {"ok": True, "pump": True}

@app.route("/pump/off", methods=["POST"])
def pump_off():
    state["pump"] = False
    broadcast(json.dumps(state))
    return {"ok": True, "pump": False}

# ═══════════════════════════════════════════
#  START
# ═══════════════════════════════════════════

def reset_state_if_disconnected():
    import time
    while True:
        time.sleep(30)
        if time.time() - state.get("last_mqtt", 0) > 30:
            state.update({"t":0,"h":0,"s":0,"g":0,
                          "flame":False,"pump":False,
                          "distance":0,"f":0,"l":0})
            broadcast(json.dumps(state))
            print("[ESP32] Deconnecte — etat remis a zero")

if __name__ == "__main__":
    import os
    os.makedirs("static", exist_ok=True)

    make_block(state)

    threading.Thread(target=start_mqtt, daemon=True).start()
    threading.Thread(target=reset_state_if_disconnected, daemon=True).start()
    threading.Thread(target=analyze_plant_ai, args=(state,), daemon=True).start()

    print("=" * 50)
    print("🌿 AgroChain Server Starting...")
    print("   Dashboard  → http://0.0.0.0:5000")
    print("   WebSocket  → ws://0.0.0.0:5000/ws")
    print("   Blockchain → http://0.0.0.0:5000/blockchain")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5000, debug=False)
