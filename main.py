from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import cv2
import numpy as np
import base64
import threading
import json
import os
import paho.mqtt.client as mqtt
from datetime import datetime
from database import init_db, get_db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 전역 상태 ──
active_connections = []
latest_frame = None

detection_state = {
    "pir_detected": False,
    "person_detected": False,
    "is_recording": False,
    "duration": 0,
    "distance": 0,
    "door_open": False,
    "last_event": None,
}

# ── MQTT 콜백 ──
def on_frame(client, userdata, msg):
    global latest_frame
    try:
        data = base64.b64decode(msg.payload)
        arr = np.frombuffer(data, np.uint8)
        latest_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        pass

def on_pir(client, userdata, msg):
    data = json.loads(msg.payload)
    detection_state["pir_detected"] = data.get("detected", False)

def on_ultrasonic(client, userdata, msg):
    data = json.loads(msg.payload)
    detection_state["distance"] = data.get("distance", 0)

def on_door(client, userdata, msg):
    data = json.loads(msg.payload)
    detection_state["door_open"] = data.get("open", False)
    if data.get("open"):
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO events (timestamp, duration_sec, clip_path, event_type)
                VALUES (?, ?, ?, ?)
            ''', (data.get("timestamp"), 0, None, "door_open"))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB 오류: {e}")

def on_detection(client, userdata, msg):
    data = json.loads(msg.payload)
    detection_state["last_event"] = data
    detection_state["person_detected"] = False
    detection_state["is_recording"] = False
    detection_state["duration"] = 0

    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (timestamp, duration_sec, clip_path, snapshot_path, event_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            data.get("timestamp"),
            data.get("duration", 0),
            data.get("clip_path"),
            data.get("snapshot_path"),
            "detection"
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB 오류: {e}")

def on_status(client, userdata, msg):
    data = json.loads(msg.payload)
    detection_state["person_detected"] = data.get("person_detected", False)
    detection_state["is_recording"] = data.get("is_recording", False)
    detection_state["duration"] = data.get("duration", 0)

# ── MQTT 설정 ──
def setup_mqtt():
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.message_callback_add("doorguard/frame", on_frame)
    mqtt_client.message_callback_add("doorguard/pir", on_pir)
    mqtt_client.message_callback_add("doorguard/ultrasonic", on_ultrasonic)
    mqtt_client.message_callback_add("doorguard/door", on_door)
    mqtt_client.message_callback_add("doorguard/detection", on_detection)
    mqtt_client.message_callback_add("doorguard/status", on_status)
    mqtt_client.connect("localhost", 1883, 60)
    mqtt_client.subscribe("doorguard/#")
    mqtt_client.loop_start()
    print("MQTT 구독 시작!")
    return mqtt_client

async def notify_clients(message):
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except:
            pass

@app.on_event("startup")
async def startup():
    init_db()
    setup_mqtt()
    print("DoorGuard v2.0 서버 시작!")

@app.get("/api/status")
async def get_status():
    return detection_state

@app.get("/api/events")
async def get_events():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events ORDER BY created_at DESC LIMIT 50")
    events = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return events

@app.get("/api/frame")
async def get_frame():
    if latest_frame is None:
        return JSONResponse({"error": "프레임 없음"})
    _, buffer = cv2.imencode('.jpg', latest_frame,
                             [cv2.IMWRITE_JPEG_QUALITY, 60])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return {"image": img_base64}

@app.get("/api/clips/{filename}")
async def get_clip(filename: str):
    path = f"clips/{filename}"
    if not os.path.exists(path):
        return JSONResponse({"error": "파일 없음"})
    return FileResponse(path, media_type="video/x-msvideo", filename=filename)

@app.get("/api/snapshots/{filename}")
async def get_snapshot(filename: str):
    path = f"snapshots/{filename}"
    if not os.path.exists(path):
        return JSONResponse({"error": "파일 없음"})
    return FileResponse(path, media_type="image/jpeg")

@app.get("/api/faces")
async def get_faces():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM faces ORDER BY last_seen DESC")
    faces = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return faces

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await asyncio.sleep(0.1)
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 60])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_json({
                    "type": "frame",
                    "image": img_base64,
                    "status": detection_state
                })
    except WebSocketDisconnect:
        active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
