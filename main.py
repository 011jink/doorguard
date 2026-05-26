from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from picamera2 import Picamera2
import asyncio
import cv2
import numpy as np
import base64
import time
import threading
import os
from datetime import datetime
from database import init_db, get_db
from detector import PersonDetector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 상태
detector = None
picam2 = None
person_start_time = None
is_recording = False
recording_frames = []
active_connections = []
THRESHOLD_SEC = 20
CLIPS_DIR = "clips"
latest_frame = None

def init():
    global detector, picam2
    init_db()
    detector = PersonDetector()
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("카메라 초기화 완료")

def save_clip(frames):
    if not frames:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_path = f"{CLIPS_DIR}/{timestamp}.avi"
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    return clip_path

def save_event(duration, clip_path):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO events (timestamp, duration_sec, clip_path)
        VALUES (?, ?, ?)
    ''', (datetime.now().isoformat(), duration, clip_path))
    conn.commit()
    event_id = cursor.lastrowid
    conn.close()
    return event_id

async def notify_clients(message):
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except:
            pass

def detection_loop():
    global person_start_time, is_recording, recording_frames, latest_frame
    buffer_frames = []
    BUFFER_SEC = 5
    FPS = 10

    while True:
        if picam2 is None:
            time.sleep(0.1)
            continue

        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        latest_frame = frame_bgr.copy()

        # 버퍼 유지 (감지 전 5초)
        buffer_frames.append(frame_bgr.copy())
        if len(buffer_frames) > BUFFER_SEC * FPS:
            buffer_frames.pop(0)

        person_detected = detector.detect(frame_bgr)

        if person_detected:
            if person_start_time is None:
                person_start_time = time.time()
                recording_frames = buffer_frames.copy()
                print("사람 감지 시작!")

            duration = time.time() - person_start_time
            recording_frames.append(frame_bgr.copy())

            if duration >= THRESHOLD_SEC and not is_recording:
                is_recording = True
                print(f"배회 감지! {duration:.1f}초 경과")

        else:
            if person_start_time is not None:
                duration = time.time() - person_start_time
                if is_recording:
                    clip_path = save_clip(recording_frames)
                    save_event(duration, clip_path)
                    print(f"클립 저장 완료: {clip_path}")
                person_start_time = None
                is_recording = False
                recording_frames = []

        time.sleep(1.0 / FPS)

@app.on_event("startup")
async def startup():
    init()
    thread = threading.Thread(target=detection_loop, daemon=True)
    thread.start()
    print("DoorGuard 서버 시작!")

@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "person_detected": person_start_time is not None,
        "is_recording": is_recording,
        "duration": round(time.time() - person_start_time, 1) if person_start_time else 0
    }

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
    _, buffer = cv2.imencode('.jpg', latest_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return {"image": img_base64}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await asyncio.sleep(0.5)
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 50])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_json({
                    "type": "frame",
                    "image": img_base64,
                    "status": {
                        "person_detected": person_start_time is not None,
                        "is_recording": is_recording,
                        "duration": round(time.time() - person_start_time, 1) if person_start_time else 0
                    }
                })
    except WebSocketDisconnect:
        active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
