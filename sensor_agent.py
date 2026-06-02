import time
import json
import cv2
import numpy as np
import base64
import threading
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt
from picamera2 import Picamera2
from detector import PersonDetector
from face_recognizer import detect_and_recognize
from datetime import datetime
import os

os.makedirs("clips", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)
os.makedirs("faces", exist_ok=True)

PIR_PIN = 17

BROKER = "localhost"
TOPIC_PIR = "doorguard/pir"
TOPIC_DETECTION = "doorguard/detection"
TOPIC_FRAME = "doorguard/frame"
TOPIC_STATUS = "doorguard/status"
TOPIC_FACE = "doorguard/face"

THRESHOLD_SEC = 1
FPS = 10
BUFFER_SEC = 5
SNAPSHOT_INTERVAL = 2

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(1)
print("카메라 초기화 완료")

detector = PersonDetector()
print("모델 로드 완료")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, 1883, 60)
client.loop_start()
print("MQTT 연결 완료")

def save_snapshot(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"snapshots/{timestamp}.jpg"
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return path

def save_clip(frames):
    if not frames:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"clips/{timestamp}.avi"
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    return path

current_candidate_id = None
candidate_lock = threading.Lock()

def run_face_recognition(snapshot_path, candidate_id):
    global current_candidate_id
    try:
        face_id, new_candidate_id = detect_and_recognize(
            snapshot_path, candidate_id)
        with candidate_lock:
            if new_candidate_id:
                current_candidate_id = new_candidate_id
        if face_id:
            client.publish(TOPIC_FACE, json.dumps({
                "face_id": face_id,
                "snapshot_path": snapshot_path,
                "timestamp": datetime.now().isoformat()
            }))
            print(f"얼굴 인식 완료: 인물 #{face_id}")
    except Exception as e:
        print(f"얼굴 인식 오류: {e}")

# AI 추론 별도 스레드
latest_frame_for_detection = None
detection_result = False
detection_lock = threading.Lock()

def detection_thread():
    global detection_result, latest_frame_for_detection
    while True:
        with detection_lock:
            frame = latest_frame_for_detection
        if frame is not None:
            result = detector.detect(frame)
            with detection_lock:
                detection_result = result
        time.sleep(0.2)

t_detect = threading.Thread(target=detection_thread, daemon=True)
t_detect.start()

person_start_time = None
is_recording = False
recording_frames = []
buffer_frames = []
snapshot_path = None
last_snapshot_time = 0
frame_count = 0
log_count = 0

print("sensor_agent 시작!")

try:
    while True:
        frame = picam2.capture_array()
        buffer_frames.append(frame.copy())
        if len(buffer_frames) > BUFFER_SEC * FPS:
            buffer_frames.pop(0)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with detection_lock:
            latest_frame_for_detection = frame_bgr.copy()

        frame_count += 1
        if frame_count % 2 == 0:
            _, buf = cv2.imencode('.jpg', frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 50])
            client.publish(TOPIC_FRAME,
                base64.b64encode(buf).decode('utf-8'))

        pir = GPIO.input(PIR_PIN)
        client.publish(TOPIC_PIR, json.dumps({"detected": bool(pir)}))

        with detection_lock:
            person = detection_result

        log_count += 1
        if log_count % 20 == 0:
            print(f"PIR:{pir} | Person:{person} | Recording:{is_recording}")

        if person:
            now = time.time()

            if person_start_time is None:
                person_start_time = now
                recording_frames = buffer_frames.copy()
                with candidate_lock:
                    current_candidate_id = None
                last_snapshot_time = 0
                snapshot_path = None
                print("사람 감지 시작!")

            duration = now - person_start_time
            recording_frames.append(frame.copy())

            # SNAPSHOT_INTERVAL초마다 스냅샷 찍어서 같은 후보에 누적
            if now - last_snapshot_time >= SNAPSHOT_INTERVAL:
                snap = save_snapshot(frame)
                last_snapshot_time = now
                if snapshot_path is None:
                    snapshot_path = snap
                print(f"스냅샷 저장: {snap}")

                with candidate_lock:
                    cid = current_candidate_id

                t = threading.Thread(
                    target=run_face_recognition,
                    args=(snap, cid),
                    daemon=True
                )
                t.start()

            if duration >= THRESHOLD_SEC and not is_recording:
                is_recording = True
                print(f"배회 감지! {duration:.1f}초")

            client.publish(TOPIC_STATUS, json.dumps({
                "person_detected": True,
                "is_recording": is_recording,
                "duration": round(duration, 1)
            }))

        else:
            if person_start_time is not None:
                duration = time.time() - person_start_time
                clip_path = None

                if is_recording:
                    clip_path = save_clip(recording_frames)
                    print(f"클립 저장: {clip_path}")

                client.publish(TOPIC_DETECTION, json.dumps({
                    "duration": round(duration, 1),
                    "clip_path": clip_path,
                    "snapshot_path": snapshot_path,
                    "timestamp": datetime.now().isoformat()
                }))
                print("감지 종료 → MQTT 발행")

                person_start_time = None
                is_recording = False
                recording_frames = []
                snapshot_path = None
                last_snapshot_time = 0
                with candidate_lock:
                    current_candidate_id = None

            client.publish(TOPIC_STATUS, json.dumps({
                "person_detected": False,
                "is_recording": False,
                "duration": 0
            }))

        time.sleep(1.0 / FPS)

except KeyboardInterrupt:
    print("sensor_agent 종료")
finally:
    picam2.stop()
    GPIO.cleanup()
    client.loop_stop()
