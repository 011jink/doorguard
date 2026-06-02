import cv2
import numpy as np
import os
import shutil
from datetime import datetime
from database import get_db

FACES_DIR = "faces"
TEMP_DIR = "temp_faces"
CASCADE_PATH = "/home/jinkyu/doorguard/haarcascade_frontalface_default.xml"
SAMPLE_THRESHOLD = 10

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_trained = False
face_labels = {}

def preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return blurred

def detect_faces(gray):
    if gray.shape[0] < 50 or gray.shape[1] < 50:
        return []
    processed = preprocess(gray)
    faces = face_cascade.detectMultiScale(
        processed, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
    )
    if len(faces) > 0:
        return faces
    faces = face_cascade.detectMultiScale(
        processed, scaleFactor=1.03, minNeighbors=2, minSize=(20, 20)
    )
    if len(faces) > 0:
        return faces
    brightened = cv2.convertScaleAbs(processed, alpha=1.5, beta=30)
    faces = face_cascade.detectMultiScale(
        brightened, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20)
    )
    return faces

def load_recognizer():
    global recognizer_trained, face_labels
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, face_image_path FROM faces")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        recognizer_trained = False
        return

    faces_data = []
    labels = []
    face_labels = {}

    for row in rows:
        face_id = row["id"]
        folder = row["face_image_path"]
        if folder and os.path.isdir(folder):
            label = None
            for existing_label, existing_id in face_labels.items():
                if existing_id == face_id:
                    label = existing_label
                    break
            if label is None:
                label = len(face_labels)
                face_labels[label] = face_id

            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (100, 100))
                    faces_data.append(img)
                    labels.append(label)

    if faces_data:
        recognizer.train(faces_data, np.array(labels))
        recognizer_trained = True
        print(f"얼굴 인식기 학습 완료: {len(set(labels))}명 ({len(faces_data)}장)")

def crop_face(img, scale=1.5):
    if img is None or img.shape[0] < 50 or img.shape[1] < 50:
        return None, None

    img_large = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)

    try:
        faces_rect = detect_faces(gray)
    except Exception as e:
        print(f"얼굴 감지 오류: {e}")
        return None, None

    if len(faces_rect) == 0:
        return None, None

    x, y, w, h = max(faces_rect, key=lambda r: r[2] * r[3])
    ox = int(x / scale)
    oy = int(y / scale)
    ow = int(w / scale)
    oh = int(h / scale)
    pad = int(ow * 0.2)
    h_img, w_img = img.shape[:2]
    x1 = max(0, ox - pad)
    y1 = max(0, oy - pad)
    x2 = min(w_img, ox + ow + pad)
    y2 = min(h_img, oy + oh + pad)

    if x2 <= x1 or y2 <= y1:
        return None, None

    face_crop = img[y1:y2, x1:x2]
    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        return None, None

    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (100, 100))
    return face_crop, face_resized

def match_existing(face_resized):
    if not recognizer_trained or not face_labels:
        return None
    try:
        label, confidence = recognizer.predict(face_resized)
        print(f"기존 인물 비교: confidence={confidence:.1f}")
        if confidence < 200:
            face_id = face_labels[label]
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE faces
                SET last_seen = ?, visit_count = visit_count + 1
                WHERE id = ?
            ''', (datetime.now().isoformat(), face_id))
            conn.commit()
            conn.close()
            print(f"기존 인물 #{face_id} 재출현! (confidence={confidence:.1f})")
            return face_id
    except Exception as e:
        print(f"인식 오류: {e}")
    return None

def get_temp_candidate(face_resized):
    if not os.path.exists(TEMP_DIR):
        return None

    for temp_id in os.listdir(TEMP_DIR):
        temp_folder = os.path.join(TEMP_DIR, temp_id)
        if not os.path.isdir(temp_folder):
            continue
        samples = os.listdir(temp_folder)
        if len(samples) < 3:
            continue

        temp_faces = []
        temp_labels = []
        for fname in samples[:20]:
            fpath = os.path.join(temp_folder, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                temp_faces.append(img)
                temp_labels.append(0)

        if len(temp_faces) < 3:
            continue

        try:
            temp_rec = cv2.face.LBPHFaceRecognizer_create()
            temp_rec.train(temp_faces, np.array(temp_labels))
            _, confidence = temp_rec.predict(face_resized)
            print(f"임시 후보 {temp_id}: confidence={confidence:.1f}")
            if confidence < 200:
                return temp_id
        except:
            continue

    return None

def detect_and_recognize(snapshot_path, candidate_id=None):
    img = cv2.imread(snapshot_path)
    if img is None:
        print("스냅샷 읽기 실패")
        return None, candidate_id

    face_crop, face_resized = crop_face(img)
    if face_crop is None:
        print("얼굴 감지 안됨")
        return None, candidate_id

    print("얼굴 감지 성공!")

    # 1. 기존 등록 인물과 비교
    matched_id = match_existing(face_resized)
    if matched_id:
        return matched_id, None

    # 2. candidate_id 있으면 그 폴더에 누적
    if candidate_id is not None:
        face_folder = os.path.join(FACES_DIR, candidate_id)
        if os.path.isdir(face_folder):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            extra_path = os.path.join(face_folder, f"{timestamp}.jpg")
            cv2.imwrite(extra_path, face_crop)
            print(f"기존 등록 인물에 추가 학습: {candidate_id}")
            load_recognizer()
            return None, candidate_id

        temp_id = candidate_id
        print(f"기존 후보에 누적: {temp_id}")
    else:
        # 3. 임시 후보 폴더에서 비슷한 얼굴 찾기
        temp_id = get_temp_candidate(face_resized)
        if temp_id is None:
            temp_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            print(f"새 임시 후보 생성: {temp_id}")

    # 임시 폴더에 저장
    temp_folder = os.path.join(TEMP_DIR, temp_id)
    os.makedirs(temp_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_path = os.path.join(temp_folder, f"{timestamp}.jpg")
    cv2.imwrite(temp_path, face_crop)

    sample_count = len(os.listdir(temp_folder))
    print(f"임시 저장: {temp_id} ({sample_count}/{SAMPLE_THRESHOLD}장)")

    # 4. SAMPLE_THRESHOLD장 모이면 정식 등록
    if sample_count >= SAMPLE_THRESHOLD:
        face_folder = os.path.join(FACES_DIR, temp_id)
        shutil.move(temp_folder, face_folder)

        conn = get_db()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO faces (face_image_path, first_seen, last_seen, visit_count)
            VALUES (?, ?, ?, 1)
        ''', (face_folder, now, now))
        conn.commit()
        face_id = cursor.lastrowid
        conn.close()
        print(f"✅ 신규 인물 #{face_id} 정식 등록! ({SAMPLE_THRESHOLD}장 완료)")

        load_recognizer()
        return face_id, None

    return None, temp_id

load_recognizer()
