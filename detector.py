import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "yolov8n_float32.tflite"
CONFIDENCE_THRESHOLD = 0.3
PERSON_CLASS_ID = 0

class PersonDetector:
    def __init__(self):
        self.interpreter = Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        print(f"모델 로드 완료: {self.input_width}x{self.input_height}")

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, frame):
        img = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        output = np.squeeze(output)
        persons_detected = False
        if output.ndim == 2:
            for detection in output.T:
                scores = detection[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                    persons_detected = True
                    break
        return persons_detected

if __name__ == "__main__":
    detector = PersonDetector()
    print("감지 테스트 중...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(dummy_frame)
    print(f"테스트 결과: {result}")
    print("detector.py 정상 동작!")
