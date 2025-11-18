import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
from utils.webcam import Webcam
from utils.detector import FaceDetector
from utils.emotion_model import EmotionModel
from config import WEB_CAM_URL, MODEL_PATH, LABELS_PATH
import time

def main():
    print("Starting optimized Emotion Sensor. Press 'q' to quit.")

    # Initialize classes
    webcam = Webcam(WEB_CAM_URL)
    detector = FaceDetector(min_detection_confidence=0.7)
    emotion_model = EmotionModel(MODEL_PATH, LABELS_PATH)

    # Create output folder if not exists
    output_folder = os.path.join("outputs", "logs")
    os.makedirs(output_folder, exist_ok=True)

    fps_display_interval = 1  # seconds
    frame_count = 0
    start_time = time.time()
    pic_count = 0

    while True:
        frame = webcam.get_frame()
        if frame is None:
            print("Failed to grab frame. Exiting...")
            break

        boxes = detector.detect_faces(frame)

        for (x, y, w, h) in boxes:
            pad = 10
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
            face_img = frame[y1:y2, x1:x2]

            try:
                label, confidence = emotion_model.predict(face_img)
            except Exception as e:
                label, confidence = "error", 0
                print(f"[ERROR] Prediction failed: {e}")

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Save detected face image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            pic_filename = f"{label}_{timestamp}_{pic_count}.png"
            cv2.imwrite(os.path.join(output_folder, pic_filename), face_img)
            pic_count += 1

        # Show FPS
        frame_count += 1
        end_time = time.time()
        if (end_time - start_time) > fps_display_interval:
            fps = frame_count / (end_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            frame_count = 0
            start_time = end_time

        cv2.imshow("Emotion Sensor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    print("Emotion Sensor stopped.")

if __name__ == "__main__":
    main()
