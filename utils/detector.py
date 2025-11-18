# EmotionSensor/utils/detector.py
import mediapipe as mp
import cv2

class FaceDetector:
    def __init__(self, min_detection_confidence=0.6, detection_scale=320):
        self.min_detection_confidence = float(min_detection_confidence)
        self.detection_scale = int(detection_scale)
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(self.min_detection_confidence)

    def detect_faces(self, frame):
        """
        Resize frame to detection_scale for speed, then map boxes back to original frame.
        Returns list of bounding boxes as (x, y, w, h) in pixel coords (original frame).
        """
        if frame is None:
            return []

        orig_h, orig_w = frame.shape[:2]
        # resize to square detection size while keeping aspect ratio
        scale_w = self.detection_scale
        scale_h = int(self.detection_scale * (orig_h / orig_w))
        small = cv2.resize(frame, (scale_w, scale_h))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        results = self.face_detection.process(rgb_small)
        boxes = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                # bbox relative to small frame
                x = int(max(bboxC.xmin * scale_w, 0))
                y = int(max(bboxC.ymin * scale_h, 0))
                w = int(min(bboxC.width * scale_w, scale_w - x))
                h = int(min(bboxC.height * scale_h, scale_h - y))
                # map back to original frame size
                scale_x = orig_w / scale_w
                scale_y = orig_h / scale_h
                x_o = int(x * scale_x)
                y_o = int(y * scale_y)
                w_o = int(w * scale_x)
                h_o = int(h * scale_y)
                # ensure valid
                x_o = max(0, x_o)
                y_o = max(0, y_o)
                w_o = min(orig_w - x_o, w_o)
                h_o = min(orig_h - y_o, h_o)
                boxes.append((x_o, y_o, w_o, h_o))
        return boxes
