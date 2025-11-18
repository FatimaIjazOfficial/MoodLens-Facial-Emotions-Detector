# EmotionSensor/utils/webcam.py
import cv2
import time

class Webcam:
    def __init__(self, url, width=320, height=240, reconnect_attempts=3, reconnect_delay=2):
        self.url = url
        self.width = int(width) if width else None
        self.height = int(height) if height else None
        self.reconnect_attempts = int(reconnect_attempts)
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.open()

    def open(self):
        attempts = 0
        while attempts < self.reconnect_attempts:
            self.cap = cv2.VideoCapture(self.url)
            if self.cap.isOpened():
                # Apply small frame for speed
                if self.width:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                if self.height:
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return
            attempts += 1
            time.sleep(self.reconnect_delay)
        raise ConnectionError(f"Cannot open webcam stream at {self.url} after {self.reconnect_attempts} attempts")

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            try:
                self.open()
            except Exception:
                return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
