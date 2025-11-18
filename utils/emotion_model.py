# EmotionSensor/utils/emotion_model.py
import os
import json
import cv2
import numpy as np
import tensorflow as tf

class EmotionModel:
    def __init__(self, model_path, labels_path):
        # Check files
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found at {labels_path}")

        # Load labels
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        # Load model
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Auto-detect input shape
        input_shape = self.model.input_shape  # e.g., (None, 64, 64, 1)
        if len(input_shape) != 4:
            raise ValueError(f"Unsupported model input shape: {input_shape}")

        _, h, w, c = input_shape
        self.input_size = (w, h)
        self.input_channels = c

    def preprocess(self, face_img):
        if face_img is None:
            raise ValueError("face_img is None")

        # Convert to grayscale if model expects 1 channel
        if self.input_channels == 1:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Resize to model input
        face_resized = cv2.resize(face_img, self.input_size)

        # Add batch and channel dimension if needed
        if self.input_channels == 1:
            x = face_resized.reshape(1, self.input_size[1], self.input_size[0], 1)
        else:
            x = face_resized.reshape(1, self.input_size[1], self.input_size[0], self.input_channels)

        x = x.astype("float32") / 255.0
        return x

    def predict(self, face_img):
        x = self.preprocess(face_img)
        preds = self.model.predict(x, verbose=0)
        preds = np.array(preds).squeeze()

        # Softmax fallback if sum != 1
        if preds.sum() <= 0 or preds.sum() > 1.0001:
            exp = np.exp(preds - np.max(preds))
            preds = exp / exp.sum()

        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = self.labels.get(str(idx), f"label_{idx}")
        return label, confidence
