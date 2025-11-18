import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

class Config:
    def __init__(self, path=CONFIG_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            self._cfg = json.load(f)

    def get(self, *keys, default=None):
        node = self._cfg
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

# convenience top-level vars
_cfg = Config()
WEB_CAM_URL = _cfg.get("webcam", "url")
FRAME_WIDTH = _cfg.get("webcam", "frame_width", default=640)
FRAME_HEIGHT = _cfg.get("webcam", "frame_height", default=480)

MODEL_PATH = os.path.join(BASE_DIR, _cfg.get("model", "model_path"))
LABELS_PATH = os.path.join(BASE_DIR, _cfg.get("model", "labels_path"))
CONFIDENCE_THRESHOLD = _cfg.get("model", "confidence_threshold", default=0.5)

MIN_DETECTION_CONFIDENCE = _cfg.get("detector", "min_detection_confidence", default=0.7)

DISPLAY_CONFIG = _cfg.get("display", default={})
LOG_CONFIG = _cfg.get("logging", default={})
WEBCAM_CONFIG = _cfg.get("webcam", default={})
