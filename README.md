## MoodLens – Real-Time Emotion Detector

MoodLens is a real-time emotion detection system that captures your face through a webcam (or mobile camera) and predicts emotions like happy, sad, angry, surprised, and more. It’s built with Python, TensorFlow, MediaPipe, and OpenCV, and supports saving snapshots for analysis.

## Features

Real-time face detection using MediaPipe.

Emotion classification with a pre-trained deep learning model.

Supports mobile IP webcam and laptop webcam.

Draws bounding boxes and emotion labels on faces.

Saves detected face images automatically with timestamps.

Displays FPS (Frames per Second) to monitor performance.

Fully configurable via config.json.

## Project Structure
MoodLens/
-
--- main.py                 # Entry point for running live emotion detection
--- config.py               # Loads and manages config.json
--- config.json             # Settings for webcam, model, display, logging
--- requirements.txt        # Required Python packages
--- utils/
-   --- webcam.py           # Webcam handling
-   --- detector.py         # Face detection
-   --- emotion_model.py    # Loads model and predicts emotions
--- models/
-   --- emotion_model.h5    # Pre-trained emotion detection model
--- data/
-   --- labels.json         # Emotion labels mapping
--- outputs/
-   --- logs/               # Saved face snapshots
--- README.md

## Installation

Clone the repository

git clone https://github.com/yourusername/moodlens.git
cd moodlens


Install Python 3.10.9 (if not installed).

Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows


Install dependencies

pip install -r requirements.txt

##  Configuration

All parameters are stored in config.json:

Webcam: URL, frame width/height, reconnect attempts.

Model: Path to emotion_model.h5, label file, confidence threshold.

Face Detector: Detection confidence, scale for speed.

Display: Show bounding boxes, labels, FPS.

Logging: Save detected face images automatically.

You can customize these settings without editing the code.

## Usage

Run the main program:

python main.py


The webcam will open and start detecting faces.

Detected faces will be labeled with predicted emotions.

Snapshots are saved automatically to outputs/logs.

Press q to quit the program.

## How It Works

Webcam Module (webcam.py):
Captures frames from your camera or mobile device.

Face Detector (detector.py):
Uses MediaPipe to detect faces and returns bounding boxes.

Emotion Model (emotion_model.py):
Loads the pre-trained TensorFlow model, preprocesses the face image, predicts the emotion, and returns the label and confidence.

Main Script (main.py):
Combines webcam feed, face detection, and emotion prediction. Draws bounding boxes, labels, and FPS. Saves snapshots for each detected face.

## Future Improvements

Add emoji display for detected emotions.

Add hand gesture support for pause/stop.

Add voice feedback for each detected emotion.

Support multiple models (age, gender) at the same time.

## Requirements

Python 3.10.9

TensorFlow 2.19.1

OpenCV 4.12.0

MediaPipe 0.10.14

NumPy, Pillow, and other dependencies listed in requirements.txt.

## License

This project is licensed under the MIT License.