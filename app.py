from flask import Flask, render_template, Response, jsonify
import cv2
import time
import sys
from pathlib import Path
import threading

# Fix imports
sys.path.append(str(Path(__file__).resolve().parent))

from config import *
from detection.person_tracker import PersonTracker
from detection.landmarks import HandLandmarker
from detection.gesture_logic import is_distress_signal
from detection.verification import VerificationEngine
from alerts.notifier import AlertEngine
from model.predict import GesturePredictor
from data.preprocess import landmarks_to_features

app = Flask(__name__)

# TARGET GESTURE to detect via ML
TARGET_GESTURE = "help" 

# Global state
tracker = PersonTracker()
landmarker = HandLandmarker()
predictor = GesturePredictor()
verifier = VerificationEngine(
    threshold_count=REPETITION_THRESHOLD, 
    time_window=TIME_WINDOW_SECONDS, 
    min_confidence=CONFIDENCE_THRESHOLD
)
notifier = AlertEngine(
    sender_email=ALERT_EMAIL_SENDER,
    receiver_email=ALERT_EMAIL_RECEIVER,
    password=ALERT_EMAIL_PASSWORD
)

# Shared data for UI
stats = {
    "persons_count": 0,
    "last_alert_time": "Never",
    "alert_count": 0,
    "system_status": "Active"
}

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            h, w = frame.shape[:2]
            
            # 1. Track Persons
            _, persons = tracker.track(frame)
            stats["persons_count"] = len(persons)
            
            # 2. Extract Hand Landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = landmarker.process(rgb)
            
            # 3. Process Detections
            alert_active = False
            if hands:
                for hand_lms in hands:
                    wrist = hand_lms[0]
                    wrist_px = (int(wrist[0] * w), int(wrist[1] * h))
                    
                    target_id = -1
                    for p in persons:
                        bx = p["box"]
                        if bx[0] <= wrist_px[0] <= bx[2] and bx[1] <= wrist_px[1] <= bx[3]:
                            target_id = p["id"]
                            break
                    
                    # Draw hand
                    frame = landmarker.draw_landmarks(frame, [hand_lms])
                    
                    if target_id != -1:
                        # 1. Rule-based heuristic (backup/primary for SOS)
                        is_sos_heuristic, conf_h = is_distress_signal(hand_lms)
                        
                        # 2. ML Model Prediction
                        features = landmarks_to_features(hand_lms)
                        label, conf_m = predictor.predict(features)
                        
                        is_sos_ml = (label == TARGET_GESTURE)
                        
                        # Decide which one to use (here we combine them)
                        if is_sos_heuristic or is_sos_ml:
                            final_conf = max(conf_h, conf_m)
                            trigger_alert, msg = verifier.update(target_id, True, final_conf)
                            
                            if trigger_alert:
                                notifier.trigger(frame, msg)
                                stats["alert_count"] += 1
                                stats["last_alert_time"] = time.strftime("%H:%M:%S")
                                alert_active = True

            # Draw Person Boxes & Overlay
            for p in persons:
                bx = p["box"]
                cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (64, 255, 64), 2)
                cv2.putText(frame, f"ID:{p['id']}", (bx[0], bx[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (64, 255, 64), 2)

            if alert_active:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                cv2.putText(frame, "MONITORING ALERT TRIGGERED", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
