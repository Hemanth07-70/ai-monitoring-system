import cv2
import time
import sys
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).resolve().parent))

from config import *
from detection.person_tracker import PersonTracker
from detection.landmarks import HandLandmarker
from detection.gesture_logic import is_distress_signal
from detection.verification import VerificationEngine
from alerts.notifier import AlertEngine

def main():
    print("--- SOS Distress Detection System ---")
    print("Initializing...")
    
    tracker = PersonTracker()
    landmarker = HandLandmarker()
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
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow("SOS MONITOR", cv2.WINDOW_NORMAL)
    
    print("Ready. Monitoring for Signal for Help...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        
        # 1. Track Persons
        _, persons = tracker.track(frame)
        
        # 2. Extract Hand Landmarks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = landmarker.process(rgb)
        
        # 3. Process Detections
        if hands:
            for hand_lms in hands:
                wrist = hand_lms[0]
                wrist_px = (int(wrist[0] * w), int(wrist[1] * h))
                
                # Associate hand with tracked person ID
                target_id = -1
                for p in persons:
                    bx = p["box"]
                    if bx[0] <= wrist_px[0] <= bx[2] and bx[1] <= wrist_px[1] <= bx[3]:
                        target_id = p["id"]
                        break
                
                # Draw hand
                frame = landmarker.draw_landmarks(frame, [hand_lms])
                
                if target_id != -1:
                    # Check for SOS gesture
                    is_sos, conf = is_distress_signal(hand_lms)
                    
                    # Verify repetition
                    trigger_alert, msg = verifier.update(target_id, is_sos, conf)
                    
                    if trigger_alert:
                        notifier.trigger(frame, msg)
                        cv2.putText(frame, "!!! ALERT SENT !!!", (50, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        # Draw Person Boxes
        for p in persons:
            bx = p["box"]
            cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {p['id']}", (bx[0], bx[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("SOS MONITOR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
