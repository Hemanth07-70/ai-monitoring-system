import cv2
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from detection.landmarks import HandLandmarker
from data.collect import save_samples

def main():
    print("--- SOS Sentinel: Hand Gesture Data Collector ---")
    landmarker = HandLandmarker()
    cap = cv2.VideoCapture(0)
    
    gesture_name = input("Enter gesture label (e.g., 'help', 'normal', 'stop'): ").strip().lower()
    if not gesture_name:
        print("Label required. Exiting.")
        return

    buffer = []
    print(f"\nCollecting for '{gesture_name}'")
    print("INSTRUCTIONS:")
    print("1. Perform the gesture in front of the camera.")
    print("2. When landmarks (green lines) appear, press SPACE to save.")
    print("3. Collect around 200-500 samples for better accuracy.")
    print("4. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = landmarker.process(rgb)
        
        if hands:
            # Draw for feedback
            frame = landmarker.draw_landmarks(frame, hands)
            # Add to temporary buffer (auto-capture frames while hand is visible)
            buffer.append(hands[0])
            if len(buffer) > 50: # Keep latest 50 frames in buffer
                buffer.pop(0)
        else:
            buffer = []

        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Buffer: {len(buffer)} (Press SPACE to Save)", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '): # SPACE
            if buffer:
                n = save_samples(gesture_name, buffer)
                print(f"Saved {n} samples to dataset/{gesture_name}/")
                buffer = []
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
