import cv2
import mediapipe as mp
import numpy as np

class HandLandmarker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb):
        """Return list of hand landmark arrays (21 x 3) per hand, or [] if none."""
        results = self.hands.process(frame_rgb)
        out = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                arr = []
                for lm in hand_landmarks.landmark:
                    arr.append([lm.x, lm.y, lm.z])
                out.append(np.array(arr, dtype=np.float32))
        return out

    def draw_landmarks(self, frame, hand_landmarks_list):
        """Draw landmarks on BGR frame."""
        if not hand_landmarks_list:
            return frame
        h, w = frame.shape[:2]
        mp_draw = mp.solutions.drawing_utils
        
        # Create a temporary landmarks object that MediaPipe can draw
        for hand_landmarks_arr in hand_landmarks_list:
            # We convert our numpy array back to MediaPipe's format for easy drawing
            for i, lm in enumerate(hand_landmarks_arr):
                x, y = int(lm[0] * w), int(lm[1] * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
            # Drawing simplified connections manually
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
            ]
            for start, end in connections:
                pt1 = (int(hand_landmarks_arr[start][0]*w), int(hand_landmarks_arr[start][1]*h))
                pt2 = (int(hand_landmarks_arr[end][0]*w), int(hand_landmarks_arr[end][1]*h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        return frame

    def close(self):
        self.hands.close()
