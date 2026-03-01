import time

class VerificationEngine:
    def __init__(self, threshold_count=3, time_window=20, min_confidence=0.85):
        self.threshold_count = threshold_count
        self.time_window = time_window
        self.min_confidence = min_confidence
        self.detections = {} # { track_id: [timestamps] }
        self.last_alert = {} # { track_id: last_alert_time }

    def update(self, person_id, is_distress, confidence):
        current_time = time.time()
        
        # Cleanup old detections
        if person_id in self.detections:
            self.detections[person_id] = [t for t in self.detections[person_id] if current_time - t <= self.time_window]
        else:
            self.detections[person_id] = []

        if is_distress and confidence >= self.min_confidence:
            # Cooldown check
            if person_id in self.last_alert and current_time - self.last_alert[person_id] < 60:
                return False, None

            self.detections[person_id].append(current_time)
            
            if len(self.detections[person_id]) >= self.threshold_count:
                self.last_alert[person_id] = current_time
                self.detections[person_id] = [] # Reset
                return True, f"MONITORING ALERT: Person ID {person_id} signaled {self.threshold_count} times!"
        
        return False, None
