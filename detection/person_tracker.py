from ultralytics import YOLO
import cv2

class PersonTracker:
    def __init__(self, model_variant="yolov8n.pt"):
        # Load pre-trained YOLOv8 model for detection + tracking
        self.model = YOLO(model_variant)

    def track(self, frame):
        """
        Track people in the frame.
        Returns:
            - results: YOLO prediction results
            - person_info: List of (id, box) where box is [x1, y1, x2, y2]
        """
        # Run YOLOv8 tracking
        results = self.model.track(frame, persist=True, classes=[0], verbose=False) # class 0 is person
        
        person_info = []
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                person_info.append({
                    "id": track_id,
                    "box": box.astype(int)
                })
        
        return results[0], person_info
