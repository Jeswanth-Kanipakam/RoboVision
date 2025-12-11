import numpy as np
import supervision as sv
from ultralytics import YOLO

class Tracker:
    """
    Wrapper around YOLO + ByteTrack with class mapping and filtering.
    """
    def __init__(self, model_path="yolo11l.pt"):
        print(f"  [Tracker] Loading model: {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception:
            print(f"  [Tracker] Fallback to yolov8m.pt")
            self.model = YOLO("yolov8m.pt")
            
        self.tracker = sv.ByteTrack()
        self.class_names = self.model.names
        
        # Mapping to the challenge classes
        self.class_mapping = {
            "dining table": "table",
            "desk": "table",
            "laptop": "table",
            "monitor": "tv",
            "keyboard": "table",
            "sofa": "couch",
            "bed": "couch", 
            "bench": "chair",
            "stool": "chair",
            "seat": "chair",
            "cabinet": "shelf",
            "refrigerator": "shelf",
            "microwave": "shelf",
            "oven": "shelf",
            "sink": "sink",
            "toilet": "wc"
        }
        
        self.target_classes_set = set(self.class_mapping.keys()).union({
            "chair", "couch", "table", "shelf", "wc", "sink", "tv", "book", "bottle", "cup"
        })
        
        self.target_class_ids = [
            cid for cid, name in self.class_names.items()
            if name in self.target_classes_set
        ]
        
        print(f"  [Tracker] Initialized. Target classes: {len(self.target_class_ids)} ids")

    def update(self, frame):
        results = self.model.predict(frame, conf=0.01, verbose=False)[0]
        detection_supervision = sv.Detections.from_ultralytics(results)
        
        mask = np.isin(detection_supervision.class_id, self.target_class_ids)
        filtered = detection_supervision[mask]
        
        tracked = self.tracker.update_with_detections(filtered)

        output = []
        for det in tracked:
            bbox = det[0].tolist()
            class_id = int(det[3])
            track_id = int(det[4])
            
            raw_name = self.class_names.get(class_id, "unknown")
            final_name = self.class_mapping.get(raw_name, raw_name)
            
            output.append([bbox, final_name, track_id])
            
        return output
