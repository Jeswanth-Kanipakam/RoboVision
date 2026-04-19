# RoboVision with RGB-D & YOLO

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v11-orange)

## 📌 Project Overview
This project processes a sequence of **RGB-D (Red, Green, Blue + Depth)** frames from a mobile robot survey to identify, track, and locate office objects on a **2D Occupancy Grid Map**. 

By combining **YOLOv11** for detection, **ByteTrack** for object tracking, and custom **geometric projection** algorithms, the system estimates the 3D pose (x, y, yaw) and dimensions of objects like chairs, wc, sink and bottles. It features a robust "Ghost Filtering" system that strictly validates objects against the map's free space to prevent false detections in walls or unknown areas.

## 📂 Project Structure
```text
├── object_tracker.py    # Wrapper for YOLO + ByteTrack logic
├── map_processor.py     # Main script: Projects 3D points & generates map
├── room.pgm             # Input: Occupancy Grid Map (Image)
├── room.yaml            # Input: Map Metadata (Resolution, Origin)
├── Output/              # (Generated) Contains result images and JSON
├── office_results.png   # Final Output: Map with bounding boxes
├── office_results.json  # Final Output: Object list with pose/dimensions
├── rosbag.py            # Utility: Extracts RGB, Depth & TF data from ROS bags
```

## 🚀 Key Features
* **State-of-the-Art Detection:** Uses `YOLO11l` for high-accuracy object recognition.
* **3D Localization:** Projects 2D pixels into 3D world coordinates using camera intrinsics and TF transforms.
* **Smart Filtering:**
    * **Ghost Removal:** Rejects objects detected in "Unknown" (gray) map regions.
    * **Size Constraints:** Enforces realistic dimensions (e.g., a bottle cannot be 1 meter wide).
    * **Wall Logic:** Allows objects to touch walls but not be embedded deep inside them.
* **Duplicate Removal:** Merges multiple detections of the same object using spatial clustering.

## 📊 Outputs

<img width="391" height="380" alt="office_results" src="https://github.com/user-attachments/assets/42dbac8b-90d7-4160-a6b0-b20e77a85c2d" />

## 🤝 Contributing

Feel free to contribute by submitting pull requests! 🚀.

## 📞 Contact

JeswanthKanipakam

Email: jeswanthkanipakam@gmail.com
