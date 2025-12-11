import cv2
import yaml
import numpy as np
import json
import glob
import os
from math import cos, sin, sqrt
from collections import defaultdict

try:
    from Challenge_Tracker import Tracker 
except ImportError:
    print("\n--- ERROR: 'challenge_tracker.py' not found. ---")
    exit()

# --- CONFIGURATION: Max reasonable dimensions (width, length) in meters ---
MAX_DIMS = {
    "cup": (0.10, 0.10),
    "bottle": (0.10, 0.10),
    "wc": (0.5, 0.7),
    "sink": (0.6, 0.5),
    "chair": (0.6, 0.6),
    "table": (1.8, 1.0),
    "couch": (2.2, 1.0),
    "shelf": (1.2, 0.5),
    "tv": (1.2, 0.2),
    "book": (0.2, 0.2)
}

def build_map_to_odom_T(origin):
    ox, oy, yaw = origin
    c = cos(yaw)
    s = sin(yaw)
    return np.array([
        [ c, -s, 0, ox ],
        [ s,  c, 0, oy ],
        [ 0,  0, 1,  0 ],
        [ 0,  0, 0,  1 ]
    ], dtype=float)

class ProjectionTransformer:
    def __init__(self, intrinsics_file_path):
        with np.load(intrinsics_file_path) as data:
            self.K = data['K']
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])

    def project_2d_to_3d_camera(self, u, v, depth_m):
        if depth_m <= 0.0 or np.isnan(depth_m):
            return None
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return np.array([x, y, z, 1.0])

    def world_to_map_pixel(self, x_world, y_world, map_res, map_org_xy, map_h):
        px = int((x_world - map_org_xy[0]) / map_res)
        py = map_h - int((y_world - map_org_xy[1]) / map_res)
        return (px, py)

def is_valid_location(map_img, px, py, map_res):
    """
    STRICT VALIDITY CHECK.
    - REJECT if center is in Gray/Unknown (val ~ 205).
    - ACCEPT if center is Free (val ~ 254).
    - CHECK if center is Wall (val ~ 0): Allow only if touching Free space.
    """
    h, w = map_img.shape[:2]
    if px < 0 or px >= w or py < 0 or py >= h:
        return False
    
    # Get pixel intensity (handle BGR)
    val = map_img[py, px]
    if isinstance(val, np.ndarray): val = val[0]
    
    # 1. THE "NO GHOSTS" RULE:
    # Unknown space in PGM maps is usually 205 (or 127). 
    # Free is 254. Wall is 0.
    # If value is between 100 and 230, it is UNKNOWN space. REJECT.
    if 100 < val < 230:
        return False

    # 2. Free Space (White) -> Valid
    if val > 230:
        return True
        
    # 3. Wall (Black) -> Check neighbors
    # If it's a wall, is it a "surface" wall? (Within 20cm of free space)
    if val < 100:
        radius_px = int(np.ceil(0.2 / map_res)) # 20cm lookaround
        
        start_x = max(0, px - radius_px)
        end_x = min(w, px + radius_px + 1)
        start_y = max(0, py - radius_px)
        end_y = min(h, py + radius_px + 1)
        
        patch = map_img[start_y:end_y, start_x:end_x]
        if len(patch.shape) == 3: patch = patch[:,:,0]
        
        # If there is ANY white pixel nearby, keep it (it's a shelf/bottle on wall)
        if np.any(patch > 230):
            return True
            
    return False

def estimate_pose_and_size_from_points(pts, class_name="unknown"):
    pts = np.asarray(pts)
    if pts.shape[0] < 5: return None
    
    xy = pts[:, :2]
    center = xy.mean(axis=0)
    centered = xy - center
    
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eig(cov)
    sort_indices = np.argsort(vals)[::-1]
    main_axis = vecs[:, sort_indices[0]]
    
    # Percentile 10-90 to ignore outliers
    proj_main = centered @ main_axis
    proj_side = centered @ vecs[:, sort_indices[1]]
    
    l_min, l_max = np.percentile(proj_main, [10, 90])
    w_min, w_max = np.percentile(proj_side, [10, 90])
    
    length = float(l_max - l_min)
    width = float(w_max - w_min)

    # Constraint dimensions
    max_w, max_l = MAX_DIMS.get(class_name, (2.0, 2.0))
    width = min(width, max_w)
    length = min(length, max_l)
    
    # Min dimensions
    width = max(0.05, width)
    length = max(0.05, length)

    yaw = float(np.arctan2(main_axis[1], main_axis[0]))
    return float(center[0]), float(center[1]), yaw, width, length

def sample_depth(depth_raw, bbox):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = depth_raw.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    crop = depth_raw[y1:y2, x1:x2]
    if crop.size == 0: return []
    
    valid_depths = crop[crop > 50] / 1000.0 
    if len(valid_depths) == 0: return []

    med_d = np.median(valid_depths)
    pts = []
    # Stride 2
    for r in range(y1, y2, 2):
        for c in range(x1, x2, 2):
            d_val = depth_raw[r, c] / 1000.0
            # Strict depth slice (+/- 25cm)
            if d_val > 0.05 and abs(d_val - med_d) < 0.25:
                pts.append((c, r, d_val))
    return pts

def draw_box(map_image, transformer, x, y, yaw, w, l, map_res, map_origin, map_h, color=(0,255,0)):
    dx = w / 2; dy = l / 2
    c = cos(yaw); s = sin(yaw)
    corners = [
        (x + dx*c - dy*s, y + dx*s + dy*c),
        (x - dx*c - dy*s, y - dx*s + dy*c),
        (x - dx*c + dy*s, y - dx*s - dy*c),
        (x + dx*c + dy*s, y + dx*s - dy*c)
    ]
    pts = []
    for (wx, wy) in corners:
        px, py = transformer.world_to_map_pixel(wx, wy, map_res, map_origin[:2], map_h)
        pts.append((int(px), int(py)))
    cv2.polylines(map_image, [np.array(pts, dtype=np.int32)], True, color, 2)

def merge_close_objects(detections, dist_thresh=0.5):
    """
    Merges objects of the same class that are physically very close.
    """
    if not detections: return []
    
    # Simple clustering
    merged = []
    while detections:
        curr = detections.pop(0)
        cx, cy, _ = curr['pose']
        
        # Find close neighbors of same class
        neighbors = []
        others = []
        for d in detections:
            dx, dy, _ = d['pose']
            dist = sqrt((cx-dx)**2 + (cy-dy)**2)
            if dist < dist_thresh and d['class'] == curr['class']:
                neighbors.append(d)
            else:
                others.append(d)
        
        # Merge current + neighbors
        if neighbors:
            all_objs = [curr] + neighbors
            avg_pose = np.mean([o['pose'] for o in all_objs], axis=0).tolist()
            # Keep max dimensions to cover both
            max_w = max([o['dimensions'][0] for o in all_objs])
            max_l = max([o['dimensions'][1] for o in all_objs])
            
            merged.append({
                "class": curr['class'],
                "pose": avg_pose,
                "dimensions": [max_w, max_l]
            })
            detections = others
        else:
            merged.append(curr)
            detections = others
    return merged

def run_challenge_1(data_folder, map_yaml_file, map_pgm_file):
    print("--- Starting Challenge 1 (Final Strict) ---")

    map_image = cv2.imread(map_pgm_file)
    map_check = cv2.imread(map_pgm_file, cv2.IMREAD_GRAYSCALE)
    if map_image is None: return

    map_h, map_w, _ = map_image.shape
    with open(map_yaml_file, 'r') as f:
        cfg = yaml.safe_load(f)
    map_res = float(cfg["resolution"])
    map_origin = cfg["origin"]

    T_map_to_odom = build_map_to_odom_T(map_origin)
    tracker = Tracker("yolo11l.pt")
    intr = ProjectionTransformer(os.path.join(data_folder, "camera_intrinsics.npz"))
    rgb_files = sorted(glob.glob(os.path.join(data_folder, "rgb_*.png")))
    
    detections = defaultdict(list)
    print(f"Processing {len(rgb_files)} frames...")

    for rgb_file in rgb_files:
        frame_id = os.path.basename(rgb_file).split('.')[0].replace("rgb_", "")
        depth_file = os.path.join(data_folder, f"depth_raw_{frame_id}.npy")
        tf_file = os.path.join(data_folder, f"tf_odom_to_cam_{frame_id}.npy")

        if not (os.path.exists(depth_file) and os.path.exists(tf_file)): continue

        depth_raw = np.load(depth_file)
        T_odom_to_cam = np.load(tf_file)
        T_map_to_cam = T_odom_to_cam @ T_map_to_odom
        T_cam_to_map = np.linalg.inv(T_map_to_cam)

        rgb = cv2.imread(rgb_file)
        tracks = tracker.update(rgb)

        for bbox, cls_name, tid in tracks:
            depth_samples = sample_depth(depth_raw, bbox)
            if not depth_samples: continue
                
            world_pts = []
            for (u,v,d) in depth_samples:
                pt_cam_h = intr.project_2d_to_3d_camera(u,v,d)
                if pt_cam_h is None: continue
                pt_world = (T_cam_to_map @ pt_cam_h)[:3]
                world_pts.append(pt_world)

            est = estimate_pose_and_size_from_points(world_pts, cls_name)
            if est is None: continue

            xw, yw, yaw, w, l = est

            # --- STRICT LOCATION CHECK ---
            px, py = intr.world_to_map_pixel(xw, yw, map_res, map_origin[:2], map_h)
            if not is_valid_location(map_check, px, py, map_res):
                continue

            detections[tid].append({
                "class": cls_name,
                "pose": [xw, yw, yaw],
                "dimensions": [w, l]
            })

    print("Merging tracks...")
    candidates = []
    for tid, items in detections.items():
        if len(items) < 5: continue # Require 5 detections to be real
        
        avg_pose = np.mean([d['pose'] for d in items], axis=0)
        avg_dims = np.mean([d['dimensions'] for d in items], axis=0)
        cls_name = items[0]['class']
        
        # Re-check Average Position
        px, py = intr.world_to_map_pixel(avg_pose[0], avg_pose[1], map_res, map_origin[:2], map_h)
        if not is_valid_location(map_check, px, py, map_res):
            continue

        candidates.append({
            "class": cls_name,
            "pose": avg_pose.tolist(),
            "dimensions": avg_dims.tolist()
        })
        
    print(f"Candidates before deduplication: {len(candidates)}")
    final = merge_close_objects(candidates)
    print(f"Final objects: {len(final)}")
    
    for obj in final:
        xw, yw, yaw = obj['pose']
        w, l = obj['dimensions']
        print(f" - {obj['class']} ({xw:.2f}, {yw:.2f})")
        draw_box(map_image, intr, xw, yw, yaw, w, l, map_res, map_origin, map_h)

    cv2.imwrite("office_results.png", map_image)
    with open("office_results.json", "w") as f:
        json.dump(final, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    BASE = "/workspaces/Challenge_AICI/Output"
    run_challenge_1(BASE, "/workspaces/Challenge_AICI/room.yaml", "/workspaces/Challenge_AICI/room.pgm")
