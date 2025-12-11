import os
import cv2
import numpy as np
import rosbag2_py
import argparse
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
import zlib
import struct

# Fix numpy float deprecation
if not hasattr(np, 'float'):
    np.float = float

# ROS message types
from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2
from tf2_msgs.msg import TFMessage
from tf_transformations import quaternion_matrix

# ---------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--bag", required=True, help="Path to the bag folder (contains metadata.yaml)")
parser.add_argument("--out", required=True, help="Output folder where frames will be saved")
args = parser.parse_args()

bag_uri = os.path.expanduser(args.bag)
out_dir = os.path.expanduser(args.out)
os.makedirs(out_dir, exist_ok=True)

print(f" Reading bag: {bag_uri}")
print(f" Outputs will be saved to: {out_dir}")

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
bridge = CvBridge()
latest_depth = None
latest_depth_stamp = None
latest_tf = None
latest_tf_stamp = None
latest_cam_info = None

# Configuration
TARGET_PARENT_FRAME = "odom"
TARGET_CHILD_FRAME = "zed_left_camera_optical_frame"
TIME_THRESHOLD = 1e8  # 100ms in nanoseconds

# ---------------------------------------------------------------------
# Prepare rosbag2 reader
# ---------------------------------------------------------------------
storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3")
converter_options = rosbag2_py.ConverterOptions("", "")
reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

# ---------------------------------------------------------------------
# List topics
# ---------------------------------------------------------------------
info = rosbag2_py.Info()
metadata = info.read_metadata(bag_uri, "sqlite3")
print("\n Topics found:")
for t in metadata.topics_with_message_count:
    print(f"  {t.topic_metadata.name} ({t.topic_metadata.type}) × {t.message_count}")

topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}

# ---------------------------------------------------------------------
# Counters and limits
# ---------------------------------------------------------------------
rgb_count, depth_count, lidar_count = 0, 0, 0
frame_limit = 2000        # Limit frames to extract
max_messages = 200000    # Safety stop
msg_count = 0
tf_tree = {}

# ---------------------------------------------------------------------
# Read messages
# ---------------------------------------------------------------------
while reader.has_next():
    topic, data, t = reader.read_next()
    msg_count += 1
    if msg_count >= max_messages:
        print(f"\n[INFO] Reached {max_messages} messages, stopping early.")
        break

    msg_type = type_map[topic]
    if "." in msg_type:
        # e.g., "sensor_msgs.msg.CompressedImage"
        module_name, class_name = msg_type.rsplit(".", 1)
    else:
        # e.g., "sensor_msgs/msg/CompressedImage"
        module_name = msg_type.replace("/", ".").rsplit(".", 1)[0]
        class_name = msg_type.split("/")[-1]
    exec(f"from {module_name} import {class_name}")
    msg_class = eval(class_name)
    msg = deserialize_message(data, msg_class)

    # ---------------------------------------------------------------
    # RGB Image (This is the "Trigger")
    # ---------------------------------------------------------------
    if topic == "/zed/zed_node/rgb/image_rect_color/compressed":

        # --- 1. Check if we have all data ---
        if latest_depth is None:
            print(f"[RGB] Frame {rgb_count}: Skipping, missing depth.")
            continue
        if latest_tf is None:
            print(f"[RGB] Frame {rgb_count}: Skipping, missing TF.")
            continue

        # --- 2. Check if data is synchronized ---
        time_rgb = t
        delta_depth = abs(time_rgb - latest_depth_stamp)
        delta_tf = abs(time_rgb - latest_tf_stamp)

        if delta_depth > TIME_THRESHOLD or delta_tf > TIME_THRESHOLD:
            print(f"[RGB] Frame {rgb_count}: Skipping, data is out of sync.")
            print(f"  Deltas: Depth={delta_depth/1e9:.3f}s, TF={delta_tf/1e9:.3f}s")
            continue

        # --- 3. SYNC OK: Save all data for this frame ---
        print(f"[RGB] Frame {rgb_count}: Saving sync'd data.")

        # Save RGB
        np_arr = np.frombuffer(msg.data, np.uint8)
        image_rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(out_dir, f"rgb_{rgb_count:05d}.png"), image_rgb)

        # Save the *synced* RAW Depth
        np.save(os.path.join(out_dir, f"depth_raw_{rgb_count:05d}.npy"), latest_depth)

        # Save the *synced* TF
        np.save(os.path.join(out_dir, f"tf_odom_to_cam_{rgb_count:05d}.npy"), latest_tf)

        rgb_count += 1
        if rgb_count >= frame_limit:
            print(f"Reached frame limit of {frame_limit}. Stopping.")
            break

    # ---------------------------------------------------------------
    # Depth image (correct decode for "32FC1; compressedDepth" PNG)
    # ---------------------------------------------------------------
    elif topic == "/zed/zed_node/depth/depth_registered/compressedDepth":
        if latest_cam_info is None:
            print("[Depth] Skipping frame, camera info not yet received.")
            continue

        height = latest_cam_info.height
        width = latest_cam_info.width

        try:
            data_bytes = bytes(msg.data)

            # Find PNG header (0x89 0x50 0x4E 0x47)
            start_idx = data_bytes.find(b"\x89PNG")
            if start_idx == -1:
                print("[Depth] No PNG header found, skipping.")
                continue

            # Extract the PNG part and decode it using OpenCV
            png_bytes = np.frombuffer(data_bytes[start_idx:], np.uint8)
            depth_image = cv2.imdecode(png_bytes, cv2.IMREAD_UNCHANGED)

            if depth_image is None or depth_image.size == 0:
                print("[Depth] cv2 PNG decode failed.")
                continue

            print(f"[Depth] Decoded PNG depth: shape={depth_image.shape}, dtype={depth_image.dtype}")

            # Store depth buffer
            latest_depth = depth_image.astype(np.float32)
            latest_depth_stamp = t
            depth_count += 1

            # Print statistics for sanity
            print(f"[Depth] Stats: min={np.nanmin(latest_depth):.3f}, max={np.nanmax(latest_depth):.3f}")

            # Save visualization (scaled)
            depth_vis = cv2.convertScaleAbs(latest_depth, alpha=20)
            cv2.imwrite(os.path.join(out_dir, f"depth_vis_{depth_count:05d}.png"), depth_vis)

        except Exception as e:
            print(f"[Depth] Decode failed: {e}")
            continue


    # ---------------------------------------------------------------
    # Camera intrinsics
    # ---------------------------------------------------------------
    elif topic == "/zed/zed_node/rgb/camera_info":
        latest_cam_info = msg
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d)
        np.savez(os.path.join(out_dir, "camera_intrinsics.npz"), K=K, D=D)
        print("[INFO] Saved camera intrinsics")

    # ---------------------------------------------------------------
    # LiDAR point cloud
    # ---------------------------------------------------------------
    elif topic.startswith("/livox/lidar"):
        points = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        )
        np.savez_compressed(os.path.join(out_dir, f"lidar_{lidar_count:05d}.npz"), points=points)
        lidar_count += 1
        print(f"[LiDAR] Saved cloud {lidar_count}")

    # ---------------------------------------------------------------
    # TF transforms
    # ---------------------------------------------------------------
    elif topic in ["/tf", "/tf_static"]:
        for transform in msg.transforms:
            parent = transform.header.frame_id
            child  = transform.child_frame_id
            translation = transform.transform.translation
            rotation    = transform.transform.rotation

            T = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
            T[0:3, 3] = [translation.x, translation.y, translation.z]

            # store each transform in a dictionary
            if "tf_tree" not in globals():
                tf_tree = {}
            tf_tree[(parent, child)] = T

        # try composing if we have all needed links
        required_chain = [
            ("odom", "base_footprint"),
            ("base_footprint", "base_link"),
            ("base_link", "zed_camera_link"),
            ("zed_camera_link", "zed_camera_center"),
            ("zed_camera_center", "zed_left_camera_frame"),
            ("zed_left_camera_frame", "zed_left_camera_optical_frame")
        ]

        if all(link in tf_tree for link in required_chain):
            latest_tf = np.eye(4)
            for link in required_chain:
                latest_tf = latest_tf @ tf_tree[link]
            latest_tf_stamp = t


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n Extraction complete:")
print(f"  RGB frames:   {rgb_count}")
print(f"  Depth frames: {depth_count}")
print(f"  LiDAR clouds: {lidar_count}")
print(f"  Saved to:     {out_dir}")