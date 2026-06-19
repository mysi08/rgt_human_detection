#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
import queue
import threading
import message_filters
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image, LaserScan, PointCloud2
from cv_bridge import CvBridge
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from .detector import YoloDetector
from . import fusion

cv2.setNumThreads(1)

class AsynchronousFusionNode(Node):
    def __init__(self):
        super().__init__("yolo_lidar_fusion")
        
        # Core Params
        self.declare_parameter('enable_debug_image', True)
        self.declare_parameter('process_fps', 10.0)
        self.declare_parameter('openvino_num_threads', 2)
        self.declare_parameter('yolo_max_candidates', 40)
        self.declare_parameter('conf_threshold', 0.40)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('model_path', "")
        
        # Camera & Calibration Params
        self.declare_parameter('cam_hfov', 60.0)
        self.declare_parameter('cam_vfov', 60.0)
        self.declare_parameter('cam_height', 1.7)
        self.declare_parameter('lidar_height', 0.3)
        self.declare_parameter('lidar_behind', 0.15)
        
        # LiDAR ROI & Geometry Params
        self.declare_parameter('max_range', 6.0)
        self.declare_parameter('human_dist_min', 1.0)
        self.declare_parameter('human_dist_max', 5.0)
        self.declare_parameter('jump_noise_floor', 0.06)
        self.declare_parameter('min_human_points', 3)
        self.declare_parameter('max_human_points', 60)
        self.declare_parameter('min_human_width', 0.05)
        self.declare_parameter('max_human_width', 0.7)
        self.declare_parameter('max_linearity_ratio', 15.0)
        self.declare_parameter('min_yolo_overlap', 0.50)
        self.declare_parameter('max_stride_length', 0.9)

        self.cfg = {
            'enable_debug_image': self.get_parameter('enable_debug_image').value,
            'process_fps': self.get_parameter('process_fps').value,
            'openvino_num_threads': self.get_parameter('openvino_num_threads').value,
            'yolo_max_candidates': self.get_parameter('yolo_max_candidates').value,
            'conf_threshold': self.get_parameter('conf_threshold').value,
            'nms_threshold': self.get_parameter('nms_threshold').value,
            'cam_hfov': self.get_parameter('cam_hfov').value,
            'cam_vfov': self.get_parameter('cam_vfov').value,
            'cam_height': self.get_parameter('cam_height').value,
            'lidar_height': self.get_parameter('lidar_height').value,
            'lidar_behind': self.get_parameter('lidar_behind').value,
            'max_range': self.get_parameter('max_range').value,
            'human_dist_min': self.get_parameter('human_dist_min').value,
            'human_dist_max': self.get_parameter('human_dist_max').value,
            'jump_noise_floor': self.get_parameter('jump_noise_floor').value,
            'min_human_points': self.get_parameter('min_human_points').value,
            'max_human_points': self.get_parameter('max_human_points').value,
            'min_human_width': self.get_parameter('min_human_width').value,
            'max_human_width': self.get_parameter('max_human_width').value,
            'max_linearity_ratio': self.get_parameter('max_linearity_ratio').value,
            'min_yolo_overlap': self.get_parameter('min_yolo_overlap').value,
            'max_stride_length': self.get_parameter('max_stride_length').value,
        }
        
        model_p_str = self.get_parameter('model_path').value
        if not model_p_str:
            pkg_share = get_package_share_directory('obt')
            model_path = Path(pkg_share) / "model/yolo11n_openvino_model/yolo11n.xml"
        else:
            model_path = Path(model_p_str)

        self._cached_angles = None
        self.bridge = CvBridge()
        self.task_queue = queue.Queue(maxsize=2)

        self.detector = YoloDetector(
            model_path=str(model_path),
            num_threads=self.cfg['openvino_num_threads'],
            conf_thresh=self.cfg['conf_threshold'],
            nms_thresh=self.cfg['nms_threshold'],
            max_candidates=self.cfg['yolo_max_candidates']
        )
        
        # Initialize the tracker
        self.tracker = fusion.CentroidTracker(max_dist=1.5, max_age=10)

        self.pub_img = self.create_publisher(Image, "d9qv9c/debug_image", 1)
        self.pub_scan = self.create_publisher(LaserScan, "d9qv9c/scan_segment_all", 1)
        self.pub_nav = self.create_publisher(String, "d9qv9c/nav_info", 1)
        self.pub_cluster_pts = self.create_publisher(MarkerArray, "d9qv9c/cluster_points", 1)
        self.pub_cluster_ctr = self.create_publisher(MarkerArray, "d9qv9c/cluster_centroids", 1)
        self.pub_human_pc = self.create_publisher(PointCloud2, "d9qv9c/human_pointcloud", 1)
        self.pub_bbox3d = self.create_publisher(MarkerArray, "d9qv9c/bbox3d", 1)

        self.sub_img = message_filters.Subscriber(self, Image, "/d9qv9c/mono_ft/image_raw")
        self.sub_scan = message_filters.Subscriber(self, LaserScan, "/d9qv9c/scan_f")
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_scan], queue_size=5, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        
        self.worker_active = True
        self.target_frame_time = 1.0 / self.cfg['process_fps']
        self.last_queued_time = 0.0 

        self.inference_thread = threading.Thread(target=self.asynchronous_consumer_worker, daemon=True)
        self.inference_thread.start()
        self.get_logger().info("LiDAR-First Fusion pipeline successfully loaded.")
    
    def sync_callback(self, img_msg, scan_msg):
        current_time = time.perf_counter()
        if (current_time - self.last_queued_time) < self.target_frame_time:
            return
        try:
            self.task_queue.put_nowait((img_msg, scan_msg))
            self.last_queued_time = current_time
        except queue.Full:
            pass 

    def asynchronous_consumer_worker(self):
        while rclpy.ok() and self.worker_active:
            try:
                img_msg, scan_msg = self.task_queue.get(block=True, timeout=None)
            except (queue.Empty, TypeError):
                continue 

            try:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
                h_img, w_img = cv_image.shape[:2]
                self.cfg['h_img'] = h_img
                self.cfg['w_img'] = w_img

                t0 = time.perf_counter()
                humans = self.detector.run(cv_image)
                yolo_ms = (time.perf_counter() - t0) * 1000

                stamp = self.get_clock().now().to_msg()
                frame_id = scan_msg.header.frame_id

                if self._cached_angles is None or len(self._cached_angles) != len(scan_msg.ranges):
                    self._cached_angles = scan_msg.angle_min + np.arange(len(scan_msg.ranges), dtype=np.float32) * scan_msg.angle_increment

                confirmed_persons = fusion.process_frustum_fusion(
                    scan_msg, humans, w_img, h_img, self.cfg, 
                    cached_angles=self._cached_angles, 
                    tracker=self.tracker
                )

                pc_msg = fusion.create_human_pointcloud(confirmed_persons, frame_id, stamp)
                self.pub_human_pc.publish(pc_msg)

                global_h_r, global_h_idx = [], []
                nav_data = []

                for person in confirmed_persons:
                    global_h_r.extend(person['r'].tolist())
                    global_h_idx.extend(person['idx'].tolist())
                    nav_data.append({
                        "name": person.get('name', 'Human'),
                        "side": "LEFT" if person['side'] == "L" else "RIGHT",
                        "angle": person['angle'],
                        "dist": person['dist'],
                        "points": person['points']
                    })

                if self.pub_cluster_pts.get_subscription_count() > 0:
                    fusion.publish_cluster_points(self.pub_cluster_pts, confirmed_persons, frame_id, stamp)
                if self.pub_cluster_ctr.get_subscription_count() > 0:
                    fusion.publish_cluster_centroids(self.pub_cluster_ctr, confirmed_persons, frame_id, stamp)
                if self.pub_bbox3d.get_subscription_count() > 0:
                    fusion.publish_3d_bboxes(self.pub_bbox3d, confirmed_persons, frame_id, stamp, self.cfg)

                seg = LaserScan()
                seg.header = scan_msg.header
                seg.angle_min = scan_msg.angle_min
                seg.angle_max = scan_msg.angle_max
                seg.angle_increment = scan_msg.angle_increment
                seg.time_increment = scan_msg.time_increment
                seg.scan_time = scan_msg.scan_time
                seg.range_min = scan_msg.range_min
                seg.range_max = scan_msg.range_max
                
                ranges_np = np.full(len(scan_msg.ranges), np.inf, dtype=np.float32)
                if global_h_idx:
                    ranges_np[global_h_idx] = global_h_r
                seg.ranges = ranges_np.tolist()
                self.pub_scan.publish(seg)

                if nav_data:
                    parts = [f"{d['name']}: {d['side']}, angle={d['angle']:.1f}deg, distance={d['dist']:.2f}m, pts={d['points']}" for d in nav_data]
                    self.pub_nav.publish(String(data=" | ".join(parts)))
                else:
                    self.pub_nav.publish(String(data="NO_HUMANS_WITH_LIDAR"))

                if self.cfg['enable_debug_image'] and self.pub_img.get_subscription_count() > 0:
                    out = cv_image.copy()
                    fusion.draw_combined_debug_view(out, scan_msg, confirmed_persons, w_img, h_img, self.cfg, self._cached_angles)
                    fps_label = f"YOLO: {yolo_ms:.1f}ms @ {self.cfg['process_fps']} FPS"
                    cv2.putText(out, fps_label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    self.pub_img.publish(self.bridge.cv2_to_imgmsg(out, "bgr8"))

            except Exception as e:
                self.get_logger().error(f"Error inside processing loop: {str(e)}")

    def stop_worker(self):
        self.worker_active = False

def main(args=None):
    rclpy.init(args=args)
    node = AsynchronousFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_worker()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()