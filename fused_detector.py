#!/usr/bin/env python3
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_FOR_THREADS_NUM"] = "1"
os.environ["TBB_NUM_THREADS"] = "1"

sys.path.insert(0, "/home/rgt/.local/lib/python3.12/site-packages")

import numpy as np
np.int = int
np.float = float
np.bool = bool
np.complex = complex
np.object = object
np.str = str

import threading
import time

import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

import openvino as ov
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import LaserScan, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from model import Detector

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
CKPT_PATH   = os.path.join(WEIGHTS_DIR, "lidar_human.pth")
YOLO_MODEL  = os.path.join(WEIGHTS_DIR, "yolo/yolo11n.xml")

CONF_THRESH = 0.8
YOLO_CONF = 0.5
NMS_IOU = 0.45
MERGE_DIST = 0.8
EMA_ALPHA = 0.1
TRACK_MAX_MISS = 6
TRACK_MIN_HITS = 2

CONFIRM_HITS_ON = 1
CONFIRM_MISSES_OFF = 8

PROCESS_HZ = 5.0
YOLO_HZ = 0.5
MARKER_HZ = 5.0
LOG_HZ = 1.0
OV_CPU_THREADS = 1
MODEL_STRIDE = 5

SHOW_CIRCLE = True
SHOW_DOT = True
SHOW_LABEL = False
CIRCLE_RADIUS = 0.40
CIRCLE_SEGMENTS = 12

ROI_X_MIN, ROI_X_MAX = 1.0, 5.0
ROI_Y_MIN, ROI_Y_MAX = -2.0, 2.0

CAM_HFOV_DEG = 40.0
CAM_VFOV_DEG = 60.0
CAM_HEIGHT = 1.6
LIDAR_HEIGHT = 0.3
LIDAR_BEHIND = 0.3
IMG_W, IMG_H = 640, 480
PERSON_CLASS = 0

SCAN_TOPIC = "/d9qv9c/scan_f_filtered"
IMAGE_TOPIC = "/d9qv9c/mono_ft/image_raw"
MARKER_TOPIC = "/markers"


class Track:
    _next_id = 0

    def __init__(self, x, y, score):
        self.id = Track._next_id
        Track._next_id += 1
        self.x = float(x)
        self.y = float(y)
        self.score = float(score)
        self.hits = 1
        self.missed = 0
        self.confirmed = False

        self.confirm_hits = 0
        self.confirm_missed = 0

    def update(self, x, y, score):
        self.x = EMA_ALPHA * float(x) + (1.0 - EMA_ALPHA) * self.x
        self.y = EMA_ALPHA * float(y) + (1.0 - EMA_ALPHA) * self.y
        self.score = EMA_ALPHA * float(score) + (1.0 - EMA_ALPHA) * self.score
        self.hits += 1
        self.missed = 0

    def update_confirmation(self, is_confirmed):
        if is_confirmed:
            self.confirm_hits += 1
            self.confirm_missed = 0
            if self.confirm_hits >= CONFIRM_HITS_ON:
                self.confirmed = True
        else:
            self.confirm_hits = 0
            self.confirm_missed += 1
            if self.confirm_missed >= CONFIRM_MISSES_OFF:
                self.confirmed = False

    def predict(self):
        self.missed += 1
        self.confirm_missed += 1
        if self.confirm_missed >= CONFIRM_MISSES_OFF:
            self.confirmed = False

    @property
    def visible(self):
        return self.hits >= TRACK_MIN_HITS and self.missed <= TRACK_MAX_MISS



class FusedDetector(Node):
    def __init__(self):
        super().__init__("fused_detector")

        self.get_logger().info("Loading Model...")
        self.detector = Detector(
            CKPT_PATH,
            gpu=False,
            stride=MODEL_STRIDE,
            panoramic_scan=True,
        )
        self.fov_set = False
        self.get_logger().info(f"Ready (stride={MODEL_STRIDE})")

        self.yolo = None
        self.infer_request = None
        self.input_name = None

        try:
            core = ov.Core()
            model = core.read_model(YOLO_MODEL)
            compiled = core.compile_model(
                model,
                "CPU",
                {
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": str(OV_CPU_THREADS),
                }
            )
            self.yolo = compiled
            self.infer_request = compiled.create_infer_request()
            self.input_name = compiled.input(0)
            self.get_logger().info(f"YOLO OpenVINO ready ({OV_CPU_THREADS} thread)")
        except Exception as e:
            self.get_logger().error(f"YOLO load failed: {e}")


        self.fx = (IMG_W / 2.0) / np.tan(np.deg2rad(CAM_HFOV_DEG / 2.0))
        self.fy = (IMG_H / 2.0) / np.tan(np.deg2rad(CAM_VFOV_DEG / 2.0))
        self.cx = IMG_W / 2.0
        self.cy = IMG_H / 2.0
        self.bridge = CvBridge()


        self.tracks = []
        self.last_yolo_boxes = []
        self.latest_img_msg = None
        self.frame_count = 0

        self.last_lidar_count = 0
        self.last_confirmed_count = 0


        self._last_process_t = 0.0
        self._last_yolo_t = 0.0
        self._last_marker_t = 0.0
        self._last_log_t = 0.0


        self._shutdown = False
        self._yolo_lock = threading.Lock()
        self._yolo_event = threading.Event()
        self._yolo_img_msg = None


        angles = np.linspace(0.0, 2.0 * np.pi, CIRCLE_SEGMENTS, endpoint=False)
        self._circle_offsets = [(CIRCLE_RADIUS * np.cos(a), CIRCLE_RADIUS * np.sin(a)) for a in angles]


        self.marker_pub = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)
        self.scan_sub = self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_cb, 10)
        self.image_sub = self.create_subscription(Image, IMAGE_TOPIC, self.image_cb, 10)


        self._yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self._yolo_thread.start()


        self.get_logger().info(
            f"Ready | process={PROCESS_HZ}Hz yolo={YOLO_HZ}Hz marker={MARKER_HZ}Hz "
            f"stride={MODEL_STRIDE} ov_threads={OV_CPU_THREADS} "
            f"circle={SHOW_CIRCLE} label={SHOW_LABEL}"
        )


    def image_cb(self, img_msg):
        self.latest_img_msg = img_msg


    def scan_cb(self, scan_msg):
        now = time.monotonic()
        if (now - self._last_process_t) < (1.0 / PROCESS_HZ):
            return
        self._last_process_t = now
        self.frame_count += 1


        if not self.fov_set:
            fov_deg = np.degrees(scan_msg.angle_max - scan_msg.angle_min)
            self.detector.set_laser_fov(fov_deg)
            self.fov_set = True


        scan = np.asarray(scan_msg.ranges, dtype=np.float32)
        np.nan_to_num(scan, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


        try:
            dets_xy, dets_cls, _ = self.detector(scan)
        except Exception as e:
            self.get_logger().error(f"failed: {e}")
            return


        if dets_xy is None or len(dets_xy) == 0:
            self.last_lidar_count = 0
            self.last_confirmed_count = 0
            self._predict_tracks_only()
            self._prune_tracks()
            self._publish_markers_if_due(scan_msg, force_clear=True)
            self._log_if_due(0)
            return


        mask = dets_cls > CONF_THRESH
        dets_xy = dets_xy[mask]
        dets_cls = dets_cls[mask]


        if len(dets_xy) == 0:
            self.last_lidar_count = 0
            self.last_confirmed_count = 0
            self._predict_tracks_only()
            self._prune_tracks()
            self._publish_markers_if_due(scan_msg, force_clear=True)
            self._log_if_due(0)
            return


        roi = (
            (dets_xy[:, 0] >= ROI_X_MIN) & (dets_xy[:, 0] <= ROI_X_MAX) &
            (dets_xy[:, 1] >= ROI_Y_MIN) & (dets_xy[:, 1] <= ROI_Y_MAX)
        )
        dets_xy = dets_xy[roi]
        dets_cls = dets_cls[roi]


        if len(dets_xy) == 0:
            self.last_lidar_count = 0
            self.last_confirmed_count = 0
            self._predict_tracks_only()
            self._prune_tracks()
            self._publish_markers_if_due(scan_msg, force_clear=True)
            self._log_if_due(0)
            return


        dets_xy, dets_cls = self._merge_leg_pairs(dets_xy, dets_cls)


        if self.yolo is not None and self.latest_img_msg is not None:
            if (now - self._last_yolo_t) >= (1.0 / YOLO_HZ):
                self._last_yolo_t = now
                with self._yolo_lock:
                    self._yolo_img_msg = self.latest_img_msg
                    self._yolo_event.set()


        with self._yolo_lock:
            yolo_boxes = list(self.last_yolo_boxes)


        confirmed = self._confirm_with_yolo(dets_xy, yolo_boxes)

        self.last_lidar_count = len(dets_xy)
        self.last_confirmed_count = int(np.count_nonzero(confirmed))

        self._update_tracks(dets_xy, dets_cls, confirmed)


        self._publish_markers_if_due(scan_msg, force_clear=False)
        self._log_if_due(len(yolo_boxes))


    def _yolo_worker(self):
        while not self._shutdown:
            self._yolo_event.wait()
            self._yolo_event.clear()


            if self._shutdown or self.yolo is None:
                continue


            with self._yolo_lock:
                img_msg = self._yolo_img_msg
                self._yolo_img_msg = None


            if img_msg is None:
                continue


            try:
                img_cv = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                boxes = self._run_yolo(img_cv)
                with self._yolo_lock:
                    self.last_yolo_boxes = boxes
            except Exception as e:
                self.get_logger().warn(f"YOLO worker error: {e}")


    def _predict_tracks_only(self):
        for t in self.tracks:
            t.predict()


    def _prune_tracks(self):
        self.tracks = [t for t in self.tracks if t.missed <= TRACK_MAX_MISS]


    def _update_tracks(self, dets_xy, dets_cls, confirmed):
        for t in self.tracks:
            t.predict()


        matched_tracks = set()


        for j in range(len(dets_xy)):
            xj = float(dets_xy[j][0])
            yj = float(dets_xy[j][1])


            best_i = -1
            best_d = 1.0


            for i, t in enumerate(self.tracks):
                if i in matched_tracks:
                    continue
                d = np.hypot(xj - t.x, yj - t.y)
                if d < best_d:
                    best_d = d
                    best_i = i


            if best_i >= 0:
                t = self.tracks[best_i]
                t.update(xj, yj, dets_cls[j])
                t.update_confirmation(bool(confirmed[j]))
                matched_tracks.add(best_i)
            else:
                nt = Track(xj, yj, dets_cls[j])
                nt.update_confirmation(bool(confirmed[j]))
                self.tracks.append(nt)


        self._prune_tracks()


    def _merge_leg_pairs(self, dets_xy, dets_cls):
        if len(dets_xy) == 0:
            return dets_xy, dets_cls


        n = len(dets_xy)
        merged = np.zeros(n, dtype=bool)
        out_xy = []
        out_cls = []


        for i in range(n):
            if merged[i]:
                continue


            xi, yi = dets_xy[i]
            best_j = -1
            best_d = MERGE_DIST


            for j in range(i + 1, n):
                if merged[j]:
                    continue
                d = np.hypot(xi - dets_xy[j][0], yi - dets_xy[j][1])
                if d < best_d:
                    best_d = d
                    best_j = j


            if best_j >= 0:
                out_xy.append([
                    0.5 * (dets_xy[i][0] + dets_xy[best_j][0]),
                    0.5 * (dets_xy[i][1] + dets_xy[best_j][1]),
                ])
                out_cls.append(max(dets_cls[i], dets_cls[best_j]))
                merged[i] = True
                merged[best_j] = True
            else:
                out_xy.append([dets_xy[i][0], dets_xy[i][1]])
                out_cls.append(dets_cls[i])
                merged[i] = True


        return np.asarray(out_xy, dtype=np.float32), np.asarray(out_cls, dtype=np.float32)


    def _confirm_with_yolo(self, dets_xy, yolo_boxes):
        if len(dets_xy) == 0 or not yolo_boxes:
            return np.zeros((len(dets_xy),), dtype=bool)


        confirmed = np.zeros((len(dets_xy),), dtype=bool)


        for i, xy in enumerate(dets_xy):
            u, v = self._project_to_image(xy[0], xy[1])
            if not (0 <= u < IMG_W and 0 <= v < IMG_H):
                continue


            for (x1, y1, x2, y2) in yolo_boxes:
                y_low = y1 + (y2 - y1) * 0.4
                if x1 < u < x2 and y_low < v < y2:
                    confirmed[i] = True
                    break


        return confirmed


    def _run_yolo(self, img_cv):
        blob = cv2.resize(img_cv, (640, 640), interpolation=cv2.INTER_LINEAR)
        blob = blob[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]


        self.infer_request.infer({self.input_name: blob})
        preds = self.infer_request.get_output_tensor(0).data[0].T


        person_scores = preds[:, 4 + PERSON_CLASS]
        mask = person_scores > YOLO_CONF
        preds_f = preds[mask]
        scores_f = person_scores[mask]


        if len(preds_f) == 0:
            return []


        sx = IMG_W / 640.0
        sy = IMG_H / 640.0
        boxes_xywh = []
        boxes_xyxy = []


        for pred in preds_f:
            cx_n, cy_n, w_n, h_n = pred[:4]
            x1 = float((cx_n - w_n / 2.0) * sx)
            y1 = float((cy_n - h_n / 2.0) * sy)
            x2 = float((cx_n + w_n / 2.0) * sx)
            y2 = float((cy_n + h_n / 2.0) * sy)
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            boxes_xyxy.append([x1, y1, x2, y2])


        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_f.tolist(), YOLO_CONF, NMS_IOU)
        if indices is None or len(indices) == 0:
            return []


        out = []
        for idx in np.array(indices).flatten():
            x1, y1, x2, y2 = boxes_xyxy[idx]
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(IMG_W, int(x2))
            y2 = min(IMG_H, int(y2))
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
        return out


    def _project_to_image(self, lx, ly):
        cam_z = float(lx) + LIDAR_BEHIND
        cam_x = -float(ly)
        cam_y = CAM_HEIGHT - LIDAR_HEIGHT


        if cam_z <= 0.0:
            return -1, -1


        u = int(self.fx * (cam_x / cam_z) + self.cx)
        v = int(self.fy * (cam_y / cam_z) + self.cy)
        return u, v


    def _publish_markers_if_due(self, scan_msg, force_clear=False):
        now = time.monotonic()
        if not force_clear and (now - self._last_marker_t) < (1.0 / MARKER_HZ):
            return


        self._last_marker_t = now
        visible_tracks = [t for t in self.tracks if t.visible]


        if force_clear or len(visible_tracks) == 0:
            self.marker_pub.publish(self._clear_markers())
            return


        self.marker_pub.publish(self._build_markers(scan_msg, visible_tracks))


    def _build_markers(self, scan_msg, visible_tracks):
        ma = MarkerArray()


        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)


        ma.markers.append(self._roi_box(scan_msg))
        lifetime = rclpy.duration.Duration(seconds=1.3).to_msg()


        for t in visible_tracks:
            r, g, b = (0.0, 1.0, 0.0) if t.confirmed else (1.0, 1.0, 0.0)


            if SHOW_CIRCLE:
                circle = Marker()
                circle.header = scan_msg.header
                circle.ns = "circle"
                circle.id = t.id
                circle.type = Marker.LINE_STRIP
                circle.action = Marker.ADD
                circle.pose.orientation.w = 1.0
                circle.scale.x = 0.03
                circle.color.r = r
                circle.color.g = g
                circle.color.b = b
                circle.color.a = 1.0
                circle.lifetime = lifetime


                for dx, dy in self._circle_offsets:
                    p = Point()
                    p.x = t.x + dx
                    p.y = t.y + dy
                    p.z = 0.0
                    circle.points.append(p)


                p0 = Point()
                p0.x = t.x + self._circle_offsets[0][0]
                p0.y = t.y + self._circle_offsets[0][1]
                p0.z = 0.0
                circle.points.append(p0)
                ma.markers.append(circle)


            if SHOW_DOT:
                dot = Marker()
                dot.header = scan_msg.header
                dot.ns = "dot"
                dot.id = t.id
                dot.type = Marker.SPHERE
                dot.action = Marker.ADD
                dot.pose.orientation.w = 1.0
                dot.pose.position.x = t.x
                dot.pose.position.y = t.y
                dot.pose.position.z = 0.0
                dot.scale.x = 0.14
                dot.scale.y = 0.14
                dot.scale.z = 0.14
                dot.color.r = r
                dot.color.g = g
                dot.color.b = b
                dot.color.a = 1.0
                dot.lifetime = lifetime
                ma.markers.append(dot)


            if SHOW_LABEL:
                label = Marker()
                label.header = scan_msg.header
                label.ns = "label"
                label.id = t.id
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.orientation.w = 1.0
                label.pose.position.x = t.x
                label.pose.position.y = t.y
                label.pose.position.z = 0.65
                label.scale.z = 0.20
                label.color.r = 1.0
                label.color.g = 1.0
                label.color.b = 1.0
                label.color.a = 1.0
                label.lifetime = lifetime
                label.text = f"ID{t.id} {'HUMAN' if t.confirmed else '?'}"
                ma.markers.append(label)


        return ma


    def _roi_box(self, scan_msg):
        m = Marker()
        m.header = scan_msg.header
        m.ns = "roi"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.03
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.35
        m.lifetime = rclpy.duration.Duration(seconds=1.3).to_msg()


        pts = [
            (ROI_X_MIN, ROI_Y_MIN),
            (ROI_X_MAX, ROI_Y_MIN),
            (ROI_X_MAX, ROI_Y_MAX),
            (ROI_X_MIN, ROI_Y_MAX),
            (ROI_X_MIN, ROI_Y_MIN),
        ]
        for px, py in pts:
            p = Point()
            p.x = px
            p.y = py
            p.z = 0.0
            m.points.append(p)

        return m


    def _clear_markers(self):
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        return ma


    def _print_status_table(self, yolo_box_count):
        visible_tracks = [t for t in self.tracks if t.visible]
        confirmed_tracks = [t for t in visible_tracks if t.confirmed]

        print("\n" + "=" * 92)
        print(
            f"FRAME {self.frame_count:04d} | "
            f"YOLO_BOXES={yolo_box_count:<2d} | "
            f"LIDAR_DETS={self.last_lidar_count:<2d} | "
            f"YOLO_CONFIRMED_DETS={self.last_confirmed_count:<2d} | "
            f"VISIBLE_TRACKS={len(visible_tracks):<2d} | "
            f"GREEN_TRACKS={len(confirmed_tracks):<2d}"
        )
        print("-" * 92)
        print(f"{'ID':<4} {'STATE':<10} {'X(m)':>8} {'Y(m)':>8} {'SCORE':>8} {'HITS':>6} {'MISS':>6}")
        print("-" * 92)

        if len(visible_tracks) == 0:
            print("No visible tracks")
        else:
            for t in visible_tracks:
                state = "VALID-HUMAN" if t.confirmed else "UN-CONFIRM"
                print(
                    f"{t.id:<4} "
                    f"{state:<10} "
                    f"{t.x:>8.2f} "
                    f"{t.y:>8.2f} "
                    f"{t.score:>8.2f} "
                    f"{t.hits:>6d} "
                    f"{t.missed:>6d}"
                )

        print("=" * 92, flush=True)


    def _log_if_due(self, yolo_box_count):
        now = time.monotonic()
        if (now - self._last_log_t) < (1.0 / LOG_HZ):
            return


        visible_count = sum(1 for t in self.tracks if t.visible)
        self.get_logger().info(
            f"F{self.frame_count:04d} visible={visible_count} total_tracks={len(self.tracks)} "
            f"yolo_boxes={yolo_box_count} lidar_dets={self.last_lidar_count} "
            f"confirmed_dets={self.last_confirmed_count}"
        )
        self._print_status_table(yolo_box_count)
        self._last_log_t = now


    def destroy_node(self):
        self._shutdown = True
        self._yolo_event.set()
        super().destroy_node()



def main():
    rclpy.init()
    node = FusedDetector()
    executor = SingleThreadedExecutor()
    executor.add_node(node)


    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()