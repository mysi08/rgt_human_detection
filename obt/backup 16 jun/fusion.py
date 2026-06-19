#!/usr/bin/env python3
import cv2
import numpy as np
import math
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class CentroidTracker:
    def __init__(self, max_dist=1.5, max_age=10):
        self.next_id = 1
        self.tracks = {}  
        self.max_dist = max_dist
        self.max_age = max_age

    def _generate_color(self, track_id):
        distinct_colors = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 1.0), (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.5, 0.0), (0.5, 1.0, 0.0),
            (0.5, 0.0, 1.0), (0.0, 0.5, 0.5), (1.0, 0.4, 0.7)
        ]
        r, g, b = distinct_colors[track_id % len(distinct_colors)]
        return (int(b * 255), int(g * 255), int(r * 255)), (r, g, b)

    def update(self, current_persons):
        assigned_ids = set()
        
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1

        for person in current_persons:
            cx, cy = person['centroid_x'], person['centroid_y']
            best_id, best_dist = None, float('inf')

            for tid, track in self.tracks.items():
                if tid in assigned_ids: continue
                tcx = track['person_data']['centroid_x']
                tcy = track['person_data']['centroid_y']
                dist = math.hypot(cx - tcx, cy - tcy)
                
                if dist < self.max_dist and dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                self.tracks[best_id]['age'] = 0
                self.tracks[best_id]['person_data'] = person
                assigned_ids.add(best_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                c_cv, c_rviz = self._generate_color(new_id)
                self.tracks[new_id] = {'age': 0, 'person_data': person, 'color_cv': c_cv, 'color_rviz': c_rviz}

        self.tracks = {k: v for k, v in self.tracks.items() if v['age'] <= self.max_age}

        output_persons = []
        for tid, t in self.tracks.items():
            p = t['person_data']
            p['track_id'] = tid
            p['name'] = f"ID_{tid}"
            p['color_cv'] = t['color_cv']
            p['color_rviz'] = t['color_rviz']
            p['age'] = t['age']
            output_persons.append(p)
            
        return output_persons


def cluster_by_adaptive_jump(scan, cfg, cached_angles):
    ranges = np.array(scan.ranges, dtype=np.float32)
    delta_theta = scan.angle_increment
    c_noise = cfg.get('jump_noise_floor', 0.06) 

    depth_mask = (ranges >= cfg['human_dist_min']) & (ranges <= cfg['human_dist_max']) & np.isfinite(ranges)
    if not np.any(depth_mask):
        return []

    idx_sub = np.where(depth_mask)[0]
    r_sub = ranges[depth_mask]
    angles_sub = cached_angles[depth_mask]

    y_sub = r_sub * np.sin(angles_sub)
    lateral_mask = np.abs(y_sub) <= 1.0 

    if not np.any(lateral_mask):
        return []

    r = r_sub[lateral_mask]
    angles = angles_sub[lateral_mask]
    idx = idx_sub[lateral_mask]
    x = r * np.cos(angles)
    y = y_sub[lateral_mask]

    clusters = []
    if len(r) == 0:
        return clusters

    current_cluster = {'x': [x[0]], 'y': [y[0]], 'r': [r[0]], 'idx': [idx[0]]}

    for i in range(1, len(r)):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        actual_dist = math.hypot(dx, dy)

        mean_range = (r[i] + r[i-1]) / 2.0
        adaptive_threshold = mean_range * math.tan(delta_theta) + c_noise

        if actual_dist <= adaptive_threshold:
            current_cluster['x'].append(x[i])
            current_cluster['y'].append(y[i])
            current_cluster['r'].append(r[i])
            current_cluster['idx'].append(idx[i])
        else:
            if len(current_cluster['x']) >= cfg.get('min_human_points', 3):
                clusters.append(current_cluster)
            current_cluster = {'x': [x[i]], 'y': [y[i]], 'r': [r[i]], 'idx': [idx[i]]}

    if len(current_cluster['x']) >= cfg.get('min_human_points', 3):
        clusters.append(current_cluster)

    for c in clusters:
        c['x'] = np.array(c['x'], dtype=np.float32)
        c['y'] = np.array(c['y'], dtype=np.float32)
        c['r'] = np.array(c['r'], dtype=np.float32)
        c['idx'] = np.array(c['idx'], dtype=np.int32)

    return clusters

def filter_clusters_geometry(clusters, cfg):
    """
    UPO/Academic Geometric Filter: Uses Compactness and Spread to destroy
    hollow curves (office chairs/desks) and strictly keep dense cylinders (legs).
    """
    valid_legs = []
    for c in clusters:
        x, y = c['x'], c['y']
        pts = len(x)

        if pts < cfg.get('min_human_points', 3) or pts > cfg.get('max_human_points', 60):
            continue

        # 1. STRICT WIDTH LIMIT: A single human leg/calf is never wider than 25cm.
        width = math.hypot(x[-1] - x[0], y[-1] - y[0])
        if width < 0.03 or width > 0.25:
            continue

        cx = float(np.mean(x))
        cy = float(np.mean(y))

        # 2. COMPACTNESS (DENSITY) CHECK:
        # Calculate the average distance of all points to the cluster's centroid.
        # Legs are solid tight circles. Desks/chairs are hollow wide arcs.
        dists = np.hypot(x - cx, y - cy)
        mean_dist_to_center = float(np.mean(dists))

        # If the points are spread out, it is not a leg.
        if mean_dist_to_center > 0.10:
            continue

        c['centroid_x'] = cx
        c['centroid_y'] = cy
        c['min_x'], c['max_x'] = float(np.min(x)), float(np.max(x))
        c['min_y'], c['max_y'] = float(np.min(y)), float(np.max(y))
        c['width'] = float(width)
        c['points'] = pts

        valid_legs.append(c)

    return valid_legs

def associate_and_merge_legs(valid_legs, humans, w_img, h_img, cfg):
    """
    1. Vertical Box Clipping: Rejects top-half desk clutter.
    2. The 2-Leg Limit: Grabs exactly 1 or 2 best legs per YOLO box.
    3. Isolated Sweep: Grabs exactly 1 missing leg if only 1 was found.
    """
    confirmed_persons = []
    claimed_indices = set()

    fx = w_img / (2.0 * math.tan(math.radians(cfg['cam_hfov']) / 2.0))
    fy = h_img / (2.0 * math.tan(math.radians(cfg['cam_vfov']) / 2.0))
    cx_img, cy_img = w_img / 2.0, h_img / 2.0
    height_diff = cfg['cam_height'] - cfg['lidar_height']

    projs = []
    for c in valid_legs:
        cam_z = c['x'] + cfg['lidar_behind']
        mask = cam_z > 0.1
        if not np.any(mask):
            projs.append(None)
            continue
        u = (fx * (-c['y'][mask]) / cam_z[mask] + cx_img).astype(np.int32)
        v = (fy * height_diff / cam_z[mask] + cy_img).astype(np.int32)
        projs.append({
            'u_min': np.min(u), 'u_max': np.max(u), 'v_min': np.min(v), 'v_max': np.max(v),
            'area': max(1, (np.max(u)-np.min(u))*(np.max(v)-np.min(v))), 'c': c
        })

    for human in humans:
        bx1, by1, bx2, by2 = human["bbox"]
        box_h = by2 - by1
        matched_candidates = []

        for i, p in enumerate(projs):
            if p is None or i in claimed_indices: continue

            # --- SEMANTIC VERTICAL CLIPPING ---
            # If the cluster's highest point in the image is in the top 40% of the box,
            # it is a monitor/desk edge/chair back. Reject it immediately.
            if p['v_min'] < by1 + (box_h * 0.40):
                continue

            iu = max(0, min(p['u_max'], bx2) - max(p['u_min'], bx1))
            iv = max(0, min(p['v_max'], by2) - max(p['v_min'], by1))
            if iu > 0 and iv > 0:
                overlap = (iu * iv) / float(p['area'])
                if overlap > cfg.get('min_yolo_overlap', 0.4):
                    matched_candidates.append((overlap, i))

        if not matched_candidates:
            continue

        # --- THE STRICT TWO-LEG LIMIT ---
        # Sort by overlap percentage and keep a maximum of 2 clusters (legs)
        matched_candidates.sort(key=lambda x: x[0], reverse=True)
        best_leg_indices = [idx for (score, idx) in matched_candidates[:2]]
        
        claimed_indices.update(best_leg_indices)
        
        mx = [projs[i]['c']['x'] for i in best_leg_indices]
        my = [projs[i]['c']['y'] for i in best_leg_indices]
        mr = [projs[i]['c']['r'] for i in best_leg_indices]
        midx = [projs[i]['c']['idx'] for i in best_leg_indices]

        # --- ISOLATED NEIGHBOR SWEEP ---
        # If we only found ONE leg in the box, find the SINGLE closest neighbor outside it.
        if len(best_leg_indices) == 1:
            max_stride = cfg.get('max_stride_length', 0.8) 
            cx_temp = float(np.mean(mx))
            cy_temp = float(np.mean(my))
            
            best_orphan_idx = -1
            best_orphan_dist = max_stride

            for i, p in enumerate(projs):
                if p is None or i in claimed_indices: continue
                # We do NOT apply the vertical height check here, because a back leg 
                # might legitimately project slightly differently due to stride physics.
                dist = math.hypot(p['c']['centroid_x'] - cx_temp, p['c']['centroid_y'] - cy_temp)
                if dist < best_orphan_dist:
                    best_orphan_dist = dist
                    best_orphan_idx = i
            
            if best_orphan_idx != -1:
                mx.append(projs[best_orphan_idx]['c']['x'])
                my.append(projs[best_orphan_idx]['c']['y'])
                mr.append(projs[best_orphan_idx]['c']['r'])
                midx.append(projs[best_orphan_idx]['c']['idx'])
                claimed_indices.add(best_orphan_idx)

        # Final Construction
        cx_all = np.concatenate(mx)
        cy_all = np.concatenate(my)
        cr_all = np.concatenate(mr)
        cidx_all = np.concatenate(midx)

        c_x = float(np.mean(cx_all))
        c_y = float(np.mean(cy_all))

        person = {
            'bbox': human['bbox'], 'conf': human['conf'],
            'x': cx_all, 'y': cy_all, 'r': cr_all, 'idx': cidx_all,
            'centroid_x': c_x, 'centroid_y': c_y,
            'min_x': float(np.min(cx_all)), 'max_x': float(np.max(cx_all)),
            'min_y': float(np.min(cy_all)), 'max_y': float(np.max(cy_all)),
            'dist': math.hypot(c_x, c_y),
            'angle': math.degrees(math.atan2(c_y, c_x)),
            'points': len(cx_all)
        }
        person['side'] = "L" if person['angle'] > 0 else "R"
        confirmed_persons.append(person)

    return confirmed_persons

def process_frustum_fusion(scan, humans, w_img, h_img, cfg, cached_angles=None, tracker=None):
    if cached_angles is None or len(cached_angles) != len(scan.ranges):
        cached_angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment

    raw_clusters = cluster_by_adaptive_jump(scan, cfg, cached_angles)
    valid_legs = filter_clusters_geometry(raw_clusters, cfg)
    confirmed_persons = associate_and_merge_legs(valid_legs, humans, w_img, h_img, cfg)

    if tracker is not None:
        confirmed_persons = tracker.update(confirmed_persons)

    return confirmed_persons

def draw_combined_debug_view(img, scan, persons, w_img, h_img, cfg, cached_angles):
    ranges = np.array(scan.ranges, dtype=np.float32)
    valid_mask = (ranges > 0.1) & (ranges < cfg['max_range']) & np.isfinite(ranges)
    
    r = ranges[valid_mask]
    angles = cached_angles[valid_mask]
    indices = np.where(valid_mask)[0]

    cam_z = r * np.cos(angles) + cfg['lidar_behind']
    valid_z = cam_z > 0.1
    
    fx = w_img / (2.0 * math.tan(math.radians(cfg['cam_hfov']) / 2.0))
    fy = h_img / (2.0 * math.tan(math.radians(cfg['cam_vfov']) / 2.0))
    u = (fx * (-r[valid_z] * np.sin(angles[valid_z])) / cam_z[valid_z] + w_img/2.0).astype(np.int32)
    v = (fy * (cfg['cam_height'] - cfg['lidar_height']) / cam_z[valid_z] + h_img/2.0).astype(np.int32)
    proj_idx = indices[valid_z]

    point_colors = np.full((len(proj_idx), 3), (255, 255, 0), dtype=np.uint8)
    
    for person in persons:
        in_person = np.isin(proj_idx, person['idx'])
        point_colors[in_person] = person.get('color_cv', (0, 255, 0))

    img_mask = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
    if np.any(img_mask):
        img[v[img_mask], u[img_mask]] = point_colors[img_mask]

    for person in persons:
        x1, y1, x2, y2 = person['bbox']
        color = person.get('color_cv', (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if 'centroid_x' in person and 'centroid_y' in person:
            cam_z_c = person['centroid_x'] + cfg['lidar_behind']
            if cam_z_c > 0.1:
                cu = int(fx * (-person['centroid_y']) / cam_z_c + w_img/2.0)
                cv = int(fy * (cfg['cam_height'] - cfg['lidar_height']) / cam_z_c + h_img/2.0)
                cv2.drawMarker(img, (cu, cv), color, markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

        name = person.get('name', 'Human')
        cv2.putText(img, f"{name}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def publish_cluster_points(pub, clusters, frame_id, stamp):
    ma = MarkerArray()
    ma.markers.append(Marker(action=Marker.DELETEALL))
    for cluster in clusters:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "cluster_points"
        m.id = cluster.get('track_id', 0)
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.06
        
        c_r, c_g, c_b = cluster.get('color_rviz', (1.0, 0.0, 0.0))
        m.color.r, m.color.g, m.color.b, m.color.a = c_r, c_g, c_b, 1.0
        
        cx, cy = cluster['x'], cluster['y']
        m.points = [Point(x=float(cx[i]), y=float(cy[i]), z=0.0) for i in range(len(cx))]
        ma.markers.append(m)
    pub.publish(ma)

def publish_cluster_centroids(pub, clusters, frame_id, stamp):
    ma = MarkerArray()
    ma.markers.append(Marker(action=Marker.DELETEALL))
    for cluster in clusters:
        cyl = Marker()
        cyl.header.frame_id = frame_id
        cyl.header.stamp = stamp
        cyl.ns = "cluster_cylinders"
        cyl.id = cluster.get('track_id', 0)
        cyl.type = Marker.CYLINDER
        cyl.action = Marker.ADD
        cyl.pose.position.x = cluster['centroid_x']
        cyl.pose.position.y = cluster['centroid_y']
        cyl.pose.orientation.w = 1.0
        cyl.scale.x = cyl.scale.y = 0.15
        cyl.scale.z = 0.05
        
        c_r, c_g, c_b = cluster.get('color_rviz', (1.0, 0.0, 0.0))
        cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = c_r, c_g, c_b, 0.7
        ma.markers.append(cyl)
    pub.publish(ma)

def publish_3d_bboxes(pub, persons, frame_id, stamp, cfg):
    MAX_H, MIN_H = 0.3, 0.1   
    ma = MarkerArray()
    ma.markers.append(Marker(action=Marker.DELETEALL))

    h_img = cfg.get('h_img', 480)
    cy_img = h_img / 2.0
    fy = h_img / (2.0 * math.tan(math.radians(cfg['cam_vfov']) / 2.0))

    for person in persons:
        cx, cy, dist = person['centroid_x'], person['centroid_y'], person['dist']
        _, by1, _, by2 = person['bbox']
        color_rviz = person.get('color_rviz', (1.0, 0.0, 0.0))

        cam_depth = dist + cfg['lidar_behind']
        floor_row = cy_img + cfg['cam_height'] * fy / cam_depth
        head_row  = cy_img - (MAX_H - cfg['cam_height']) * fy / cam_depth

        by1_c = float(np.clip(by1, head_row, floor_row))
        by2_c = float(np.clip(by2, head_row, floor_row))

        pixel_height = by2_c - by1_c
        box_h = float(np.clip((pixel_height / fy) * cam_depth, MIN_H, MAX_H))

        min_x, max_x = person['min_x'], person['max_x']
        min_y, max_y = person['min_y'], person['max_y']

        C = [
            Point(x=min_x, y=min_y, z=0.0), Point(x=max_x, y=min_y, z=0.0),
            Point(x=min_x, y=max_y, z=0.0), Point(x=max_x, y=max_y, z=0.0),
            Point(x=min_x, y=min_y, z=box_h), Point(x=max_x, y=min_y, z=box_h),
            Point(x=min_x, y=max_y, z=box_h), Point(x=max_x, y=max_y, z=box_h),
        ]
        edges = [(0,1),(2,3),(4,5),(6,7), (0,2),(1,3),(4,6),(5,7), (0,4),(1,5),(2,6),(3,7)]
        
        wire = Marker()
        wire.header.frame_id = frame_id
        wire.header.stamp = stamp
        wire.ns = "bbox3d_wire"
        wire.id = person.get('track_id', 0)
        wire.type = Marker.LINE_LIST
        wire.action = Marker.ADD
        wire.pose.orientation.w = 1.0
        wire.scale.x = 0.04          
        wire.color.r, wire.color.g, wire.color.b = color_rviz
        wire.color.a = 1.0
        for a, b in edges:
            wire.points.extend([C[a], C[b]])
        ma.markers.append(wire)

        txt = Marker()
        txt.header.frame_id = frame_id
        txt.header.stamp = stamp
        txt.ns = "bbox3d_label"
        txt.id = person.get('track_id', 0)
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = cx - (max_x - min_x) * 0.5 
        txt.pose.position.y = cy
        txt.pose.position.z = box_h + 0.15   
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.16           
        txt.color.r, txt.color.g, txt.color.b = color_rviz
        txt.color.a = 1.0
        txt.text = f"{person.get('name', 'Human')} | D: {dist:.2f}m\nH: {box_h:.2f}m"
        ma.markers.append(txt)

    pub.publish(ma)

def create_human_pointcloud(clusters, frame_id, stamp):
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.is_dense = True

    if not clusters:
        msg.width = 0
        msg.row_step = 0
        msg.data = b''
        return msg
        
    x_vals = np.concatenate([c['x'] for c in clusters])
    y_vals = np.concatenate([c['y'] for c in clusters])
    z_vals = np.zeros_like(x_vals)
    
    points = np.column_stack((x_vals, y_vals, z_vals)).astype(np.float32)
    msg.width = len(points)
    msg.row_step = msg.point_step * len(points)
    msg.data = points.tobytes()
    return msg