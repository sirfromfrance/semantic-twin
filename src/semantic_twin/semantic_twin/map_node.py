#!/usr/bin/env python3
"""Height map of the environment, built from the drone's depth images.

"""
import bisect
import os
import time
import json
import math
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from std_msgs.msg import String
from cv_bridge import CvBridge

from semantic_twin import colour
from semantic_twin.integration_node import quat_to_mat, slerp


class MapNode(Node):
    def __init__(self):
        super().__init__('sdt_map_node')
        P = self.declare_parameter
        P('depth_topic', '/uav0/camera/depth/image_raw')
        P('camera_info_topic', '/uav0/camera/depth/camera_info')
        P('pose_topic', '/uav0/mavros/local_position/pose')
        P('camera_optical_frame', 'camera_optical_frame')
        P('map_frame', 'map')
        P('resolution', 0.5)
        P('x_min', -40.0)
        P('x_max', 40.0)
        P('y_min', -30.0)
        P('y_max', 30.0)
        P('pixel_stride', 6)
        P('min_depth', 1.0)
        P('max_depth', 25.0)
        P('min_period', 0.5)          
        P('blocked_height', 1.0)      
        P('min_points', 2)            
        P('export_path', '/tmp/sdt_height_map.json')
        P('rgb_topic', '/uav0/camera/image_raw')
        P('rgb_max_age', 0.15)         # s between the depth image and the colour image used for it
        P('dump_dir', '')              # if set: save a camera frame (with its depth) every dump_period s, to tune the colour thresholds
        P('dump_period', 15.0)
        P('veg_min_height', 1.0)       # only points above this count for the foliage test
        P('fire_max_height', 0.7)      # flames sit at ground level: only points below this count as fire
        P('veg_sat_min', colour.VEG_SAT_MIN)
        P('veg_val_min', colour.VEG_VAL_MIN)
        P('fire_sat_min', colour.FIRE_SAT_MIN)
        P('fire_val_min', colour.FIRE_VAL_MIN)

        g = lambda n: self.get_parameter(n).value
        self.res = float(g('resolution'))
        self.x0, self.y0 = float(g('x_min')), float(g('y_min'))
        self.nx = int(round((float(g('x_max')) - self.x0) / self.res))
        self.ny = int(round((float(g('y_max')) - self.y0) / self.res))
        self.stride = int(g('pixel_stride'))
        self.min_depth, self.max_depth = float(g('min_depth')), float(g('max_depth'))
        self.min_period = float(g('min_period'))
        self.blocked_height = float(g('blocked_height'))
        self.min_points = int(g('min_points'))
        self.export_path = g('export_path')
        self.rgb_max_age = float(g('rgb_max_age'))
        self.dump_dir, self.dump_period, self.last_dump = g('dump_dir'), float(g('dump_period')), -1e9
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
        self.veg_min_height, self.fire_max_height = float(g('veg_min_height')), float(g('fire_max_height'))
        self.veg_kw = dict(sat_min=float(g('veg_sat_min')), val_min=float(g('veg_val_min')))
        self.fire_kw = dict(sat_min=float(g('fire_sat_min')), val_min=float(g('fire_val_min')))
        self.map_frame = g('map_frame')
        self.optical_frame = g('camera_optical_frame')

        self.max_z = np.full((self.ny, self.nx), -np.inf)
        self.count = np.zeros((self.ny, self.nx), dtype=np.int32)
        self.hi = np.zeros((self.ny, self.nx), dtype=np.int32)     # points above veg_min_height
        self.veg = np.zeros((self.ny, self.nx), dtype=np.int32)    # ...of which foliage-coloured
        self.fire = np.zeros((self.ny, self.nx), dtype=np.int32)   # fire-coloured points near the ground
        self.rgb_buf = deque(maxlen=8)

        self.bridge = CvBridge()
        self.K = None
        self.pose_hist = deque(maxlen=800)
        self.cam_to_base = None
        self.last_t = -1e9
        self.last_debug = -1e9
       
        self.pending = deque(maxlen=30)
        self.pose_wait = 0.5           # s: frames further ahead of the newest pose than this are dropped
        self.stats = dict(frames=0, used=0, no_pose=0, waited=0, points=0, out_of_grid=0, no_rgb=0, cb_ms=0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, g('camera_info_topic'), self.info_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, g('pose_topic'), self.pose_cb, qos_profile_sensor_data)
        self.create_subscription(Image, g('depth_topic'), self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, g('rgb_topic'), self.rgb_cb, qos_profile_sensor_data)
        self.debug_pub = self.create_publisher(Image, 'semantic_map/colour_debug', 1)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.occ_pub = self.create_publisher(OccupancyGrid, 'semantic_map/occupancy', latched)
        self.create_service(Trigger, '~/export_grid', self.export_cb)
        self.grid_json_pub = self.create_publisher(String, 'semantic_map/grid_json', 1)
        self.create_timer(5.0, self.publish_grid_json)
        self.create_timer(2.0, self.publish_occupancy)
        self.create_timer(10.0, self.log_stats)
        self.get_logger().info(
            f'Map node ready: grid {self.nx}x{self.ny} cells of {self.res} m, '
            f'x[{self.x0},{self.x0 + self.nx * self.res}] y[{self.y0},{self.y0 + self.ny * self.res}]')

    def pose_cb(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9     # when the pose was measured, not when this callback ran
        self.pose_hist.append((st if st > 0 else self.get_clock().now().nanoseconds * 1e-9,
                               np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])))
        while self.pending and self.pending[0][0] <= self.pose_hist[-1][0]:
            self.depth_cb(self.pending.popleft()[1], retry=True)

    def pose_at(self, t):
        hist = list(self.pose_hist)
        if len(hist) < 2:
            return None
        times = [h[0] for h in hist]
        if t < times[0] or t > times[-1]:
            return None
        i = max(1, bisect.bisect_left(times, t))
        (t0, p0, q0), (t1, p1, q1) = hist[i - 1], hist[i]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return quat_to_mat(slerp(q0, q1, a)), p0 + a * (p1 - p0)

    def get_cam_to_base(self):
        if self.cam_to_base is None:
            try:
                tf = self.tf_buffer.lookup_transform('base_link', self.optical_frame, Time(),
                                                     timeout=Duration(seconds=0.2))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                return None
            r, t = tf.transform.rotation, tf.transform.translation
            self.cam_to_base = (quat_to_mat([r.x, r.y, r.z, r.w]), np.array([t.x, t.y, t.z]))
        return self.cam_to_base

    def rgb_cb(self, msg):
        self.rgb_buf.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, msg))

    def rgb_at(self, t):
        if not self.rgb_buf:
            return None
        ts, msg = min(self.rgb_buf, key=lambda b: abs(b[0] - t))
        if abs(ts - t) > self.rgb_max_age:
            return None
        return self.bridge.imgmsg_to_cv2(msg, 'rgb8'), msg

    def info_cb(self, msg):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])   # fx fy cx cy

    def depth_cb(self, msg, retry=False):
        t0 = time.perf_counter()
        try:
            self._depth_cb(msg, retry)
        finally:
            self.stats['cb_ms'] = max(self.stats.get('cb_ms', 0), int(1000 * (time.perf_counter() - t0)))

    def _depth_cb(self, msg, retry=False):
        if not retry:
            self.stats['frames'] += 1
        if self.K is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t - self.last_t < self.min_period:
            return
        pose, c2b = self.pose_at(t), self.get_cam_to_base()
        if (pose is None and not retry and self.pose_hist
                and 0 < t - self.pose_hist[-1][0] <= self.pose_wait):
            self.pending.append((t, msg))      # newer than the newest pose: retry when the next pose arrives
            self.stats['waited'] += 1
            return
        if pose is None or c2b is None:
            self.stats['no_pose'] += 1
            return
        self.last_t = t

        depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough').astype(np.float32)
        h, w = depth.shape[:2]
        s = self.stride
        vs, us = np.mgrid[s // 2:h:s, s // 2:w:s]
        z = depth[vs, us]
        ok = np.isfinite(z) & (z > self.min_depth) & (z < self.max_depth)
        if not ok.any():
            return
        u, v, z = us[ok].astype(np.float64), vs[ok].astype(np.float64), z[ok].astype(np.float64)
        fx, fy, cx, cy = self.K
        opt = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)       # optical frame
        r_mb, t_mb = pose
        r_bc, t_bc = c2b
        r_mo, t_mo = r_mb @ r_bc, r_mb @ t_bc + t_mb
        pts = opt @ r_mo.T + t_mo

        ix = np.floor((pts[:, 0] - self.x0) / self.res).astype(int)
        iy = np.floor((pts[:, 1] - self.y0) / self.res).astype(int)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        self.stats['out_of_grid'] += int((~inside).sum())
        ix, iy, pz = ix[inside], iy[inside], pts[inside, 2]
        np.maximum.at(self.max_z, (iy, ix), pz)
        np.add.at(self.count, (iy, ix), 1)
        rgb = self.rgb_at(t)
        if rgb is None:
            self.stats['no_rgb'] += 1
        else:
            img, rmsg = rgb
            sy, sx = img.shape[0] / h, img.shape[1] / w
            px = img[np.minimum((vs[ok][inside] * sy).astype(int), img.shape[0] - 1),
                     np.minimum((us[ok][inside] * sx).astype(int), img.shape[1] - 1)]
            if self.dump_dir and t - self.last_dump >= self.dump_period:
                self.last_dump = t
                r_mb0 = pose[0]
                stamp = f'{t:.1f}'
                cv2.imwrite(f'{self.dump_dir}/rgb_{stamp}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                np.save(f'{self.dump_dir}/depth_{stamp}.npy', depth.astype(np.float16))
                np.save(f'{self.dump_dir}/pose_{stamp}.npy', np.concatenate([pose[1], np.array(pose[0]).ravel()]))
            high = pz >= self.veg_min_height
            np.add.at(self.hi, (iy, ix), high.astype(np.int32))
            np.add.at(self.veg, (iy, ix), (high & colour.vegetation(px, **self.veg_kw)).astype(np.int32))
            np.add.at(self.fire, (iy, ix), ((pz <= self.fire_max_height) & colour.fire(px, **self.fire_kw)).astype(np.int32))
            if self.debug_pub.get_subscription_count() and t - self.last_debug > 1.0:
                self.last_debug = t
                over = img.copy()
                over[colour.vegetation(img, **self.veg_kw)] = (0, 255, 0)
                over[colour.fire(img, **self.fire_kw)] = (255, 0, 255)
                out = self.bridge.cv2_to_imgmsg(((img.astype(np.float32) + over) / 2).astype(np.uint8), 'rgb8')
                out.header = rmsg.header
                self.debug_pub.publish(out)
        self.stats['used'] += 1
        self.stats['points'] += int(inside.sum())

    def classes(self):
        cls = np.full((self.ny, self.nx), -1, dtype=np.int8)
        seen = self.count >= self.min_points
        cls[seen] = 0
        cls[seen & (self.max_z >= 0.5)] = 1
        cls[seen & (self.max_z >= self.blocked_height)] = 2
        return cls

    def publish_occupancy(self):
        cls = self.classes()
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.res
        msg.info.width, msg.info.height = self.nx, self.ny
        msg.info.origin.position.x, msg.info.origin.position.y = self.x0, self.y0
        msg.info.origin.orientation.w = 1.0
        occ = np.full(cls.shape, -1, dtype=np.int8)
        occ[cls == 0] = 0
        occ[cls == 1] = 40
        occ[cls == 2] = 100
        msg.data = occ.flatten().tolist()
        self.occ_pub.publish(msg)

    def log_stats(self):
        cls = self.classes()
        self.get_logger().info(
            'last 10 s: frames={frames} used={used} no_pose={no_pose} waited={waited} points={points} out_of_grid={out_of_grid} no_rgb={no_rgb} slowest_callback_ms={cb_ms} | '
            .format(**self.stats) +
            f'observed cells={int((cls >= 0).sum())} obstacle cells={int((cls == 2).sum())}')
        for k in self.stats:
            self.stats[k] = 0

    def grid_dict(self):
        cls = self.classes()
        z = np.where(self.count >= self.min_points, np.round(self.max_z, 2), np.nan)
        return {
            'resolution': self.res, 'x0': self.x0, 'y0': self.y0, 'nx': self.nx, 'ny': self.ny,
            'blocked_height': self.blocked_height, 'min_points': self.min_points,
            'max_z': [[None if np.isnan(v) else float(v) for v in row] for row in z],
            'observed_cells': int((cls >= 0).sum()),
            'count': self.count.tolist(), 'hi': self.hi.tolist(), 'veg': self.veg.tolist(), 'fire': self.fire.tolist(),
        }

    def publish_grid_json(self):
        if (self.count >= self.min_points).any():
            self.grid_json_pub.publish(String(data=json.dumps(self.grid_dict())))

    def export_cb(self, request, response):
        data = self.grid_dict()
        try:
            with open(self.export_path, 'w') as f:
                json.dump(data, f)
            response.success = True
            response.message = f'Exported {data["observed_cells"]} observed cells to {self.export_path}'
        except OSError as e:
            response.success, response.message = False, str(e)
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
