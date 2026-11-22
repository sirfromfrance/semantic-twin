#!/usr/bin/env python3

import bisect
import json
import math
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, PoseStamped
from std_srvs.srv import Trigger
from std_msgs.msg import String

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


class SemanticCluster:

    def __init__(self, cluster_id, point, confidence, stamp):
        self.id = cluster_id
        self.x, self.y, self.z = point
        self.confidence = confidence
        self.count = 1
        self.first_seen = stamp
        self.last_seen = stamp
        self.ext_sum = 0.0   # sum of the vertical extents of the people-pixels
        self.ext_n = 0 #number of extent measurements

    def add_extent(self, extent):
        if extent is not None:
            self.ext_sum += extent
            self.ext_n += 1

    def mean_extent(self):
        return self.ext_sum / self.ext_n if self.ext_n else None

    def update(self, point, confidence, stamp):
        w = self.count
        self.x = (self.x * w + point[0]) / (w + 1)
        self.y = (self.y * w + point[1]) / (w + 1)
        self.z = (self.z * w + point[2]) / (w + 1)
        self.confidence = max(self.confidence, confidence)
        self.count += 1
        self.last_seen = stamp

    def distance_to(self, point):
        return math.hypot(self.x - point[0], self.y - point[1])


class IntegrationNode(Node):
    def __init__(self):
        super().__init__('sdt_integration_node')

        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('depth_topic', '/uav0/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/uav0/camera/depth/camera_info')
        self.declare_parameter('map_frame', 'map')
       
        self.declare_parameter('camera_optical_frame', 'camera_optical_frame')
        self.declare_parameter('merge_radius', 1.0)  # metres, horizontal
       
        self.declare_parameter('min_sightings', 3)
        self.declare_parameter('depth_patch_half', 3)
      
        self.declare_parameter('pose_topic', '/uav0/mavros/local_position/pose')
        self.declare_parameter('use_pose_history', True)
       
        self.declare_parameter('upright_min_extent', 0.45)
        self.declare_parameter('log_path', '/tmp/sdt_detections.csv')
        self.declare_parameter('min_confidence', 0.35)
        self.declare_parameter('marker_topic', 'semantic_map/victims')
        self.declare_parameter('export_path', '/tmp/sdt_semantic_map.json')
       
        self.declare_parameter('depth_scale_to_metres', 1.0)

        self.map_frame = self.get_parameter('map_frame').value
        self.camera_optical_frame = self.get_parameter('camera_optical_frame').value
        self.merge_radius = float(self.get_parameter('merge_radius').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.min_sightings = int(self.get_parameter('min_sightings').value)
        self.patch_half = int(self.get_parameter('depth_patch_half').value)
        log_path = self.get_parameter('log_path').value
        self.log_file = open(log_path, 'w') if log_path else None
        if self.log_file:
            self.log_file.write('sim_time,confidence,x,y,z,z_extent\n')
        self.depth_scale = float(self.get_parameter('depth_scale_to_metres').value)
        export_path = self.get_parameter('export_path').value
        self.export_path = export_path

        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.camera_info_ready = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.clusters = []
        self._next_id = 0

        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback,
            qos_profile_sensor_data)

        detections_topic = self.get_parameter('detections_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.detections_sub = message_filters.Subscriber(self, Detection2DArray, detections_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic,
                                             qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.detections_sub, self.depth_sub], queue_size=60, slop=0.1)
        self.sync.registerCallback(self.detections_callback)

        marker_topic = self.get_parameter('marker_topic').value
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)

        self.use_pose_history = bool(self.get_parameter('use_pose_history').value)
        self.upright_min_extent = float(self.get_parameter('upright_min_extent').value)
        self.pose_hist = deque(maxlen=800)     # (arrival time s, position, quaternion xyzw)
        self.cam_to_base = None                # static (R, t) of the optical frame in base_link
        self.create_subscription(PoseStamped, self.get_parameter('pose_topic').value,
                                 self.pose_callback, qos_profile_sensor_data)
        self.stats = dict(hist=0, latest=0, msgs=0, dets=0, low_conf=0, off_image=0, no_depth=0, tf_fail=0, used=0)
        self.create_timer(10.0, self.log_stats)
        self.export_srv = self.create_service(Trigger, '~/export_map', self.export_map_callback)
        self.people_json_pub = self.create_publisher(String, 'semantic_map/people_json', 10)
        self.create_timer(1.0, self.publish_people_json)

        self.get_logger().info(
            f"Integration node is ready: '{detections_topic}' + '{depth_topic}' "
            f"-> '{marker_topic}' (map_frame={self.map_frame}, merge_radius={self.merge_radius}m)")

    def log_stats(self):
        self.get_logger().info(
            'last 10 s: pose_at_image_time={hist} pose_latest={latest} msgs={msgs} detections={dets} low_conf={low_conf} off_image={off_image} '
            'no_depth={no_depth} tf_fail={tf_fail} used={used}'.format(**self.stats))
        for k in self.stats:
            self.stats[k] = 0

    def pose_callback(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        self.pose_hist.append((self.get_clock().now().nanoseconds * 1e-9,
                               np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])))

    def _pose_at(self, t):
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

    def _camera_to_base(self, camera_frame):
        if self.cam_to_base is None:
            try:
                tf = self.tf_buffer.lookup_transform('base_link', camera_frame, Time(),
                                                     timeout=Duration(seconds=0.2))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                return None
            r, t = tf.transform.rotation, tf.transform.translation
            self.cam_to_base = (quat_to_mat([r.x, r.y, r.z, r.w]), np.array([t.x, t.y, t.z]))
        return self.cam_to_base

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_model.fromCameraInfo(msg)
        self.camera_info_ready = True

    def detections_callback(self, detections: Detection2DArray, depth_msg: Image):
        if not self.camera_info_ready:
            self.get_logger().warn('No CameraInfo received yet, skipping detections.',
                                    throttle_duration_sec=5.0)
            return
        if not detections.detections:
            return
        self.stats['msgs'] += 1

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        camera_frame = self.camera_optical_frame or depth_msg.header.frame_id
        t_img = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
        pose = self._pose_at(t_img) if self.use_pose_history else None
        cam_to_base = self._camera_to_base(camera_frame) if pose is not None else None
        transform = None
        if pose is None or cam_to_base is None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, camera_frame, Time(), timeout=Duration(seconds=0.2))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.stats['tf_fail'] += len(detections.detections)
                self.get_logger().warn(f'tf lookup {camera_frame}->{self.map_frame} failed: {e}',
                                        throttle_duration_sec=5.0)
                return

        stamp = time.time()
        changed = False

        for det in detections.detections:
            if not det.results:
                continue
            self.stats['dets'] += 1
            confidence = det.results[0].hypothesis.score
            if confidence < self.min_confidence:
                self.stats['low_conf'] += 1
                continue

            u = int(det.bbox.center.position.x)
            v = int(det.bbox.center.position.y)
            if not (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
                self.stats['off_image'] += 1
                continue

            depth_m = self._robust_depth(depth_image, u, v)
            if depth_m <= 0.0:
                self.stats['no_depth'] += 1
                continue

            point_camera = self._deproject(u, v, depth_m)

            if transform is None:
                r_mb, t_mb = pose
                r_bc, t_bc = cam_to_base
                r_mo, t_mo = r_mb @ r_bc, r_mb @ t_bc + t_mb      # optical frame -> map
                self.stats['hist'] += 1
            else:
                q, tr = transform.transform.rotation, transform.transform.translation
                r_mo = quat_to_mat([q.x, q.y, q.z, q.w])
                t_mo = np.array([tr.x, tr.y, tr.z])
                self.stats['latest'] += 1
            p_map = r_mo @ np.array(point_camera) + t_mo
            map_point = (float(p_map[0]), float(p_map[1]), float(p_map[2]))
            extent = self._z_extent(depth_image, det, r_mo, t_mo)

            if self.log_file:
                t_sim = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
                ext_txt = '' if extent is None else f'{extent:.3f}'
                self.log_file.write(f'{t_sim:.2f},{confidence:.3f},{map_point[0]:.3f},{map_point[1]:.3f},{map_point[2]:.3f},{ext_txt}\n')
                self.log_file.flush()
            self.stats['used'] += 1
            self._merge_or_add(map_point, confidence, stamp, extent)
            changed = True

        if changed:
            self._publish_markers()

    def _robust_depth(self, depth_image, u, v):
        r = self.patch_half
        patch = depth_image[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1].astype(float)
        patch = patch * self.depth_scale
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size < 3:
            return -1.0
        return float(np.median(valid))

    def _z_extent(self, depth_image, det, r_mo, t_mo):
        uc, vc = det.bbox.center.position.x, det.bbox.center.position.y
        hw, hh = 0.3 * det.bbox.size_x, 0.3 * det.bbox.size_y
        h_img, w_img = depth_image.shape[:2]
        u0, u1 = int(max(0, uc - hw)), int(min(w_img, uc + hw))
        v0, v1 = int(max(0, vc - hh)), int(min(h_img, vc + hh))
        if u1 - u0 < 2 or v1 - v0 < 2:
            return None
        d = depth_image[v0:v1:3, u0:u1:3].astype(float) * self.depth_scale
        vv, uu = np.mgrid[v0:v1:3, u0:u1:3]
        ok = np.isfinite(d) & (d > 0.0)
        if ok.sum() < 15:
            return None
        d, uu, vv = d[ok], uu[ok], vv[ok]
        fx, fy = self.camera_model.fx(), self.camera_model.fy()
        cx, cy = self.camera_model.cx(), self.camera_model.cy()
        pts = np.stack([(uu - cx) * d / fx, (vv - cy) * d / fy, d])      # 3 x N, optical frame
        z = (r_mo[2] @ pts) + t_mo[2]
        return float(np.percentile(z, 90) - np.percentile(z, 10))

    def posture_of(self, cluster):
        ext = cluster.mean_extent()
        if ext is None:
            return 'unknown', None
        return ('upright' if ext >= self.upright_min_extent else 'lying'), ext

    def _deproject(self, u, v, depth_m):

        fx, fy = self.camera_model.fx(), self.camera_model.fy()
        cx, cy = self.camera_model.cx(), self.camera_model.cy()
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        return (x, y, z)

    def _merge_or_add(self, point, confidence, stamp, extent=None):
        for cluster in self.clusters:
            if cluster.distance_to(point) < self.merge_radius:
                cluster.update(point, confidence, stamp)
                cluster.add_extent(extent)
                if cluster.count == self.min_sightings:
                    self.get_logger().info(
                        f'Confirmed person #{cluster.id} at '
                        f'({cluster.x:.2f}, {cluster.y:.2f}, {cluster.z:.2f})')
                return
        cluster = SemanticCluster(self._next_id, point, confidence, stamp)
        cluster.add_extent(extent)
        self._next_id += 1
        self.clusters.append(cluster)

    def _confirmed(self):
        return [c for c in self.clusters if c.count >= self.min_sightings]

    def _publish_markers(self):
        array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for cluster in self._confirmed():
            sphere = Marker()
            sphere.header.frame_id = self.map_frame
            sphere.header.stamp = now
            sphere.ns = 'victims'
            sphere.id = cluster.id * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = cluster.x
            sphere.pose.position.y = cluster.y
            sphere.pose.position.z = cluster.z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.6
            posture, _ = self.posture_of(cluster)
            sphere.color.r, sphere.color.g, sphere.color.b = {
                'upright': (0.1, 0.8, 0.2), 'lying': (0.9, 0.15, 0.1)}.get(posture, (0.6, 0.6, 0.6))
            sphere.color.a = 0.9
            array.markers.append(sphere)

            label = Marker()
            label.header.frame_id = self.map_frame
            label.header.stamp = now
            label.ns = 'victims_label'
            label.id = cluster.id * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = cluster.x
            label.pose.position.y = cluster.y
            label.pose.position.z = cluster.z + 0.8
            label.pose.orientation.w = 1.0
            label.scale.z = 0.4
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text = f'#{cluster.id} {posture} ({cluster.confidence:.2f}, n={cluster.count})'
            array.markers.append(label)

        self.marker_pub.publish(array)

    def _payload(self):
        payload = []
        for c in self.clusters:
            posture, ext = self.posture_of(c)
            payload.append({
                'id': c.id,
                'x': c.x, 'y': c.y, 'z': c.z,
                'confidence': c.confidence,
                'detections': c.count,
                'confirmed': c.count >= self.min_sightings,
                'posture': posture,
                # Role is inferred from posture only: a person lying on the ground is treated as a casualty.
                'role': 'casualty' if posture == 'lying' else 'survivor' if posture == 'upright' else 'unknown',
                'z_extent': ext,
                'first_seen': c.first_seen,
                'last_seen': c.last_seen,
            })
        return payload

    def publish_people_json(self):
        self.people_json_pub.publish(String(data=json.dumps(self._payload())))

    def export_map_callback(self, request, response):
        payload = self._payload()
        try:
            with open(self.export_path, 'w') as f:
                json.dump(payload, f, indent=2)
            response.success = True
            response.message = f'Exported {len(payload)} markers to {self.export_path}'
        except OSError as e:
            response.success = False
            response.message = str(e)
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = IntegrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
