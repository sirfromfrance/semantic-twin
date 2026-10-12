#!/usr/bin/env python3

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)


class SemanticCluster:
    """One accumulated estimate of a person's position in the map frame."""

    def __init__(self, cluster_id, point, confidence, stamp):
        self.id = cluster_id
        self.x, self.y, self.z = point
        self.confidence = confidence
        self.count = 1
        self.first_seen = stamp
        self.last_seen = stamp

    def update(self, point, confidence, stamp):
        # Running average, weighted by how many detections already went in.
        w = self.count
        self.x = (self.x * w + point[0]) / (w + 1)
        self.y = (self.y * w + point[1]) / (w + 1)
        self.z = (self.z * w + point[2]) / (w + 1)
        self.confidence = max(self.confidence, confidence)
        self.count += 1
        self.last_seen = stamp

    def distance_to(self, point):
        return math.sqrt((self.x - point[0]) ** 2 +
                          (self.y - point[1]) ** 2 +
                          (self.z - point[2]) ** 2)


class IntegrationNode(Node):
    def __init__(self):
        super().__init__('sdt_integration_node')

        self.declare_parameter('detections_topic', '/cv/detections')
        self.declare_parameter('depth_topic', '/uav0/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/uav0/camera/camera_info')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('merge_radius', 2.0)  # metres
        self.declare_parameter('min_confidence', 0.35)
        self.declare_parameter('marker_topic', 'semantic_map/victims')
        self.declare_parameter('export_path', '/tmp/sdt_semantic_map.json')
        # Depth images in metres (32FC1, e.g. simulated/RGB-D) vs millimetres
        # (16UC1, e.g. many real RGB-D cameras) need different scaling.
        self.declare_parameter('depth_scale_to_metres', 1.0)

        self.map_frame = self.get_parameter('map_frame').value
        self.merge_radius = float(self.get_parameter('merge_radius').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
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
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        detections_topic = self.get_parameter('detections_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.detections_sub = message_filters.Subscriber(self, Detection2DArray, detections_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.detections_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.detections_callback)

        marker_topic = self.get_parameter('marker_topic').value
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)

        self.export_srv = self.create_service(Trigger, '~/export_map', self.export_map_callback)

        self.get_logger().info(
            f"Integration node ready: '{detections_topic}' + '{depth_topic}' "
            f"-> '{marker_topic}' (map_frame={self.map_frame}, merge_radius={self.merge_radius}m)")

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

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        camera_frame = depth_msg.header.frame_id
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, camera_frame, depth_msg.header.stamp,
                timeout=Duration(seconds=0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'tf lookup {camera_frame}->{self.map_frame} failed: {e}',
                                    throttle_duration_sec=5.0)
            return

        stamp = time.time()
        changed = False

        for det in detections.detections:
            if not det.results:
                continue
            confidence = det.results[0].hypothesis.score
            if confidence < self.min_confidence:
                continue

            u = int(det.bbox.center.position.x)
            v = int(det.bbox.center.position.y)
            if not (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
                continue

            depth_raw = depth_image[v, u]
            if depth_raw is None or math.isnan(float(depth_raw)) or float(depth_raw) <= 0.0:
                continue
            depth_m = float(depth_raw) * self.depth_scale

            point_camera = self._deproject(u, v, depth_m)

            point_stamped = PointStamped()
            point_stamped.header.frame_id = camera_frame
            point_stamped.header.stamp = depth_msg.header.stamp
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = point_camera

            point_map = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            map_point = (point_map.point.x, point_map.point.y, point_map.point.z)

            self._merge_or_add(map_point, confidence, stamp)
            changed = True

        if changed:
            self._publish_markers()

    def _deproject(self, u, v, depth_m):
        """Pinhole model: pixel (u, v) + metric depth -> point in the camera
        optical frame (x-right, y-down, z-forward)."""
        fx, fy = self.camera_model.fx(), self.camera_model.fy()
        cx, cy = self.camera_model.cx(), self.camera_model.cy()
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        return (x, y, z)

    def _merge_or_add(self, point, confidence, stamp):
        for cluster in self.clusters:
            if cluster.distance_to(point) < self.merge_radius:
                cluster.update(point, confidence, stamp)
                return
        cluster = SemanticCluster(self._next_id, point, confidence, stamp)
        self._next_id += 1
        self.clusters.append(cluster)
        self.get_logger().info(
            f'New semantic marker #{cluster.id} at '
            f'({cluster.x:.2f}, {cluster.y:.2f}, {cluster.z:.2f}), conf={confidence:.2f}')

    def _publish_markers(self):
        array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for cluster in self.clusters:
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
            sphere.color.r = 1.0 - cluster.confidence
            sphere.color.g = cluster.confidence
            sphere.color.b = 0.0
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
            label.text = f'#{cluster.id} ({cluster.confidence:.2f}, n={cluster.count})'
            array.markers.append(label)

        self.marker_pub.publish(array)

    def export_map_callback(self, request, response):
        payload = [
            {
                'id': c.id,
                'x': c.x, 'y': c.y, 'z': c.z,
                'confidence': c.confidence,
                'detections': c.count,
                'first_seen': c.first_seen,
                'last_seen': c.last_seen,
            }
            for c in self.clusters
        ]
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
