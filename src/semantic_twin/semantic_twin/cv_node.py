#!/usr/bin/env python3
import sys
import os
import math
from builtins import KeyboardInterrupt

local_pip_path = os.path.expanduser('~/.local/lib/python3.10/site-packages')
if local_pip_path not in sys.path:
    sys.path.insert(1, local_pip_path)

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import message_filters
from ultralytics import YOLO

import tf2_ros
import tf2_geometry_msgs 

COCO_PERSON_CLASS_ID = 0


class CvNode(Node):
    def __init__(self):
        super().__init__('cv_node')

        self.declare_parameter('image_topic', '/uav0/camera/image_raw')
        self.declare_parameter('depth_topic', '/uav0/camera/depth/image_raw')

        self.declare_parameter('camera_info_topic', '/uav0/camera/depth/camera_info')
        self.declare_parameter('detections_topic', 'detections')
        self.declare_parameter('debug_image_topic', 'debug_image')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('imgsz', 640)  
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('depth_patch_half', 3)   
        self.declare_parameter('min_depth', 0.1)        
        self.declare_parameter('max_depth', 30.0)      

        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        debug_image_topic = self.get_parameter('debug_image_topic').value
        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.patch_half = int(self.get_parameter('depth_patch_half').value)
        self.min_depth = float(self.get_parameter('min_depth').value)
        self.max_depth = float(self.get_parameter('max_depth').value)

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        self.fx = self.fy = self.cx0 = self.cy0 = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.detections_pub = self.create_publisher(Detection2DArray, detections_topic, 10)
        self.point_cam_pub = self.create_publisher(PointStamped, 'person_point_camera', 10)
        self.point_world_pub = self.create_publisher(PointStamped, 'person_point_world', 10)
        #camera pub
        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(Image, debug_image_topic, 10)

        self.info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, qos_profile_sensor_data
        )

        #color image
        self.image_sub = message_filters.Subscriber(self, Image, image_topic, qos_profile=qos_profile_sensor_data)
        #depth image 
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub], queue_size=2, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(
            f"CV node ready: '{image_topic}' + '{depth_topic}' + '{camera_info_topic}' "
            f"'{detections_topic}', target_frame='{self.target_frame}'"
        )


#callback for camera info
    def camera_info_callback(self, msg: CameraInfo):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx0 = float(msg.k[2])
        self.cy0 = float(msg.k[5])

#using median of depth patch
    def robust_depth(self, depth_frame, u, v):
        h, w = depth_frame.shape[:2]
        u = int(min(max(u, 0), w - 1))
        v = int(min(max(v, 0), h - 1))
        r = self.patch_half
        patch = depth_frame[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
        valid = patch[np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)]
        if valid.size < 3:
            return -1.0
        return float(np.median(valid))

    def backproject(self, u, v, z):
        x_opt = (u - self.cx0) * z / self.fx
        y_opt = (v - self.cy0) * z / self.fy
        z_opt = z
        return z_opt, -x_opt, -y_opt   # x_link, y_link, z_link

    def sync_callback(self, image_msg: Image, depth_msg: Image):
        if self.fx is None:
            self.get_logger().warn('Waiting for camera_info...', throttle_duration_sec=5.0)
            return

        frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')  # metres

        results = self.model.predict(
            frame, classes=[COCO_PERSON_CLASS_ID], imgsz=self.imgsz,
            conf=self.confidence_threshold, verbose=False,
        )

        detections = Detection2DArray()
        detections.header = image_msg.header

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            u_c, v_c, w, h = box.xywh[0].cpu().numpy()
            z = self.robust_depth(depth_frame, u_c, v_c)

            det = Detection2D()
            det.header = image_msg.header
            det.bbox.center.position.x = float(u_c)
            det.bbox.center.position.y = float(v_c)
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'person'
            hyp.hypothesis.score = conf

            label_xyz = 'depth ERROR'
            if z > 0:
                x_l, y_l, z_l = self.backproject(u_c, v_c, z)

                pt = PointStamped()
                pt.header = image_msg.header
                pt.point.x, pt.point.y, pt.point.z = float(x_l), float(y_l), float(z_l)
                self.point_cam_pub.publish(pt)

                hyp.pose.pose.position.x = float(x_l)
                hyp.pose.pose.position.y = float(y_l)
                hyp.pose.pose.position.z = float(z_l)
                label_xyz = f'x={x_l:.2f} y={y_l:.2f} z={z_l:.2f}m'

                if self.target_frame:
                    try:
                        pt_latest = PointStamped()
                        pt_latest.header.frame_id = pt.header.frame_id
                        pt_latest.point = pt.point
                        pt_world = self.tf_buffer.transform(
                            pt_latest, self.target_frame, timeout=Duration(seconds=0.1)
                        )
                        pt_world.header.stamp = image_msg.header.stamp
                        self.point_world_pub.publish(pt_world)
                    except Exception as e:  # noqa: BLE001
                        self.get_logger().warn(
                            f"tf2 {pt.header.frame_id} -> {self.target_frame} failed: {e}",
                            throttle_duration_sec=2.0,
                        )
            else:
                hyp.pose.pose.position.z = -1.0  

            det.results.append(hyp)
            detections.detections.append(det)

            if self.publish_debug_image:
                x_min, y_min = int(u_c - w / 2), int(v_c - h / 2)
                x_max, y_max = int(u_c + w / 2), int(v_c + h / 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                text = f'person {conf:.2f} | {label_xyz}'
                cv2.putText(frame, text, (x_min, max(12, y_min - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (int(u_c), int(v_c)), 4, (255, 0, 0), -1)

        self.detections_pub.publish(detections)

        if self.publish_debug_image:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            debug_msg.header = image_msg.header
            self.debug_image_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()