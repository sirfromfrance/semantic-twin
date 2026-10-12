#!/usr/bin/env python3

import sys
import os
import math

local_pip_path = os.path.expanduser('~/.local/lib/python3.10/site-packages')
if local_pip_path not in sys.path:
    sys.path.insert(1, local_pip_path)

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import message_filters
from ultralytics import YOLO

COCO_PERSON_CLASS_ID = 0

class CvNode(Node):
    def __init__(self):
        super().__init__('sdt_cv_node')

        self.declare_parameter('image_topic', '/uav0/camera/image_raw')
        self.declare_parameter('depth_topic', '/uav0/camera/depth/image_raw')
        self.declare_parameter('detections_topic', 'detections')
        self.declare_parameter('debug_image_topic', 'debug_image')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('publish_debug_image', True)

        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        debug_image_topic = self.get_parameter('debug_image_topic').value
        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        self.detections_pub = self.create_publisher(Detection2DArray, detections_topic, 10)
        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(Image, debug_image_topic, 10)

        self.image_sub = message_filters.Subscriber(self, Image, image_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(
            f"CV node (Sensor Fusion) ready: '{image_topic}' + '{depth_topic}' -> '{detections_topic}'"
        )

    def sync_callback(self, image_msg: Image, depth_msg: Image):
        frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')

        results = self.model.predict(frame, classes=[COCO_PERSON_CLASS_ID], verbose=False)

        detections = Detection2DArray()
        detections.header = image_msg.header

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            
            x_center, y_center, w, h = box.xywh[0].cpu().numpy()

            
            cy = int(min(max(y_center, 0), depth_frame.shape[0] - 1))
            cx = int(min(max(x_center, 0), depth_frame.shape[1] - 1))
            distance = float(depth_frame[cy, cx])

            det = Detection2D()
            det.header = image_msg.header
            
            # 1. Bounding Box залишається суворо 2D (лише пікселі X та Y)
            det.bbox.center.position.x = float(x_center)
            det.bbox.center.position.y = float(y_center)
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'person'
            hyp.hypothesis.score = conf
            
            # 2. Фізичну дистанцію (Z) записуємо у спеціальне поле 3D-позиції
            if not math.isnan(distance) and not math.isinf(distance):
                hyp.pose.pose.position.z = float(distance)
            else:
                hyp.pose.pose.position.z = -1.0 # Маркер помилки далекоміра

            det.results.append(hyp)
            detections.detections.append(det)

            # Візуалізація для rqt_image_view
            if self.publish_debug_image:
                x_min, y_min = int(x_center - w / 2), int(y_center - h / 2)
                x_max, y_max = int(x_center + w / 2), int(y_center + h / 2)
                cv_rectangle_color = (0, 0, 255)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), cv_rectangle_color, 2)
                
                if hyp.pose.pose.position.z > 0:
                    text = f'person {conf:.2f} | Dist: {distance:.2f}m'
                else:
                    text = f'person {conf:.2f} | Dist: ERROR'
                    
                cv2.putText(frame, text, (x_min, max(0, y_min - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

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