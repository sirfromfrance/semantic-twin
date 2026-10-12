#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray


class GroundTruthNode(Node):
    def __init__(self):
        super().__init__('sdt_ground_truth_node')

        # Flat [x1, y1, z1, x2, y2, z2, ...] list, one triple per victim,
        # matching the positions the actor models were spawned at.
        self.declare_parameter('victim_positions', [])
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('marker_topic', 'ground_truth/victims')
        self.declare_parameter('publish_rate_hz', 1.0)

        flat = list(self.get_parameter('victim_positions').value)
        if len(flat) % 3 != 0:
            self.get_logger().error(
                f'victim_positions must be a flat list of x,y,z triples, got {len(flat)} values; '
                'dropping the remainder.')
            flat = flat[: len(flat) - (len(flat) % 3)]
        self.victims = [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]

        self.map_frame = self.get_parameter('map_frame').value
        marker_topic = self.get_parameter('marker_topic').value
        rate = float(self.get_parameter('publish_rate_hz').value)

        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.timer = self.create_timer(1.0 / rate, self.publish_markers)

        self.get_logger().info(
            f'Ground truth node ready: {len(self.victims)} victim(s) -> {marker_topic}')

    def publish_markers(self):
        array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, (x, y, z) in enumerate(self.victims):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = now
            marker.ns = 'ground_truth'
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = x, y, z
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = 0.5
            marker.scale.z = 1.8
            marker.color.r = 0.0
            marker.color.g = 0.6
            marker.color.b = 1.0
            marker.color.a = 0.4
            array.markers.append(marker)

        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
