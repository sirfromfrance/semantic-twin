import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped


class UavMission(Node):
    """Per-drone mission: wait for FCU -> OFFBOARD -> arm -> takeoff -> waypoints -> hold/land.

    Run inside the drone's namespace (uav<N>); waypoints are in that drone's local frame.
    """

    def __init__(self):
        super().__init__('uav_mission')

        self.declare_parameter('takeoff_alt', 2.0)
        self.declare_parameter('waypoints', [0.0, 0.0, 2.0])  # flat [x, y, z, x, y, z, ...]
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('land_at_end', False)

        self.takeoff_alt = self.get_parameter('takeoff_alt').value
        wp = self.get_parameter('waypoints').value
        self.waypoints = [tuple(wp[i:i + 3]) for i in range(0, len(wp) - len(wp) % 3, 3)]
        self.threshold = self.get_parameter('threshold').value
        self.land_at_end = self.get_parameter('land_at_end').value

        self.state = State()
        self.pose = None
        self.wp_index = 0
        self.phase = 'WAIT'
        self.last_request = self.get_clock().now()

        self.target = PoseStamped()
        self.target.header.frame_id = 'map'
        self.target.pose.orientation.w = 1.0

        self.create_subscription(State, 'mavros/state', self.state_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, 'mavros/local_position/pose', self.pose_cb, qos_profile_sensor_data)
        self.sp_pub = self.create_publisher(PoseStamped, 'mavros/setpoint_position/local', 10)
        self.arm_client = self.create_client(CommandBool, 'mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, 'mavros/set_mode')

        self.create_timer(0.05, self.loop)

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg):
        self.pose = msg

    def set_target(self, x, y, z):
        self.target.pose.position.x = float(x)
        self.target.pose.position.y = float(y)
        self.target.pose.position.z = float(z)

    def distance_to_target(self):
        p, t = self.pose.pose.position, self.target.pose.position
        return math.dist((p.x, p.y, p.z), (t.x, t.y, t.z))

    def throttled(self, period=1.0):
        """True at most once per `period` seconds, so service requests aren't spammed."""
        now = self.get_clock().now()
        if (now - self.last_request).nanoseconds / 1e9 < period:
            return False
        self.last_request = now
        return True

    def set_phase(self, phase):
        self.phase = phase
        self.get_logger().info(f'-> {phase}')

    def loop(self):
        # OFFBOARD needs a continuous setpoint stream from before the mode switch.
        self.target.header.stamp = self.get_clock().now().to_msg()
        self.sp_pub.publish(self.target)

        if self.phase == 'WAIT':
            if self.state.connected and self.pose is not None:
                p = self.pose.pose.position
                self.set_target(p.x, p.y, p.z)
                self.set_phase('ARM')

        elif self.phase == 'ARM':
            if self.state.armed and self.state.mode == 'OFFBOARD':
                self.set_target(self.pose.pose.position.x, self.pose.pose.position.y, self.takeoff_alt)
                self.set_phase('TAKEOFF')
            elif self.throttled():
                if self.state.mode != 'OFFBOARD':
                    self.mode_client.call_async(SetMode.Request(custom_mode='OFFBOARD'))
                elif not self.state.armed:
                    self.arm_client.call_async(CommandBool.Request(value=True))

        elif self.phase == 'TAKEOFF':
            if self.distance_to_target() < self.threshold:
                self.get_logger().info(f'Reached takeoff altitude {self.takeoff_alt} m')
                if self.waypoints:
                    self.set_target(*self.waypoints[0])
                    self.set_phase('MISSION')
                else:
                    self.set_phase('DONE')

        elif self.phase == 'MISSION':
            if self.distance_to_target() < self.threshold:
                self.get_logger().info(f'WP {self.wp_index} reached')
                self.wp_index += 1
                if self.wp_index < len(self.waypoints):
                    self.set_target(*self.waypoints[self.wp_index])
                else:
                    self.set_phase('DONE')

        elif self.phase == 'DONE':
            if self.land_at_end and self.throttled(2.0):
                self.mode_client.call_async(SetMode.Request(custom_mode='AUTO.LAND'))


def main(args=None):
    rclpy.init(args=args)
    node = UavMission()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
