import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger


class UavMission(Node):
  

    def __init__(self):
        super().__init__('uav_mission')

        self.declare_parameter('takeoff_alt', 2.0)
        self.declare_parameter('waypoints', [0.0, 0.0, 2.0])  # flat [x, y, z, x, y, z, ...]
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('land_at_end', False)
       
        self.declare_parameter('autostart', False)
        self.declare_parameter('max_speed', 1.0)      # m/s along each leg
        self.declare_parameter('max_yaw_rate', 0.5)   # rad/s

        self.takeoff_alt = self.get_parameter('takeoff_alt').value
        wp = self.get_parameter('waypoints').value
        self.waypoints = [tuple(wp[i:i + 3]) for i in range(0, len(wp) - len(wp) % 3, 3)]
        self.threshold = self.get_parameter('threshold').value
        self.land_at_end = self.get_parameter('land_at_end').value
        self.started = bool(self.get_parameter('autostart').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate').value)
        self.dt = 0.05
        self.goal_yaw = 0.0
        self.sp_yaw = 0.0

        self.state = State()
        self.pose = None
        self.wp_index = 0
        self.phase = 'WAIT'
        self.last_request = self.get_clock().now()

        self.target = PoseStamped()
        self.target.header.frame_id = 'map'
        self.target.pose.orientation.w = 1.0
        # Published setpoint: moves towards `target` at max_speed instead of jumping to it.
        self.sp = PoseStamped()
        self.sp.header.frame_id = 'map'
        self.sp.pose.orientation.w = 1.0

        self.create_subscription(State, 'mavros/state', self.state_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, 'mavros/local_position/pose', self.pose_cb, qos_profile_sensor_data)
        self.sp_pub = self.create_publisher(PoseStamped, 'mavros/setpoint_position/local', 10)
        self.arm_client = self.create_client(CommandBool, 'mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, 'mavros/set_mode')

        self.create_service(Trigger, 'start_mission', self.start_cb)
        self.create_timer(self.dt, self.loop)
        if not self.started:
            self.get_logger().info('Waiting for start: ros2 service call start_mission std_srvs/srv/Trigger')

    def start_cb(self, request, response):
        self.started = True
        response.success = True
        response.message = 'mission started'
        self.get_logger().info('Start requested')
        return response

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg):
        self.pose = msg

    def set_target(self, x, y, z):
        self.target.pose.position.x = float(x)
        self.target.pose.position.y = float(y)
        self.target.pose.position.z = float(z)

    def face_target(self):
        p, t = self.pose.pose.position, self.target.pose.position
        if math.hypot(t.x - p.x, t.y - p.y) < 1.0:
            return
        self.goal_yaw = math.atan2(t.y - p.y, t.x - p.x)

    def advance_setpoint(self):
        """Move the published setpoint towards the goal, limited by max_speed / max_yaw_rate."""
        sp, t = self.sp.pose.position, self.target.pose.position
        dx, dy, dz = t.x - sp.x, t.y - sp.y, t.z - sp.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        step = self.max_speed * self.dt
        if dist <= step:
            sp.x, sp.y, sp.z = t.x, t.y, t.z
        else:
            sp.x += dx / dist * step
            sp.y += dy / dist * step
            sp.z += dz / dist * step
        dyaw = math.atan2(math.sin(self.goal_yaw - self.sp_yaw), math.cos(self.goal_yaw - self.sp_yaw))
        max_dyaw = self.max_yaw_rate * self.dt
        self.sp_yaw += max(-max_dyaw, min(max_dyaw, dyaw))
        self.sp.pose.orientation.z = math.sin(self.sp_yaw / 2.0)
        self.sp.pose.orientation.w = math.cos(self.sp_yaw / 2.0)

    def distance_to_target(self):
        p, t = self.pose.pose.position, self.target.pose.position
        return math.dist((p.x, p.y, p.z), (t.x, t.y, t.z))

    def throttled(self, period=1.0):
        now = self.get_clock().now()
        if (now - self.last_request).nanoseconds / 1e9 < period:
            return False
        self.last_request = now
        return True

    def set_phase(self, phase):
        self.phase = phase
        self.get_logger().info(f'-> {phase}')

    def loop(self):
        self.sp.header.stamp = self.get_clock().now().to_msg()
        if self.phase != 'WAIT':
            self.advance_setpoint()
        self.sp_pub.publish(self.sp)

        if self.phase == 'WAIT':
            if self.started and self.state.connected and self.pose is not None:
                p = self.pose.pose.position
                self.set_target(p.x, p.y, p.z)
                self.sp.pose.position.x, self.sp.pose.position.y, self.sp.pose.position.z = p.x, p.y, p.z
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
                    self.face_target()
                    self.set_phase('MISSION')
                else:
                    self.set_phase('DONE')

        elif self.phase == 'MISSION':
            if self.distance_to_target() < self.threshold:
                self.get_logger().info(f'WP {self.wp_index} reached')
                self.wp_index += 1
                if self.wp_index < len(self.waypoints):
                    self.set_target(*self.waypoints[self.wp_index])
                    self.face_target()
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
