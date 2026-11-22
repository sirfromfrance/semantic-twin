#!/usr/bin/env python3

import json
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger

from semantic_twin import twin_core as tc


class TwinNode(Node):
    def __init__(self):
        super().__init__('sdt_twin_node')
        P = self.declare_parameter
        P('people_topic', 'semantic_map/people_json')
        P('grid_topic', 'semantic_map/grid_json')
        P('state_path', os.path.expanduser('~/ROS/thesis_ws/results/twin_state.json'))
        P('export_path', '/tmp/sdt_twin.json')
        P('use_sensed_fire', False)         # colour-sensed fire drives routes and triage (not reliable in the tests)
        P('load_state', False)             # true: continue the twin of an earlier flight
        P('hazard_world', 'disaster_scene.world')
        P('flight_name', '')
        P('save_period', 5.0)
        g = lambda n: self.get_parameter(n).value
        self.state_path, self.export_path = g('state_path'), g('export_path')

        hazards = tc.hazards_from_world(
            os.path.join(get_package_share_directory('semantic_twin'), 'worlds', g('hazard_world')))
        if g('load_state') and os.path.exists(self.state_path):
            import shutil
            shutil.copy(self.state_path, self.state_path + '.bak')       # the state is overwritten by autosave
            self.twin = tc.Twin.load(self.state_path)
            self.twin.hazards = hazards
            self.twin.use_sensed_fire = bool(g('use_sensed_fire'))
            n_rej = self.twin.clean_people()
            if n_rej:
                self.get_logger().info(f'{n_rej} stored observation(s) rejected by the screening rules')
            self.get_logger().info(f'Loaded twin from {self.state_path}: {len(self.twin.people)} people, '
                                   f'{len(self.twin.flights)} earlier flight(s)')
        else:
            self.twin = tc.Twin(hazards=hazards, use_sensed_fire=bool(g('use_sensed_fire')))
        name = self.twin.begin_flight(g('flight_name') or None, reuse_empty=True)
        self.dirty = True

        self.create_subscription(String, g('people_topic'), self.people_cb, 10)
        self.create_subscription(String, g('grid_topic'), self.grid_cb, 1)
        self.create_subscription(String, 'semantic_twin/query', self.query_cb, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.twin_pub = self.create_publisher(String, 'semantic_twin/twin_json', latched)
        self.result_pub = self.create_publisher(String, 'semantic_twin/query_result', 10)
        self.create_service(Trigger, '~/export_twin', self.export_cb)
        self.create_service(Trigger, '~/new_flight', self.new_flight_cb)
        self.create_timer(float(g('save_period')), self.autosave)
        self.get_logger().info(f'Twin node ready, current flight: {name}, hazards loaded: {len(hazards)}')

    def people_cb(self, msg):
        try:
            self.twin.update_people(json.loads(msg.data))
            self.dirty = True
        except (ValueError, KeyError) as e:
            self.get_logger().warn(f'bad people message: {e}', throttle_duration_sec=5.0)

    def grid_cb(self, msg):
        try:
            self.twin.update_grid(tc.grid_from_dict(json.loads(msg.data)))
            self.dirty = True
        except (ValueError, KeyError) as e:
            self.get_logger().warn(f'bad grid message: {e}', throttle_duration_sec=5.0)

    def query_cb(self, msg):
        try:
            result = self.twin.query(json.loads(msg.data))
        except (ValueError, KeyError) as e:
            result = {'ok': False, 'error': str(e)}
        self.result_pub.publish(String(data=json.dumps(result)))

    def write(self):
        snap = self.twin.snapshot()
        with open(self.export_path, 'w') as f:
            json.dump(snap, f, indent=1)
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self.twin.save(self.state_path)
        self.twin_pub.publish(String(data=json.dumps(snap)))
        self.dirty = False
        return snap

    def autosave(self):
        if not self.dirty:
            return
        try:
            s = self.write()['summary']
        except OSError as e:
            self.get_logger().warn(f'could not save: {e}', throttle_duration_sec=10.0)
            return
        self.get_logger().info(
            f"twin: {s['people']} people ({s['casualties']} lying, {s['high_priority']} high priority), "
            f"{s['structures']} structures, {s['observed_cells']} observed cells")

    def export_cb(self, request, response):
        try:
            s = self.write()['summary']
            response.success = True
            response.message = f"twin saved to {self.export_path} and {self.state_path}: {s}"
        except OSError as e:
            response.success, response.message = False, str(e)
        self.get_logger().info(response.message)
        return response

    def new_flight_cb(self, request, response):
        name = self.twin.begin_flight(reuse_empty=True)
        response.success, response.message = True, f'started {name}'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TwinNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.write()
        except Exception: 
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
