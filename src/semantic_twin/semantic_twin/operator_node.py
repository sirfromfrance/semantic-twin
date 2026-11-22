#!/usr/bin/env python3
"""Operator interface: serves the live semantic twin as a web page.

  GET /            the page
  GET /twin.json   latest twin snapshot (from semantic_twin/twin_json)
  GET /query?q=J   forwards the JSON request J to the twin node and returns its answer
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String

from semantic_twin.operator_page import PAGE


class OperatorNode(Node):
    def __init__(self):
        super().__init__('sdt_operator_node')
        self.declare_parameter('port', 8080)
        self.declare_parameter('host', '127.0.0.1')
        self.latest = None
        self.answer = None
        self.answered = threading.Event()
        self.query_lock = threading.Lock()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, 'semantic_twin/twin_json', self.twin_cb, latched)
        self.create_subscription(String, 'semantic_twin/query_result', self.result_cb, 10)
        self.query_pub = self.create_publisher(String, 'semantic_twin/query', 10)
        host, port = self.get_parameter('host').value, int(self.get_parameter('port').value)
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, body, ctype):
                data = body.encode()
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                u = urlparse(self.path)
                if u.path in ('/', '/index.html'):
                    self.send(200, PAGE.replace('__TWIN__', 'null'), 'text/html; charset=utf-8')
                elif u.path == '/twin.json':
                    if node.latest is None:
                        self.send(503, json.dumps({'error': 'no twin received yet'}), 'application/json')
                    else:
                        self.send(200, node.latest, 'application/json')
                elif u.path == '/query':
                    try:
                        q = json.loads(parse_qs(u.query)['q'][0])
                    except (KeyError, ValueError):
                        self.send(400, json.dumps({'ok': False, 'error': 'bad query'}), 'application/json')
                        return
                    self.send(200, json.dumps(node.ask(q)), 'application/json')
                else:
                    self.send(404, 'not found', 'text/plain')

        self.server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.get_logger().info(f'Operator interface at http://{host}:{port}/')

    def twin_cb(self, msg):
        self.latest = msg.data

    def result_cb(self, msg):
        self.answer = msg.data
        self.answered.set()

    def ask(self, q):
        with self.query_lock:
            self.answered.clear()
            self.query_pub.publish(String(data=json.dumps(q)))
            if not self.answered.wait(5.0):
                return {'ok': False, 'error': 'the twin node did not answer'}
            return json.loads(self.answer)


def main(args=None):
    rclpy.init(args=args)
    node = OperatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
