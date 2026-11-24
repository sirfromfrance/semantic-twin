#!/usr/bin/env python3
"""Operator interface: serves the live semantic twin as a web page.

  GET /            the page
  GET /twin.json   latest twin snapshot (from semantic_twin/twin_json); kept for compatibility
  GET /query?q=J   forwards the JSON request J to the twin node (kept for compatibility)
  /api/v1/...      REST API, version 1 (see rest_api.py); the page uses only this
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String

from semantic_twin import rest_api
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

            def api(self, method):
                u = urlparse(self.path)
                body = b''
                if method == 'POST':
                    n = int(self.headers.get('Content-Length') or 0)
                    if n > 65536:
                        self.send(413, json.dumps({'ok': False, 'error': 'body too large'}), 'application/json')
                        return
                    body = self.rfile.read(n)
                if (method == 'GET' and u.path.rstrip('/') == rest_api.PREFIX
                        and 'text/html' in (self.headers.get('Accept') or '')):
                    self.send(200, rest_api.index_html(node.snapshot()), 'text/html; charset=utf-8')
                    return
                code, obj = rest_api.handle(method, u.path, u.query, body, node.ask, node.snapshot)
                self.send(code, json.dumps(obj), 'application/json')

            def do_POST(self):
                if urlparse(self.path).path.startswith(rest_api.PREFIX):
                    self.api('POST')
                else:
                    self.send(404, 'not found', 'text/plain')

            def do_GET(self):
                u = urlparse(self.path)
                if u.path.startswith(rest_api.PREFIX):
                    self.api('GET')
                elif u.path in ('/', '/index.html'):
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
        self._parsed = None

    def snapshot(self):
        """Latest twin snapshot as a dict (parsed once per update), or None."""
        latest = self.latest
        if latest is None:
            return None
        cached = getattr(self, '_parsed', None)
        if cached is None or cached[0] is not latest:
            cached = (latest, json.loads(latest))
            self._parsed = cached
        return cached[1]

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
