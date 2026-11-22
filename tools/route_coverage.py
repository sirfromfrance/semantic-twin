#!/usr/bin/env python3

import importlib.util
import math
import sys

HERE = __file__.rsplit('/tools/', 1)[0]
spec = importlib.util.spec_from_file_location('ev', HERE + '/tools/evaluate_map.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

PITCH = math.radians(45)
TH, TV = math.tan(math.radians(30)), math.tan(math.radians(23.4))


def samples(route, step=0.5):
    pts = [(0.0, 0.0, 0.0)] + route          # starts at the origin, climbs to the first waypoint's altitude
    for (x0, y0, z0), (x1, y1, z1) in zip(pts, pts[1:]):
        n = max(1, int(math.dist((x0, y0, z0), (x1, y1, z1)) / step))
        yaw = math.atan2(y1 - y0, x1 - x0) if math.hypot(x1 - x0, y1 - y0) > 1.0 else None
        for i in range(n):
            t = i / n
            yield x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t, yaw


def main():
    route = [tuple(map(float, w.split(','))) for w in sys.argv[1].split()]
    world = sys.argv[2] if len(sys.argv) > 2 else HERE + '/src/semantic_twin/worlds/disaster_scene_open.world'
    people = ev.world_people(world)
    seen = {p[0]: 0.0 for p in people}
    last_yaw = 0.0
    for x, y, z, yaw in samples(route):
        if yaw is None:
            yaw = last_yaw
        last_yaw = yaw
        f = (math.cos(PITCH) * math.cos(yaw), math.cos(PITCH) * math.sin(yaw), -math.sin(PITCH))
        left = (-math.sin(yaw), math.cos(yaw), 0.0)
        up = (math.sin(PITCH) * math.cos(yaw), math.sin(PITCH) * math.sin(yaw), math.cos(PITCH))
        for name, px, py, kind in people:
            v = (px - x, py - y, (0.2 if kind == 'lying' else 0.9) - z)
            d = sum(a * b for a, b in zip(v, f))
            if d <= 0.5 or math.sqrt(sum(a * a for a in v)) > 25:
                continue
            if abs(sum(a * b for a, b in zip(v, left))) / d < TH and abs(sum(a * b for a, b in zip(v, up))) / d < TV:
                seen[name] += 0.5  
    for name, s in sorted(seen.items(), key=lambda kv: kv[1]):
        print(f'{name:18s} {s:5.1f} s in view' + ('   <-- never' if s == 0 else '   <-- short' if s < 4 else ''))
    print(f'\nnot in view at all: {sum(v == 0 for v in seen.values())}/{len(seen)};  under 4 s: {sum(0 < v < 4 for v in seen.values())}')


if __name__ == '__main__':
    main()
