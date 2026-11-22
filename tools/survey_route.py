#!/usr/bin/env python3

import argparse
import heapq
import importlib.util
import math

import numpy as np

HERE = __file__.rsplit('/tools/', 1)[0]


def load(name):
    spec = importlib.util.spec_from_file_location(name, f'{HERE}/tools/{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PITCH = math.radians(45)
TH, TV = math.tan(math.radians(30)), math.tan(math.radians(23.4))
MAX_DEPTH = 25.0


class Safe:
    def __init__(self, world, bounds, alt, margin, tree, bt, tg):
        self.x0, self.x1, self.y0, self.y1 = bounds
        self.alt, self.margin, self.tree = alt, margin, tree
        self.h = {}
        self.trees = []
        for name, mesh, pose in bt.read_world(world):
            if mesh in bt.BUILDING_MESHES:
                for k, v in tg.mesh_height_cells(mesh, pose[0], pose[1]).items():
                    self.h[k] = max(self.h.get(k, -9.0), v)
            elif mesh in ('pine_tree', 'oak_tree'):
                self.trees.append((pose[0], pose[1]))

    def ok(self, x, y):
        if any(math.hypot(x - tx, y - ty) < self.tree for tx, ty in self.trees):
            return False
        kx, ky = math.floor(x), math.floor(y)
        top = max(self.h.get((kx + dx, ky + dy), 0.0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        return top + self.margin <= self.alt

    def leg_ok(self, a, b, step=0.5):
        n = max(1, int(math.dist(a, b) / step))
        return all(self.ok(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1))

    def astar(self, a, b, cell=1.0, pad=8.0):
       
        ox, oy = self.x0 - pad, self.y0 - pad

        def idx(p):
            return (round((p[0] - ox) / cell), round((p[1] - oy) / cell))

        def pos(i):
            return (ox + i[0] * cell, oy + i[1] * cell)
        nx, ny = int((self.x1 - self.x0 + 2 * pad) / cell) + 1, int((self.y1 - self.y0 + 2 * pad) / cell) + 1
        s, g = idx(a), idx(b)
        openq, cost, prev = [(0.0, s)], {s: 0.0}, {}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                break
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nb = (cur[0] + dx, cur[1] + dy)
                    if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                        continue
                    p = pos(nb)
                    if not self.ok(*p):
                        continue
                    c = cost[cur] + math.hypot(dx, dy)
                    if c < cost.get(nb, 1e18):
                        cost[nb], prev[nb] = c, cur
                        heapq.heappush(openq, (c + math.dist(nb, g), nb))
        if g not in cost:
            return None
        path, cur = [], g
        while cur != s:
            path.append(pos(cur))
            cur = prev[cur]
        path = [a] + path[::-1]
        path[-1] = b
        out, i = [path[0]], 0
        while i < len(path) - 1:       # keep the farthest point that is still visible
            j = len(path) - 1
            while j > i + 1 and not self.leg_ok(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out


def lawnmower(safe, axis, spacing, offset=0.0, min_run=6.0, step=0.5):
    x0, x1, y0, y1 = safe.x0, safe.x1, safe.y0, safe.y1
    lines = []
    if axis == 'x':
        vals = list(np.arange(y0 + offset, y1 + 1e-6, spacing))
        if vals[-1] < y1 - spacing * 0.4:
            vals.append(y1)
        for k, v in enumerate(vals):
            lines.append((k, [(x, v) for x in np.arange(x0, x1 + 1e-6, step)]))
    else:
        vals = list(np.arange(x0 + offset, x1 + 1e-6, spacing))
        if vals[-1] < x1 - spacing * 0.4:
            vals.append(x1)
        for k, v in enumerate(vals):
            lines.append((k, [(v, y) for y in np.arange(y0, y1 + 1e-6, step)]))
    runs = []
    for k, pts in lines:
        if k % 2:
            pts = pts[::-1]
        run = []
        for p in pts + [None]:
            if p is not None and safe.ok(*p):
                run.append(p)
            else:
                if len(run) * step >= min_run:
                    runs.append((run[0], run[-1]))
                run = []
    return runs


def build_route(safe, axis, spacing, start=(0.0, 0.0), offset=0.0, return_home=True):
    runs = lawnmower(safe, axis, spacing, offset)
    route = []
    cur = start
    problems = []
    for a, b in runs:
        for target in (a, b):
            if safe.leg_ok(cur, target):
                seg = [cur, target]
            else:
                seg = safe.astar(cur, target)
                if seg is None:
                    problems.append(f'no safe path from {cur} to {target}')
                    continue
            route += seg[1:]
            cur = target
    if return_home and cur != start:
        seg = [cur, start] if safe.leg_ok(cur, start) else safe.astar(cur, start)
        if seg is None:
            problems.append(f'no safe path back to {start}')
        else:
            route += seg[1:]
    # drop collinear and duplicate points
    out = []
    for p in route:
        if out and math.dist(out[-1], p) < 0.05:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            cross = (b[0] - a[0]) * (p[1] - b[1]) - (b[1] - a[1]) * (p[0] - b[0])
            dot = (b[0] - a[0]) * (p[0] - b[0]) + (b[1] - a[1]) * (p[1] - b[1])
            if abs(cross) < 1e-6 and dot > 0:
                out[-1] = p
                continue
        out.append(p)
    return out, problems


def coverage(route, start, alt, bounds, step=1.0):
    x0, x1, y0, y1 = bounds
    gx, gy = np.meshgrid(np.arange(x0 + 0.5, x1, 1.0), np.arange(y0 + 0.5, y1, 1.0))
    gx, gy = gx.ravel(), gy.ravel()
    seen = np.zeros(gx.size, dtype=bool)
    pts = [start] + route
    last_yaw = 0.0
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(math.dist(a, b) / step))
        yaw = math.atan2(b[1] - a[1], b[0] - a[0]) if math.dist(a, b) > 0.5 else last_yaw
        last_yaw = yaw
        f = np.array([math.cos(PITCH) * math.cos(yaw), math.cos(PITCH) * math.sin(yaw), -math.sin(PITCH)])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        up = np.array([math.sin(PITCH) * math.cos(yaw), math.sin(PITCH) * math.sin(yaw), math.cos(PITCH)])
        for i in range(n + 1):
            px = a[0] + (b[0] - a[0]) * i / n
            py = a[1] + (b[1] - a[1]) * i / n
            v = np.stack([gx - px, gy - py, np.full(gx.size, -alt)], axis=1)
            d = v @ f
            with np.errstate(all='ignore'):
                ok = (d > 1.0) & (np.linalg.norm(v, axis=1) < MAX_DEPTH) \
                    & (np.abs(v @ left) / d < TH) & (np.abs(v @ up) / d < TV)
            seen |= ok
    return seen.reshape((len(np.arange(y0 + 0.5, y1, 1.0)), -1)), (gx, gy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('world', nargs='?')
    ap.add_argument('--bounds', type=float, nargs=4, default=[-20, 20, -12, 12], metavar=('X0', 'X1', 'Y0', 'Y1'))
    ap.add_argument('--alt', type=float, default=12.0)
    ap.add_argument('--spacing', type=float, default=9.0)
    ap.add_argument('--speed', type=float, default=0.7)
    ap.add_argument('--axis', choices=['x', 'y', 'auto'], default='auto')
    ap.add_argument('--margin', type=float, default=3.0)
    ap.add_argument('--tree', type=float, default=5.0)
    a = ap.parse_args()
    bt, tg = load('build_twin'), load('twin_grid')
    world = a.world or bt.launch_world()
    safe = Safe(world, a.bounds, a.alt, a.margin, a.tree, bt, tg)
    start = (0.0, 0.0)
    if not safe.ok(*start):
        print('warning: the take-off point (0, 0) is not clear at this altitude')

    cands = []
    for axis in ([a.axis] if a.axis != 'auto' else ['x', 'y']):
        for off in range(0, int(a.spacing), 1):
            r, pr = build_route(safe, axis, a.spacing, start, float(off))
            if not r:
                continue
            ok = not pr and all(safe.leg_ok(p, q) for p, q in zip([start] + r, r))
            ln = sum(math.dist(p, q) for p, q in zip([start] + r, r))
            sn, _ = coverage(r, start, a.alt, a.bounds)
            cands.append((ok, sn.mean(), ln, axis, off, r, pr))
    if not cands:
        raise SystemExit('no safe route found; try a higher --alt or smaller --bounds')
    good = [c for c in cands if c[0]]
    pool = good or cands
    best_cov = max(c[1] for c in pool)
    pool = [c for c in pool if c[1] >= best_cov - 0.03]       # equally good coverage: take the shortest
    ok, _, length, axis, off, route, problems = min(pool, key=lambda c: c[2])
    a.axis = axis
    legs_ok = ok
    seen, (gx, gy) = coverage(route, start, a.alt, a.bounds)
    print(f'tried {len(cands)} line layouts ({len(good)} safe); chose axis {axis}, offset {off} m')

    cells = {}
    for name, mesh, pose in bt.read_world(world):
        if mesh in bt.BUILDING_MESHES:
            for k, v in tg.mesh_height_cells(mesh, pose[0], pose[1]).items():
                cells[k] = max(cells.get(k, -9.0), v)
    inb = [(k, v) for k, v in cells.items()
           if a.bounds[0] <= k[0] + 0.5 <= a.bounds[1] and a.bounds[2] <= k[1] + 0.5 <= a.bounds[3] and v >= 1.0]
    x0, y0 = a.bounds[0], a.bounds[2]
    hit = sum(1 for k, _ in inb if 0 <= int(k[1] + 0.5 - y0) < seen.shape[0] and 0 <= int(k[0] + 0.5 - x0) < seen.shape[1]
              and seen[int(k[1] + 0.5 - y0), int(k[0] + 0.5 - x0)])

    print(f'world {world.rsplit("/", 1)[-1]}, bounds {a.bounds}, altitude {a.alt} m, line spacing {a.spacing} m, axis {a.axis}')
    print(f'{len(route)} waypoints, {length:.0f} m, about {length / a.speed / 60:.1f} min at {a.speed} m/s')
    print(f'every leg safe (margin {a.margin} m over ruins, {a.tree} m from trees): {legs_ok}')
    for p in problems:
        print('PROBLEM:', p)
    print(f'estimated camera coverage of the rectangle: {100 * seen.mean():.0f}% of ground cells; '
          f'of the building cells inside it: {100 * hit / max(len(inb), 1):.0f}% ({hit}/{len(inb)})')
    flat = ', '.join(f'{x:.1f}, {y:.1f}, {a.alt:.1f}' for x, y in route)
    print('\nwaypoints argument for disaster_launch.xml:')
    print(f'waypoints:="[{flat}]" takeoff_alt:={a.alt:.1f} max_speed:={a.speed}')
    print('\nfor tools/route_coverage.py / clearance.py:')
    print(' '.join(f'{x:.1f},{y:.1f},{a.alt:.1f}' for x, y in route))


if __name__ == '__main__':
    main()
