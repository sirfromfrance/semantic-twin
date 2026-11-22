#!/usr/bin/env python3

import importlib.util
import math
import os
import sys

import numpy as np

HERE = __file__.rsplit('/tools/', 1)[0]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    tc = load(f'{HERE}/src/semantic_twin/semantic_twin/twin_core.py', 'twin_core')
    bt, tg = load(f'{HERE}/tools/build_twin.py', 'build_twin'), load(f'{HERE}/tools/twin_grid.py', 'twin_grid')
    state = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/ROS/thesis_ws/results/twin_state.json')
    world = sys.argv[2] if len(sys.argv) > 2 else bt.launch_world()
    t = tc.Twin.load(state)
    snap = t.snapshot()
    g = t._grid_dict()
    res, x0, y0 = g['resolution'], g['x0'], g['y0']
    ny, nx = g['z'].shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    cx, cy = x0 + (xs + 0.5) * res, y0 + (ys + 0.5) * res

    z = np.zeros((ny, nx))
    for name, mesh, pose in bt.read_world(world):
        if mesh in bt.BUILDING_MESHES:
            for (kx, ky), h in tg.mesh_height_cells(mesh, pose[0], pose[1], cell=res).items():
                ix, iy = int(kx - math.floor(x0 / res)), int(ky - math.floor(y0 / res))
                if 0 <= ix < nx and 0 <= iy < ny:
                    z[iy, ix] = max(z[iy, ix], h)
        elif mesh in bt.VEHICLE_MODELS:
            z[np.hypot(cx - pose[0], cy - pose[1]) <= 1.8] = np.maximum(z[np.hypot(cx - pose[0], cy - pose[1]) <= 1.8], 2.0)
        elif mesh in ('pine_tree', 'oak_tree'):
            m = np.hypot(cx - pose[0], cy - pose[1]) <= 0.5
            z[m] = np.maximum(z[m], 3.0)
        elif name and name.startswith(('rubble_', 'column_', 'rebar_')):
            m = np.hypot(cx - pose[0], cy - pose[1]) <= 0.8
            z[m] = np.maximum(z[m], 0.8)
    gt = dict(g)
    gt['z'] = z
    flat = np.zeros((ny, nx), dtype=np.int8)
    wall = z >= tc.RUBBLE_MAX_M

    rows = []
    for p in snap['people']:
        ev = p['evacuation']
        gt_free = tc.plan_route(flat, gt, (p['x'], p['y']), t.hazards)
        row = {'id': p['id'], 'sensed': ev['status'], 'truth': gt_free['status'], 'collide': None, 'ratio': None}
        if ev['status'] == 'ok':
            pts, bad, n = ev['path'], 0, 0
            for a, b in zip(pts, pts[1:]):
                k = max(1, int(math.dist(a, b) / 0.25))
                for i in range(k + 1):
                    x, y = a[0] + (b[0] - a[0]) * i / k, a[1] + (b[1] - a[1]) * i / k
                    iy, ix = int((y - y0) // res), int((x - x0) // res)
                    n += 1
                    bad += bool(0 <= iy < ny and 0 <= ix < nx and wall[iy, ix])
            row['collide'] = bad / max(n, 1)
            same = tc.plan_route(flat, gt, (p['x'], p['y']), t.hazards, goal=tuple(ev['goal']))
            if same['status'] == 'ok':
                row['ratio'] = ev['length_m'] / same['length_m']
        rows.append(row)

    print(f'world {world.rsplit("/", 1)[-1]}; {len(rows)} people\n')
    print('person  sensed map     ground truth   path in real walls   length vs truth')
    for r in rows:
        print(f"#{r['id']:<6d} {r['sensed']:14s} {r['truth']:14s} "
              f"{'-' if r['collide'] is None else format(r['collide'], '.1%'):>10s}"
              f"{'-' if r['ratio'] is None else format(r['ratio'], '.2f'):>20s}")
    ok = [r for r in rows if r['sensed'] == 'ok']
    agree = sum(r['sensed'] == r['truth'] for r in rows)
    print(f"\nreachable on the sensed map: {len(ok)}/{len(rows)}; same answer as the ground truth: {agree}/{len(rows)}")
    if ok:
        print(f"routes that touch a real wall: {sum(r['collide'] > 0 for r in ok)}/{len(ok)}; "
              f"mean share of route points in walls: {np.mean([r['collide'] for r in ok]):.1%}")
        rat = [r['ratio'] for r in ok if r['ratio']]
        if rat:
            print(f"length vs ground-truth route to the same goal: mean {np.mean(rat):.2f}, max {max(rat):.2f}")


if __name__ == '__main__':
    main()
