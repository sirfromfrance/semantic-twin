#!/usr/bin/env python3

import importlib.util
import json
import math
import sys

import numpy as np

HERE = __file__.rsplit('/tools/', 1)[0]


def load(name):
    spec = importlib.util.spec_from_file_location(name, f'{HERE}/tools/{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    tg, bt = load('twin_grid'), load('build_twin')
    grid_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/sdt_height_map.json'
    world = sys.argv[2] if len(sys.argv) > 2 else bt.launch_world()
    people_path = '/tmp/sdt_semantic_map.json'
    g = tg.load_grid(grid_path)
    res, x0, y0 = g['resolution'], g['x0'], g['y0']
    ny, nx = g['z'].shape
    bh = g['blocked_height']

    truth = np.full((ny, nx), np.nan)
    for name, mesh, pose in bt.read_world(world):
        if mesh in bt.BUILDING_MESHES:
            for (kx, ky), h in tg.mesh_height_cells(mesh, pose[0], pose[1], cell=res).items():
                ix, iy = int(kx - math.floor(x0 / res)), int(ky - math.floor(y0 / res))
                if 0 <= ix < nx and 0 <= iy < ny and (math.isnan(truth[iy, ix]) or h > truth[iy, ix]):
                    truth[iy, ix] = h
    gt_block = truth >= bh

    seen = ~np.isnan(g['z'])
    sensed_block = seen & (g['z'] >= bh)
    ys, xs = np.mgrid[0:ny, 0:nx]
    cx, cy = x0 + (xs + 0.5) * res, y0 + (ys + 0.5) * res

    explained = gt_block.copy()
    _, _, discs = bt.build_static(world)
    for dx, dy, r in discs:
        explained |= np.hypot(cx - dx, cy - dy) <= r + res
    try:
        for p in json.load(open(people_path)):
            explained |= np.hypot(cx - p['x'], cy - p['y']) <= 1.2
    except OSError:
        pass
    near_gt = np.zeros_like(gt_block)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near_gt |= np.roll(np.roll(gt_block, dy, 0), dx, 1)

    n_gt = int(gt_block.sum())
    n_gt_seen = int((gt_block & seen).sum())
    hit = gt_block & sensed_block
    both = gt_block & seen & ~np.isnan(g['z'])
    err = np.abs(g['z'][both] - truth[both])
    false_obst = sensed_block & ~explained & ~near_gt

    print(f'grid: {int(seen.sum())} observed cells of {nx * ny} ({res} m cells), world {world.rsplit("/", 1)[-1]}')
    print(f'building cells in the mesh (>= {bh} m): {n_gt}; observed by the drone: {n_gt_seen} '
          f'({100 * n_gt_seen / max(n_gt, 1):.0f}% coverage)')
    print(f'of the observed ones, sensed as obstacle: {int(hit.sum())}/{n_gt_seen} '
          f'({100 * hit.sum() / max(n_gt_seen, 1):.0f}% recall)')
    if err.size:
        print(f'height error on observed building cells: median {np.median(err):.2f} m, mean {err.mean():.2f} m '
              f'(sensed = highest point the camera saw; roofs seen from above are fine, hidden tops are lower)')
    print(f'obstacle cells not explained by a building, known object or person: {int(false_obst.sum())} '
          f'of {int(sensed_block.sum())} sensed obstacle cells')
    structure_labels(g, gt_block, world, bt)


def structure_labels(g, gt_block, world, bt):
    """Score the environment labels: each sensed structure vs what the world file has under it."""
    spec = importlib.util.spec_from_file_location('twin_core', f'{HERE}/src/semantic_twin/semantic_twin/twin_core.py')
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    if g.get('veg') is not None and g.get('hi') is not None:
        g['veg_mask'] = tc.veg_mask_from(np.array(g['hi']), np.array(g['veg']), g['z'])
    cls = tc.classify(g)
    structs = tc.structures(cls, g)
    res, x0, y0 = g['resolution'], g['x0'], g['y0']
    ny, nx = g['z'].shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    cx, cy = x0 + (xs + 0.5) * res, y0 + (ys + 0.5) * res
    objs = {'tree': [], 'vehicle': []}
    for name, mesh, pose in bt.read_world(world):
        if mesh in ('pine_tree', 'oak_tree'):
            objs['tree'].append((pose[0], pose[1], 6.0))
        elif mesh in bt.VEHICLE_MODELS:
            objs['vehicle'].append((pose[0], pose[1], 3.0))
    mapping = {'building_ruin': 'building', 'vegetation': 'tree', 'tall_object': 'tree', 'vehicle_or_large_object': 'vehicle',
               'small_obstacle': 'other', 'spike_artifact': 'other'}
    print(f"\nenvironment labels: {len(structs)} sensed structures")
    ok = 0
    for st in structs:
        m = st['_mask']
        bfrac = float((m & gt_block).sum()) / m.sum()
        truth, best = 'other', 0.0
        if bfrac >= 0.5:
            truth, best = 'building', bfrac
        else:
            for kind, discs in objs.items():
                cov = np.zeros_like(m)
                for dx, dy, r in discs:
                    cov |= np.hypot(cx - dx, cy - dy) <= r
                f = float((m & cov).sum()) / m.sum()
                if f > best and f >= 0.3:
                    truth, best = kind, f
        guess = mapping[st['kind']]
        flag = 'ok' if guess == truth else 'WRONG'
        ok += guess == truth
        print(f"  {st['name']:12s} at ({st['x']:6.1f}, {st['y']:6.1f}) {st['area_m2']:6.1f} m2 {st['max_height']:4.1f} m  "
              f"labelled {st['kind']:24s} world has: {truth:9s} ({best:.0%}) {flag}")
    print(f"  labels agreeing with the world: {ok}/{len(structs)}  (trees and vehicles are guessed from size and height only)")


if __name__ == '__main__':
    main()
