#!/usr/bin/env python3
"""Score the colour layer (sensed fire, foliage) of a twin state against the world file.

usage: python3 -I tools/evaluate_colour.py [TWIN_STATE.json] [WORLD.world]

Fire: a sensed fire is right if a fire model is within 3 m; a world fire is missed if its area was
observed but nothing was sensed. Foliage: cells marked foliage are right if they lie under a tree
(within 6 m of a tree model); the other cells are listed with the nearest world object.
The world file is used only here, to score, never by the system.
"""
import importlib.util
import math
import sys

import numpy as np

HERE = __file__.rsplit('/tools/', 1)[0]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    tc = load(f'{HERE}/src/semantic_twin/semantic_twin/twin_core.py', 'twin_core')
    bt = load(f'{HERE}/tools/build_twin.py', 'build_twin')
    state = sys.argv[1] if len(sys.argv) > 1 else f'{HERE}/results/twin_state.json'
    world = sys.argv[2] if len(sys.argv) > 2 else bt.launch_world()
    t = tc.Twin.load(state)
    if t.fire is None:
        raise SystemExit('this twin has no colour data (flown before the colour layer existed)')
    items = bt.read_world(world)
    fires = [(n, p[0], p[1]) for n, m, p in items if n.startswith('fire_')]
    trees = [(n, p[0], p[1]) for n, m, p in items if m in ('pine_tree', 'oak_tree')]
    others = [(n, p[0], p[1]) for n, m, p in items if not n.startswith(('person', 'casualty', 'randy', 'ground', 'sun', 'rubble', 'scorch', 'smoke', 'fire_', 'rebar', 'column'))]
    print(f'twin {state.rsplit("/", 1)[-1]}  world {world.rsplit("/", 1)[-1]}')

    sensed = t.sensed_fire()
    print(f'\nfire: {len(sensed)} sensed, {len(fires)} in the world')
    r = t.res
    tp = 0
    for n, fx, fy in fires:
        ix, iy = int((fx - r['x0']) // r['resolution']), int((fy - r['y0']) // r['resolution'])
        win = t.z[max(iy - 3, 0):iy + 4, max(ix - 3, 0):ix + 4]
        observed = bool((~np.isnan(win)).any())
        hit = [s for s in sensed if math.hypot(s['x'] - fx, s['y'] - fy) < 3.0]
        if hit:
            tp += 1
            s = min(hit, key=lambda s: math.hypot(s['x'] - fx, s['y'] - fy))
            print(f'  {n:8s} ({fx:6.1f},{fy:6.1f}) sensed at ({s["x"]},{s["y"]}), error {math.hypot(s["x"] - fx, s["y"] - fy):.2f} m, {s["points"]} points')
        else:
            print(f'  {n:8s} ({fx:6.1f},{fy:6.1f}) {"MISSED (area was observed)" if observed else "area not observed"}')
    for s in sensed:
        if not any(math.hypot(s['x'] - fx, s['y'] - fy) < 3.0 for _, fx, fy in fires):
            nearest = min(others, key=lambda o: math.hypot(o[1] - s['x'], o[2] - s['y']), default=None)
            print(f'  FALSE fire at ({s["x"]},{s["y"]}) {s["points"]} points; nearest world object: '
                  f'{nearest[0] if nearest else "-"} ({math.hypot(nearest[1] - s["x"], nearest[2] - s["y"]):.1f} m)')
    print(f'  found {tp}/{len(fires)}, false {sum(1 for s in sensed if not any(math.hypot(s["x"] - fx, s["y"] - fy) < 3.0 for _, fx, fy in fires))}')

    veg = t.veg_mask()
    ys, xs = np.mgrid[0:veg.shape[0], 0:veg.shape[1]]
    cx, cy = r['x0'] + (xs + .5) * r['resolution'], r['y0'] + (ys + .5) * r['resolution']
    under = np.zeros(veg.shape, dtype=bool)
    for _, tx, ty in trees:
        under |= np.hypot(cx - tx, cy - ty) <= 6.0
    nv = int(veg.sum())
    good = int((veg & under).sum())
    print(f'\nfoliage: {nv} cells marked, {good} under a tree ({100 * good / max(nv, 1):.0f}%)')
    bad = veg & ~under
    if bad.any():
        for comp in tc._components(bad):
            if len(comp) < 3:
                continue
            my, mx = np.mean([c[0] for c in comp]), np.mean([c[1] for c in comp])
            x, y = r['x0'] + (mx + .5) * r['resolution'], r['y0'] + (my + .5) * r['resolution']
            nearest = min(others, key=lambda o: math.hypot(o[1] - x, o[2] - y))
            print(f'  FALSE foliage cluster {len(comp)} cells at ({x:.1f},{y:.1f}); nearest world object {nearest[0]} ({math.hypot(nearest[1] - x, nearest[2] - y):.1f} m)')
    canopy = (~np.isnan(t.z)) & (np.nan_to_num(t.z) >= tc.VEG_MIN_HEIGHT) & np.zeros(veg.shape, dtype=bool)
    for name, tx, ty in trees:
        near = np.hypot(cx - tx, cy - ty) <= 4.0
        tall = near & (np.nan_to_num(t.z, nan=-9) >= tc.VEG_MIN_HEIGHT)
        print(f'  tree {name}: {int((tall & veg).sum())} of {int(tall.sum())} tall cells within 4 m marked foliage')

    g = t._grid_dict()
    cls = tc.classify(g)
    st = tc.structures(cls, g)
    print('\nstructures:')
    for s in st:
        print(f'  {s["name"]:12s} {s["kind"]:24s} {s["area_m2"]:6.1f} m2 {s["max_height"]:4.1f} m at ({s["x"]}, {s["y"]})')


if __name__ == '__main__':
    main()
