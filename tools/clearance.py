#!/usr/bin/env python3

import glob
import math
import re
import sys

import numpy as np

ROOT = __file__.rsplit('/tools/', 1)[0] + '/src/semantic_twin_assets/models/'
BUILDINGS = {'collapsed_industrial_building': (-7.405, 10.4637),
             'collapsed_police_station': (10.2388, 3.37176),
             'collapsed_fire_station': (-20.8883, -2.03876)}
TREES = {'pine_tree': (-9.04634, 1.98892), 'oak_tree': (8.8463, -8.4822)}
MARGIN = 3.0  
TREE_CLEAR = 5.0   


def heights():
    H = {}
    for name, (px, py) in BUILDINGS.items():
        d = open(glob.glob(ROOT + name + '/meshes/*.dae')[0]).read()
        u = re.search(r'<unit[^>]*meter="([^"]+)"', d)
        u = float(u.group(1)) if u else 1.0
        for a in re.findall(r'<float_array[^>]*positions[^>]*>([^<]+)', d):
            for x, y, z in np.array(list(map(float, a.split()))).reshape(-1, 3) * u:
                k = (math.floor(px + x), math.floor(py + y))
                H[k] = max(H.get(k, -9.0), z)
    return H


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    x, y = float(sys.argv[1]), float(sys.argv[2])
    z = float(sys.argv[3]) if len(sys.argv) > 3 else None
    H = heights()
    near = max(H.get((xx, yy), -9.0) for xx in range(int(x) - 2, int(x) + 3) for yy in range(int(y) - 2, int(y) + 3))
    print(f'highest ruin within ~2 m of ({x:g}, {y:g}): {max(near, 0):.1f} m')
    ok = True
    for n, (tx, ty) in TREES.items():
        d = math.hypot(x - tx, y - ty)
        flag = '  <-- TOO CLOSE' if d < TREE_CLEAR else ''
        print(f'distance to {n} at ({tx:.1f}, {ty:.1f}): {d:.1f} m{flag}')
        ok &= d >= TREE_CLEAR
    if z is not None:
        margin = z - max(near, 0)
        print(f'altitude {z:g} m -> {margin:.1f} m above the ruins (want >= {MARGIN:g})')
        ok &= margin >= MARGIN
    print('OK' if ok else 'NOT SAFE: pick a higher z or move the point')


if __name__ == '__main__':
    main()
