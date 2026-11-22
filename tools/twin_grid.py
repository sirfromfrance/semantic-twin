
import glob
import json
import math
import re

import numpy as np

ROOT = __file__.rsplit('/tools/', 1)[0] + '/src/semantic_twin_assets/models/'
TALL = 3.0
MIN_STRUCTURE_CELLS = 6


def mesh_height_cells(mesh_name, px, py, cell=1.0, step=0.25):
    d = open(glob.glob(ROOT + mesh_name + '/meshes/*.dae')[0]).read()
    unit = re.search(r'<unit[^>]*meter="([^"]+)"', d)
    unit = float(unit.group(1)) if unit else 1.0
    pos = re.search(r'<float_array[^>]*positions[^>]*>([^<]+)', d)
    verts = np.array(pos.group(1).split(), dtype=float).reshape(-1, 3) * unit
    cells = {}
    for block in re.findall(r'<triangles[^>]*>(.*?)</triangles>', d, re.S):
        offsets = [int(o) for o in re.findall(r'<input[^>]*offset="(\d+)"', block)]
        vert_off = int(re.search(r'semantic="VERTEX"[^>]*offset="(\d+)"', block).group(1)) \
            if re.search(r'semantic="VERTEX"[^>]*offset="(\d+)"', block) else 0
        stride = max(offsets) + 1 if offsets else 1
        idx = np.array(re.search(r'<p>([^<]+)</p>', block).group(1).split(), dtype=int)[vert_off::stride]
        for a, b, c in idx[:len(idx) // 3 * 3].reshape(-1, 3):
            A, B, C = verts[a], verts[b], verts[c]
            n = int(min(max(np.ptp([A, B, C], axis=0)[:2].max() / step, 1), 120))
            ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
            m = (ii + jj) <= n
            u, v = ii[m] / n, jj[m] / n
            P = A + np.outer(u, B - A) + np.outer(v, C - A)
            for kx, ky, z in zip(np.floor((px + P[:, 0]) / cell).astype(int),
                                 np.floor((py + P[:, 1]) / cell).astype(int), P[:, 2]):
                k = (int(kx), int(ky))
                if z > cells.get(k, -9.0):
                    cells[k] = float(z)
    return cells


def load_grid(path):
    g = json.load(open(path))
    g['z'] = np.array([[np.nan if v is None else v for v in row] for row in g['max_z']], dtype=float)
    return g


def classify(g, people=(), person_clear=1.2):
   
    z = g['z']
    cls = np.zeros(z.shape, dtype=np.int8)
    seen = ~np.isnan(z)
    cls[seen] = 1
    cls[seen & (z >= 0.5)] = 2
    cls[seen & (z >= g['blocked_height'])] = 3
    cls[seen & (z >= TALL)] = 4
    res = g['resolution']
    ys, xs = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    cx, cy = g['x0'] + (xs + 0.5) * res, g['y0'] + (ys + 0.5) * res
    for p in people:
        near = np.hypot(cx - p['x'], cy - p['y']) <= person_clear
        cls[near & (cls >= 2) & (cls < 4)] = 1
    return cls


def rows(cls):
    ch = np.array(['.', '_', 'o', '#', 'T'])
    return [''.join(ch[r]) for r in cls]


def structures(cls, g):
    blocked = cls >= 3
    seen = np.zeros(cls.shape, dtype=bool)
    out = []
    for sy, sx in zip(*np.nonzero(blocked)):
        if seen[sy, sx]:
            continue
        stack, comp = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < cls.shape[0] and 0 <= xx < cls.shape[1] and blocked[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        if len(comp) >= MIN_STRUCTURE_CELLS:
            out.append(comp)
    res = g['resolution']
    result = []
    for i, comp in enumerate(sorted(out, key=lambda c: (-len(c)))):
        ys, xs = zip(*comp)
        result.append({
            'name': f'structure_{i}', 'cells': len(comp), 'area_m2': round(len(comp) * res * res, 1),
            'x': round(g['x0'] + (np.mean(xs) + 0.5) * res, 2), 'y': round(g['y0'] + (np.mean(ys) + 0.5) * res, 2),
            'max_height': round(float(max(g['z'][y, x] for y, x in comp)), 1),
            'bbox': [round(g['x0'] + min(xs) * res, 2), round(g['x0'] + (max(xs) + 1) * res, 2),
                     round(g['y0'] + min(ys) * res, 2), round(g['y0'] + (max(ys) + 1) * res, 2)],
            'members': comp})
    return result


def cell_centres(g, mask):
    ys, xs = np.nonzero(mask)
    return (g['x0'] + (xs + 0.5) * g['resolution'], g['y0'] + (ys + 0.5) * g['resolution'])


def sensed_relations(person, cls, g, structs):
    res = g['resolution']
    x, y = person['x'], person['y']
    ix, iy = int((x - g['x0']) // res), int((y - g['y0']) // res)
    inside = None
    if 0 <= iy < cls.shape[0] and 0 <= ix < cls.shape[1] and cls[iy, ix] >= 3:
        for s in structs:
            if (iy, ix) in s['members_set']:
                inside = s['name']
    best = None
    for s in structs:
        mx, my = cell_centres(g, s['mask'])
        d = float(np.min(np.hypot(mx - x, my - y)))
        if best is None or d < best[1]:
            best = (s['name'], d)
    fx, fy = cell_centres(g, cls == 1)
    if 0 <= iy < cls.shape[0] and 0 <= ix < cls.shape[1] and cls[iy, ix] == 1:
        og = 0.0
    else:
        og = float(np.min(np.hypot(fx - x, fy - y))) if len(fx) else None
    return inside, best, og
