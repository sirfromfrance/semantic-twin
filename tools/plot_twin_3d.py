#!/usr/bin/env python3
"""3D picture of a saved twin: the 2.5D height grid as a surface, plus people and map changes.

Offline figure tool: reads a twin state file, writes PNG images. Changes nothing in the twin.
usage: python3 tools/plot_twin_3d.py STATE.json [-o OUT_PREFIX] [--title TEXT]
       writes OUT_PREFIX_view1.png and OUT_PREFIX_view2.png (two viewing angles)
"""
import argparse
import importlib.util
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D

HERE = __file__.rsplit('/tools/', 1)[0]


def load_core():
    spec = importlib.util.spec_from_file_location('twin_core', f'{HERE}/src/semantic_twin/semantic_twin/twin_core.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('state')
    ap.add_argument('-o', '--out', default='twin_3d')
    ap.add_argument('--title', default=None)
    a = ap.parse_args()

    tc = load_core()
    t = tc.Twin.load(a.state)
    g = t._grid_dict()
    snap = t.snapshot()
    z = np.array(g['z'], dtype=float)
    res, x0, y0 = g['resolution'], g['x0'], g['y0']
    ny, nx = z.shape

    # crop to the scene: cells with something >= 1 m high, plus 6 m around
    ys, xs = np.nonzero(np.nan_to_num(z, nan=-1.0) >= 1.0)
    pad = int(6 / res)
    ya, yb = max(ys.min() - pad, 0), min(ys.max() + pad + 1, ny)
    xa, xb = max(xs.min() - pad, 0), min(xs.max() + pad + 1, nx)
    zc = z[ya:yb, xa:xb]
    X, Y = np.meshgrid(x0 + (np.arange(xa, xb) + 0.5) * res, y0 + (np.arange(ya, yb) + 0.5) * res)

    unknown = np.isnan(zc)
    zplot = np.where(unknown, 0.0, np.clip(zc, 0.0, None))
    norm = colors.Normalize(vmin=0.0, vmax=max(6.0, float(np.nanpercentile(zc, 99))))
    cmap = colors.LinearSegmentedColormap.from_list(
        'ruin', [(0.0, '#e8e1d0'), (0.08, '#d8ccb0'), (0.25, '#c49a5a'), (0.5, '#8a6a45'), (1.0, '#5a5a5a')])
    fc = cmap(norm(zplot))                             # light = open ground, tan = rubble, brown/grey = walls
    fc[unknown] = (1.0, 1.0, 1.0, 1.0)                 # never seen: white, drawn flat

    people = snap['people']
    regions = (snap.get('grid_changes') or {}).get('regions') or []
    flights = [f['name'] for f in t.flights]
    title = a.title or f"Semantic twin (2.5D height grid, {res} m cells) after {', '.join(flights)}"

    def zat(x, y):
        ix, iy = int((x - x0) // res), int((y - y0) // res)
        if 0 <= iy < ny and 0 <= ix < nx and not np.isnan(z[iy, ix]):
            return max(float(z[iy, ix]), 0.0)
        return 0.0

    for k, (elev, azim) in enumerate([(55, -60), (30, 35)], 1):
        fig = plt.figure(figsize=(13, 9), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        ax.computed_zorder = False                     # draw people and changes on top of the surface
        ax.plot_surface(X, Y, zplot, facecolors=fc, rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=True, zorder=1)
        for p in people:
            col = '#e0443e' if p['posture'] == 'lying' else '#2ca02c'
            gone = p['status'] != 'present'
            zb = 0.0 if p['posture'] == 'lying' else 0.2
            ax.plot([p['x'], p['x']], [p['y'], p['y']], [zb, 7.5], color=col, lw=1.4,
                    ls=':' if gone else '-', alpha=0.95, zorder=10)
            ax.scatter([p['x']], [p['y']], [7.5], s=60, depthshade=False, zorder=11,
                       facecolors='white' if gone else col, edgecolors=col, linewidths=1.5)
            ax.text(p['x'], p['y'], 8.2, f"#{p['id']}", fontsize=7, ha='center', zorder=12)
        for r in regions:
            ax.scatter([r['x']], [r['y']], [zat(r['x'], r['y']) + 0.3], marker='s', s=90, depthshade=False,
                       color='#ff9f1c', edgecolors='k', linewidths=0.5, zorder=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('height (m)')
        ax.set_zlim(0, 10)
        ax.set_box_aspect((X.max() - X.min(), Y.max() - Y.min(), 10 * 1.5))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11)
        handles = [Line2D([0], [0], marker='o', color='#2ca02c', ls='-', label='person, upright (present)'),
                   Line2D([0], [0], marker='o', color='#e0443e', ls='-', label='person, lying (present)'),
                   Line2D([0], [0], marker='o', color='#2ca02c', ls=':', markerfacecolor='none',
                          label='person not found in the latest flight'),
                   Line2D([0], [0], marker='s', color='#ff9f1c', ls='', markeredgecolor='k', label='map change region'),
                   Line2D([0], [0], marker='s', color='white', markeredgecolor='0.5', ls='', label='never observed')]
        ax.legend(handles=handles, loc='upper left', fontsize=8)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label='highest point seen in the cell (m)')
        out = f'{a.out}_view{k}.png'
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('wrote', out)


if __name__ == '__main__':
    main()
