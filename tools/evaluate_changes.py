#!/usr/bin/env python3

import importlib.util
import json
import math
import sys

HERE = __file__.rsplit('/tools/', 1)[0]


def load(name):
    spec = importlib.util.spec_from_file_location(name, f'{HERE}/tools/{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def greedy(expected, reported, radius=1.5):
    pairs = sorted((math.hypot(e[0] - r[0], e[1] - r[1]), i, j)
                   for i, e in enumerate(expected) for j, r in enumerate(reported))
    ue, ur, hit = set(), set(), []
    for d, i, j in pairs:
        if d <= radius and i not in ue and j not in ur:
            ue.add(i)
            ur.add(j)
            hit.append((i, j, d))
    return hit, [e for i, e in enumerate(expected) if i not in ue], [r for j, r in enumerate(reported) if j not in ur]


def main():
    args = sys.argv[1:]
    twin_path = args.pop(0) if len(args) == 3 else '/tmp/sdt_twin.json'
    ev, bt = load('evaluate_map'), load('build_twin')
    before = {p[0]: p for p in ev.world_people(args[0])}
    after = {p[0]: p for p in ev.world_people(args[1])}
    t = json.load(open(twin_path))
    cur = t['meta']['current_flight']

    exp_left, exp_new, exp_moved = [], [], []
    for n in sorted(set(before) | set(after)):
        a, b = before.get(n), after.get(n)
        if a is None:
            exp_new.append((b[1], b[2], n))
        elif b is None:
            exp_left.append((a[1], a[2], n))
        else:
            d = math.hypot(a[1] - b[1], a[2] - b[2])
            if d > 1.5:
                exp_left.append((a[1], a[2], n + ' (old place)'))
                exp_new.append((b[1], b[2], n + ' (new place)'))
            elif d > 1.0:
                exp_moved.append((b[1], b[2], n))

    people = t['people']
    rep_left = [(p['x'], p['y'], p['id']) for p in people if p['status'] == 'not_found_in_latest']
    rep_new = [(p['x'], p['y'], p['id']) for p in people
               if p['flights_seen'] == [cur] and len(t['meta']['flights']) > 1]
    rep_moved = [(p['x'], p['y'], p['id']) for p in people if any(c.startswith('moved') for c in p['changes'])]

    print(f'twin flight {cur}; world before -> after: {args[0].rsplit("/", 1)[-1]} -> {args[1].rsplit("/", 1)[-1]}')
    for title, exp, rep in (('left / not found', exp_left, rep_left), ('appeared', exp_new, rep_new), ('moved (1.0-1.5 m)', exp_moved, rep_moved)):
        hit, miss, extra = greedy(exp, rep)
        print(f'\n{title}: expected {len(exp)}, reported {len(rep)}, correct {len(hit)}')
        for i, j, d in hit:
            print(f'  ok      {exp[i][2]}  <->  twin #{rep[j][2]} ({d:.2f} m)')
        for e in miss:
            print(f'  MISSED  {e[2]} at ({e[0]:.1f}, {e[1]:.1f})  (not seen by the drone, or area not re-observed)')
        for r in extra:
            print(f'  EXTRA   twin #{r[2]} at ({r[0]:.1f}, {r[1]:.1f})')

    wb = {n: p for n, _, p in bt.read_world(args[0]) if n and not n.startswith(('person', 'survivor', 'casualty', 'randy'))}
    wa = {n: p for n, _, p in bt.read_world(args[1]) if n and not n.startswith(('person', 'survivor', 'casualty', 'randy'))}
    added = [(n, wa[n][0], wa[n][1]) for n in wa if n not in wb]
    removed = [(n, wb[n][0], wb[n][1]) for n in wb if n not in wa]
    regions = (t.get('grid_changes') or {}).get('regions', [])
    print(f"\nheight map: {(t.get('grid_changes') or {}).get('changed_cells', 0)} changed cells in {len(regions)} region(s)")
    for kind, objs in (('added', added), ('removed', removed)):
        for n, x, y in objs:
            near = [g for g in regions if math.hypot(g['x'] - x, g['y'] - y) <= 3.0]
            print(f"  {kind} object {n} at ({x:.1f}, {y:.1f}): " +
                  (f"found region at ({near[0]['x']}, {near[0]['y']}), {near[0]['area_m2']} m2, {near[0]['kind']}" if near
                   else 'NOT found in the changed cells (outside the new coverage?)'))
    other = [g for g in regions if not any(math.hypot(g['x'] - x, g['y'] - y) <= 3.0 for _, x, y in added + removed)]
    if other:
        print(f'  {len(other)} other changed region(s) not explained by an added or removed model: '
              + ', '.join(f"({g['x']}, {g['y']}) {g['area_m2']} m2" for g in other[:6]))


if __name__ == '__main__':
    main()
