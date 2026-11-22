#!/usr/bin/env python3

import json
import math
import re
import sys

HERE = __file__.rsplit('/tools/', 1)[0]
PERSON_RE = r'person|randy|walking|lying|victim|human'


def world_people(path):
    lines = open(path).read().split('\n')
    people, i = [], 0
    while i < len(lines):
        m = re.match(r'^(\s*)<model name="([^"]+)"', lines[i])
        if m:
            ind, name, j = m.group(1), m.group(2), i + 1
            while j < len(lines) and not re.match(r'^' + ind + '</model>', lines[j]):
                j += 1
            body = '\n'.join(lines[i:j])
            if re.search(r'model://[A-Za-z0-9_]*(' + PERSON_RE + ')', body):
                poses = [l for l in lines[i:j] if re.match(r'^' + ind + r'  <pose>', l)]
                if poses:
                    v = re.sub(r'</?pose>', '', poses[-1]).split()
                    x, y = float(v[0]), float(v[1])
                    lying = abs(float(v[3])) > 1.0  # roll
                    if lying:
                       
                      
                        yaw = float(v[5])
                        sign = 1.0 if float(v[3]) > 0 else -1.0
                        x, y = x + sign * 0.955 * math.sin(yaw), y - sign * 0.955 * math.cos(yaw)
                    people.append((name, x, y, 'lying' if lying else 'upright'))
            i = j
        i += 1
    text = open(path).read()
    for m in re.finditer(r'<include>(.*?)</include>', text, re.S):
        b = m.group(1)
        u, n, p = (re.search(r'<uri>model://([^<]+)', b), re.search(r'<name>([^<]+)', b),
                   re.search(r'<pose[^>]*>([^<]+)', b))
        if u and p and re.search(PERSON_RE, u.group(1)):
            v = p.group(1).split()
            people.append((n.group(1) if n else u.group(1), float(v[0]), float(v[1]),
                           'sitting' if 'sitting' in u.group(1) else 'upright'))
    return people


def launch_world():
    base = HERE + '/src/semantic_twin/'
    m = re.search(r'name="world_file" default="([^"]+)"', open(base + 'launch/disaster_launch.xml').read())
    return base + 'worlds/' + (m.group(1) if m else 'disaster_scene.world')


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map', nargs='?', default='/tmp/sdt_semantic_map.json')
    ap.add_argument('world', nargs='?', default=launch_world())
    ap.add_argument('--radius', type=float, default=1.5, help='match radius in metres')
    ap.add_argument('--all', action='store_true', help='also count unconfirmed clusters')
    ap.add_argument('--within', type=float, help='only people within this distance (m) of the origin')
    ap.add_argument('--box', help='only people inside x0,x1,y0,y1 (map frame)')
    ap.add_argument('--names', help='only these people, comma separated (e.g. casualty_1,randy_2)')
    opt = ap.parse_args()
    radius, map_path, world = opt.radius, opt.map, opt.world
    markers = json.load(open(map_path))
    if not opt.all:  
        markers = [m for m in markers if m.get('confirmed', True)]
    people = world_people(world)
    keep = set(range(len(people)))   # indices reported; matching always uses everyone
    if opt.within is not None:
        keep &= {i for i, p in enumerate(people) if math.hypot(p[1], p[2]) <= opt.within}
    if opt.box:
        x0, x1, y0, y1 = map(float, opt.box.split(','))
        keep &= {i for i, p in enumerate(people) if x0 <= p[1] <= x1 and y0 <= p[2] <= y1}
    if opt.names:
        keep &= {i for i, p in enumerate(people) if p[0] in set(opt.names.split(','))}

    pairs = sorted((math.hypot(m['x'] - p[1], m['y'] - p[2]), mi, pi)
                   for mi, m in enumerate(markers) for pi, p in enumerate(people))
    used_m, used_p, match = set(), set(), {}
    for d, mi, pi in pairs:
        if d <= radius and mi not in used_m and pi not in used_p:
            used_m.add(mi); used_p.add(pi); match[pi] = (mi, d)

    print(f'{len(keep)} of {len(people)} people reported, {len(markers)} markers (match radius {radius} m)\n')
    print(f'{"person":16s} {"kind":8s} {"world (x, y)":16s} {"found":6s} {"err m":6s} {"dets":5s}')
    for pi, (name, x, y, kind) in enumerate(people):
        if pi not in keep:
            continue
        if pi in match:
            mi, d = match[pi]
            post = markers[mi].get('posture')
            truth = 'lying' if kind == 'lying' else 'upright'   # sitting is counted as upright
            tag = '' if post is None else f'  posture {post}' + ('' if post == truth else f' (truth {truth}) WRONG')
            print(f'{name:16s} {kind:8s} ({x:6.1f},{y:6.1f})   yes    {d:5.2f}  {markers[mi].get("detections", "")}{tag}')
        else:
            print(f'{name:16s} {kind:8s} ({x:6.1f},{y:6.1f})   NO')
    print('\nmarkers with no person within radius:')
    for mi, m in enumerate(markers):
        if mi not in used_m:
            near = min(people, key=lambda p: math.hypot(m['x'] - p[1], m['y'] - p[2]))
            d = math.hypot(m['x'] - near[1], m['y'] - near[2])
            print(f'  #{m["id"]} ({m["x"]:.2f}, {m["y"]:.2f}, z={m["z"]:.2f}, n={m.get("detections", "")})'
                  f'  nearest: {near[0]} {d:.2f} m')
    errs = [d for pi, (_, d) in match.items() if pi in keep]
    tp, fp = len(errs), len(markers) - len(match)
    print(f'\nfound {tp}/{len(keep)} (recall {tp / max(1, len(keep)):.0%}); '
          f'markers on people outside the reported set: {len(match) - tp}; '
          f'false markers {fp} (precision {len(match) / max(1, len(markers)):.0%})')
    if errs:
        print(f'position error: mean {sum(errs) / len(errs):.2f} m, max {max(errs):.2f} m')
    judged = [(markers[match[pi][0]]['posture'], 'lying' if people[pi][3] == 'lying' else 'upright')
              for pi in match if pi in keep and 'posture' in markers[match[pi][0]]]
    if judged:
        print(f'posture correct: {sum(a == b for a, b in judged)}/{len(judged)} of the matched people '
              f'(sitting counted as upright)')
    for kind in sorted({people[i][3] for i in keep}):
        idx = [i for i in keep if people[i][3] == kind]
        print(f'  {kind:8s}: found {sum(i in match for i in idx)}/{len(idx)}')


if __name__ == '__main__':
    main()
