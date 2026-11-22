#!/usr/bin/env python3

import argparse
import glob
import json
import math
import re
import xml.etree.ElementTree as ET

import importlib.util

import numpy as np

_spec = importlib.util.spec_from_file_location('twin_grid', __file__.rsplit('/', 1)[0] + '/twin_grid.py')
twin_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(twin_grid)

ROOT = __file__.rsplit('/tools/', 1)[0] + '/src/semantic_twin_assets/models/'
BUILDING_MESHES = {'collapsed_industrial_building': 'industrial building',
                   'collapsed_police_station': 'police station',
                   'collapsed_fire_station': 'fire station'}
VEHICLE_MODELS = {'ambulance', 'suv', 'pickup', 'hatchback', 'bus', 'dumpster'}
SOLID_HEIGHT = 1.0      # a ruin cell counts as blocked above this height 
DEBRIS_LINK = 2.5       # rubble pieces closer than this belong to one debris field
VEHICLE_R = 2.0
TREE_R = 1.5
HAZARD_NEAR = 3.0       # person this close to fire counts as endangered
GRID = 1.0


def launch_world():
    base = __file__.rsplit('/tools/', 1)[0] + '/src/semantic_twin/'
    m = re.search(r'name="world_file" default="([^"]+)"', open(base + 'launch/disaster_launch.xml').read())
    return base + 'worlds/' + (m.group(1) if m else 'disaster_scene.world')


def parse_pose(text):
    v = [float(t) for t in (text or '0 0 0 0 0 0').split()]
    return v + [0.0] * (6 - len(v))


def building_cells(mesh_name, px, py):
    cells = twin_grid.mesh_height_cells(mesh_name, px, py)
    return {k: h for k, h in cells.items() if h >= SOLID_HEIGHT}


def read_world(path):
    w = ET.parse(path).getroot().find('world')
    items = []
    for m in w.findall('model'):
        uris = [e.text for e in m.iter('uri')]
        items.append((m.get('name'), uris[0].split('/')[2] if uris else '', parse_pose(m.findtext('pose'))))
    for i in w.findall('include'):
        uri = i.findtext('uri') or ''
        items.append((i.findtext('name'), uri.replace('model://', '').split('/')[0], parse_pose(i.findtext('pose'))))
    return items


def cluster_points(points, link):
    groups = []
    for p in points:
        near = [g for g in groups if any(math.dist(p, q) < link for q in g)]
        merged = [p]
        for g in near:
            merged += g
            groups.remove(g)
        groups.append(merged)
    return groups


def build_static(world_path):
    items = read_world(world_path)
    static = {'buildings': [], 'debris_fields': [], 'hazards': [], 'vehicles': [], 'trees': []}
    blocked_discs = []   # (x, y, r)
    blocked_cells = set()

    for name, mesh, pose in items:
        x, y = pose[0], pose[1]
        if mesh in BUILDING_MESHES:
            cells = building_cells(mesh, x, y)
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            static['buildings'].append({
                'name': name, 'label': BUILDING_MESHES[mesh], 'x': x, 'y': y,
                'bbox': [min(xs), max(xs) + 1, min(ys), max(ys) + 1],
                'max_height': round(max(cells.values()), 1),
                'cells': sorted([list(c) for c in cells])})
            blocked_cells |= set(cells)
        elif name.startswith(('fire_', 'smoke_')):
            static['hazards'].append({'name': name, 'type': name.split('_')[0], 'x': x, 'y': y})
        elif mesh in VEHICLE_MODELS:
            static['vehicles'].append({'name': name, 'x': x, 'y': y})
            blocked_discs.append((x, y, VEHICLE_R))
        elif mesh in ('pine_tree', 'oak_tree'):
            static['trees'].append({'name': name, 'x': x, 'y': y})
            blocked_discs.append((x, y, TREE_R))

    rubble = [(p[0], p[1]) for n, _, p in items if n and n.startswith(('rubble_', 'column_', 'rebar_'))]
    for i, g in enumerate(sorted(cluster_points(rubble, DEBRIS_LINK), key=lambda g: (np.mean([p[0] for p in g]), np.mean([p[1] for p in g])))):
        cx, cy = float(np.mean([p[0] for p in g])), float(np.mean([p[1] for p in g]))
        r = max(math.dist((cx, cy), p) for p in g) + 0.5
        static['debris_fields'].append({'name': f'debris_{i}', 'x': round(cx, 2), 'y': round(cy, 2),
                                        'radius': round(r, 2), 'pieces': len(g)})
        blocked_discs.append((cx, cy, r))
    return static, blocked_cells, blocked_discs


def open_ground_distance(x, y, blocked_cells, blocked_discs):
    def free(px, py):
        if (math.floor(px), math.floor(py)) in blocked_cells:
            return False
        return all(math.hypot(px - cx, py - cy) > r for cx, cy, r in blocked_discs)
    if free(x, y):
        return 0.0
    best = None
    for ring in range(1, 60):
        d = ring * 0.5
        for k in range(0, 360, 10):
            px, py = x + d * math.cos(math.radians(k)), y + d * math.sin(math.radians(k))
            if free(px, py):
                best = d
                break
        if best is not None:
            return best
    return None


def nearest(items, x, y):
    if not items:
        return None
    it = min(items, key=lambda i: math.hypot(i['x'] - x, i['y'] - y))
    return it, math.hypot(it['x'] - x, it['y'] - y)


def relations(person, static, blocked_cells, blocked_discs):
    x, y = person['x'], person['y']
    rel = {}
    cell = (math.floor(x), math.floor(y))
    inside = [b['name'] for b in static['buildings'] if tuple(cell) in {tuple(c) for c in b['cells']}]
    rel['inside_building'] = inside[0] if inside else None
    dist = {b['name']: min(math.hypot(c[0] + 0.5 - x, c[1] + 0.5 - y) for c in b['cells']) for b in static['buildings']}
    nb = min(dist, key=dist.get)
    rel['nearest_building'] = {'name': nb, 'distance_m': round(dist[nb], 1)}
    df = [d for d in static['debris_fields'] if math.hypot(d['x'] - x, d['y'] - y) <= d['radius']]
    rel['in_debris_field'] = df[0]['name'] if df else None
    hz = nearest([h for h in static['hazards'] if h['type'] == 'fire'], x, y)
    rel['nearest_fire_m'] = round(hz[1], 1) if hz else None
    sm = nearest([h for h in static['hazards'] if h['type'] == 'smoke'], x, y)
    rel['nearest_smoke_m'] = round(sm[1], 1) if sm else None
    og = open_ground_distance(x, y, blocked_cells, blocked_discs)
    rel['open_ground_m'] = None if og is None else round(og, 1)
    return rel


def relations_sensed(person, static, cls, g, structs):
    x, y = person['x'], person['y']
    inside, best, og = twin_grid.sensed_relations(person, cls, g, structs)
    rel = {'inside_building': inside, 'in_debris_field': None,
           'nearest_building': {'name': best[0], 'distance_m': round(best[1], 1)} if best else {'name': None, 'distance_m': None}}
    hz = nearest([h for h in static['hazards'] if h['type'] == 'fire'], x, y)
    rel['nearest_fire_m'] = round(hz[1], 1) if hz else None
    sm = nearest([h for h in static['hazards'] if h['type'] == 'smoke'], x, y)
    rel['nearest_smoke_m'] = round(sm[1], 1) if sm else None
    rel['open_ground_m'] = None if og is None else round(og, 1)
    return rel


def triage(person, rel):
    """Explainable heuristic, not a medical assessment. Higher score = look at first."""
    score, why = 0, []
    if person['role'] == 'casualty':
        score += 3
        why.append('lying (possible casualty)')
    if rel['nearest_fire_m'] is not None and rel['nearest_fire_m'] <= HAZARD_NEAR:
        score += 3
        why.append(f"{rel['nearest_fire_m']} m from fire")
    if rel['inside_building'] or rel['in_debris_field']:
        score += 2
        why.append('inside ruin/debris' if rel['inside_building'] else 'in debris field')
    if rel['open_ground_m'] is not None and rel['open_ground_m'] > 3.0:
        score += 1
        why.append(f"{rel['open_ground_m']} m to open ground")
    level = 'high' if score >= 4 else 'medium' if score >= 2 else 'low'
    return {'score': score, 'level': level, 'reasons': why}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map', nargs='?', default='/tmp/sdt_semantic_map.json')
    ap.add_argument('world', nargs='?', default=launch_world(), help='default: the world disaster_launch.xml loads')
    ap.add_argument('-o', '--out', default='/tmp/sdt_twin.json')
    ap.add_argument('--all', action='store_true', help='include unconfirmed clusters')
    ap.add_argument('--grid', help='height grid exported by map_node: use the sensed map for buildings/ruins/open ground')
    a = ap.parse_args()

    static, cells, discs = build_static(a.world)
    people_in = json.load(open(a.map))
    sensed = None
    if a.grid:
        g = twin_grid.load_grid(a.grid)
        confirmed = [p for p in people_in if a.all or p.get('confirmed', True)]
        cls = twin_grid.classify(g, confirmed)
        structs = twin_grid.structures(cls, g)
        for st in structs:
            st['members_set'] = set(st['members'])
            mask = np.zeros(cls.shape, dtype=bool)
            for yy, xx in st['members']:
                mask[yy, xx] = True
            st['mask'] = mask
        sensed = (g, cls, structs)
    people = []
    for p in people_in:
        if not (a.all or p.get('confirmed', True)):
            continue
        p = dict(p)
        p.setdefault('posture', 'unknown')
        p.setdefault('role', 'unknown')
        rel = relations_sensed(p, static, sensed[1], sensed[0], sensed[2]) if sensed else relations(p, static, cells, discs)
        people.append({
            'id': p['id'], 'x': round(p['x'], 2), 'y': round(p['y'], 2), 'z': round(p['z'], 2),
            'posture': p['posture'], 'role': p['role'],
            'confidence': round(p.get('confidence', 0.0), 2), 'detections': p.get('detections'),
            'confirmed': p.get('confirmed', True),
            'relations': rel, 'triage': triage(p, rel)})
    people.sort(key=lambda q: -q['triage']['score'])

    out = {
        'meta': {'frame': 'map', 'world': a.world.rsplit('/', 1)[-1], 'source_map': a.map,
                 'static_layer': ('buildings/ruins/open ground from the drone\'s depth height map; fire/smoke from the scene description'
                                  if sensed else 'from scene description (stand-in for a known pre-disaster map)'),
                 'dynamic_layer': 'from drone detections',
                 'triage': 'heuristic: lying +3, fire <=3 m +3, in ruin/debris +2, >3 m from open ground +1'},
        'summary': {'people': len(people),
                    'casualties': sum(q['role'] == 'casualty' for q in people),
                    'survivors': sum(q['role'] == 'survivor' for q in people),
                    'high_priority': sum(q['triage']['level'] == 'high' for q in people)},
        'static': {k: ([{kk: vv for kk, vv in b.items() if kk != 'cells'} | {'cell_count': len(b['cells'])}
                        for b in v] if k == 'buildings' else v) for k, v in static.items()},
        'people': people}
    out['static']['building_cells'] = {b['name']: b['cells'] for b in static['buildings']}
    if sensed:
        g, cls, structs = sensed
        out['static']['grid'] = {'resolution': g['resolution'], 'x0': g['x0'], 'y0': g['y0'], 'nx': g['nx'], 'ny': g['ny'],
                                 'blocked_height': g['blocked_height'], 'rows': twin_grid.rows(cls)}
        out['static']['structures'] = [{k: v for k, v in st.items() if k not in ('members', 'members_set', 'mask')} for st in structs]
        out['static']['building_cells'] = {}
        out['static']['buildings'] = []
        out['static']['debris_fields'] = []
        out['summary']['observed_cells'] = int((cls > 0).sum())
        out['summary']['structures'] = len(structs)
    json.dump(out, open(a.out, 'w'), indent=1)

    print(f"twin written to {a.out}: {out['summary']}")
    if sensed:
        print(f"sensed map: {out['summary']['observed_cells']} observed cells, {out['summary']['structures']} structures")
    print(f"static: {len(static['buildings'])} buildings, {len(static['debris_fields'])} debris fields, "
          f"{len(static['hazards'])} hazards, {len(static['vehicles'])} vehicles, {len(static['trees'])} trees")
    for q in people:
        r = q['relations']
        print(f"  #{q['id']} ({q['x']:6.1f},{q['y']:6.1f}) {q['posture']:7s} {q['role']:9s} "
              f"{q['triage']['level']:6s} {r['inside_building'] or r['in_debris_field'] or '-':18s} "
              f"nearest {r['nearest_building']['name']} {r['nearest_building']['distance_m']} m, "
              f"fire {r['nearest_fire_m']} m, open ground {r['open_ground_m']} m")


if __name__ == '__main__':
    main()
