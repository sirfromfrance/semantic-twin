
import json
import math
import re
import time
import xml.etree.ElementTree as ET

import numpy as np

SCHEMA = 'sdt-twin/2'
TALL = 3.0
MIN_STRUCTURE_CELLS = 6
HAZARD_NEAR = 3.0
MATCH_RADIUS = 1.5     
MIN_PERSON_Z = -0.1     # person centre below the ground surface: not a person
DUP_RADIUS = 1.0        # two clusters this close in one flight are one person if one is weak
DUP_RATIO = 3.0         # ...and the stronger one has at least this many times the detections
MOVED_DIST = 1.0        
CHANGE_HEIGHT = 1.0   
CHANGE_MIN_POINTS = 8 
VEG_MIN_HEIGHT = 2.5      # a cell is foliage only if it is at least this high (trees; not a green dumpster)
VEG_MIN_POINTS = 4        # points above 1 m needed in a cell before its colour counts
VEG_FRACTION = 0.5        # share of those points that must be foliage-coloured
FIRE_CELL_MIN = 2         # fire-coloured points for a cell to count as hot
FIRE_CLUSTER_MIN = 6      # fire-coloured points for a cluster of hot cells to be reported as a fire
HAZARD_MATCH_M = 3.0      # a sensed fire this close to a prior fire is the same fire
LEVELS = {'low': 0, 'medium': 1, 'high': 2}


def grid_from_dict(g):
    g = dict(g)
    g['z'] = np.array([[np.nan if v is None else v for v in row] for row in g['max_z']], dtype=float)
    g['n'] = np.array(g['count'], dtype=int) if g.get('count') is not None else None
    for k in ('hi', 'veg', 'fire'):
        g[k] = np.array(g[k], dtype=int) if g.get(k) is not None else None
    return g


def load_grid(path):
    return grid_from_dict(json.load(open(path)))


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


SEED_ERODE = 2         
SEED_MIN_CELLS = 12


def _erode(mask, k):
    out = mask.copy()
    for _ in range(k):
        nxt = np.ones_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nxt &= np.roll(np.roll(out, dy, 0), dx, 1)
        out = nxt
    return out


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack, comp = [(int(sy), int(sx))], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        comps.append(comp)
    return comps


def veg_mask_from(hi, veg, z):
    """Foliage cells: tall, enough points above 1 m, and most of those points foliage-coloured."""
    return (hi >= VEG_MIN_POINTS) & (veg >= VEG_FRACTION * hi) & (np.nan_to_num(z, nan=-9.0) >= VEG_MIN_HEIGHT)


def structures(cls, g):
   
    from collections import deque
    veg = g.get('veg_mask')
    vegc = (veg & (cls >= 3)) if veg is not None else np.zeros(cls.shape, dtype=bool)
    blocked = (cls >= 3) & ~vegc
    label = np.full(cls.shape, -1, dtype=int)
    queue = deque()
    seeds = [c for c in _components(_erode((cls >= 4) & ~vegc, SEED_ERODE)) if len(c) >= SEED_MIN_CELLS]
    for i, comp in enumerate(seeds):
        for y, x in comp:
            label[y, x] = i
            queue.append((y, x))
    while queue:                     
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < cls.shape[0] and 0 <= xx < cls.shape[1] and blocked[yy, xx] and label[yy, xx] < 0:
                    label[yy, xx] = label[y, x]
                    queue.append((yy, xx))
    groups = [[(int(y), int(x)) for y, x in zip(*np.nonzero(label == i))] for i in range(len(seeds))]
    rest = blocked & (label < 0)
    groups += [c for c in _components(rest) if len(c) >= MIN_STRUCTURE_CELLS]
    veg_groups = [c for c in _components(vegc) if len(c) >= MIN_STRUCTURE_CELLS]
    res = g['resolution']
    out = []
    tagged = [(c, False) for c in groups] + [(c, True) for c in veg_groups]
    for i, (comp, is_veg) in enumerate(sorted(tagged, key=lambda t: -len(t[0]))):
        ys, xs = zip(*comp)
        mask = np.zeros(cls.shape, dtype=bool)
        for y, x in comp:
            mask[y, x] = True
        out.append({
            'name': f'structure_{i}', 'cells': len(comp), 'area_m2': round(len(comp) * res * res, 1),
            'x': round(g['x0'] + (np.mean(xs) + 0.5) * res, 2), 'y': round(g['y0'] + (np.mean(ys) + 0.5) * res, 2),
            'max_height': round(float(max(g['z'][y, x] for y, x in comp)), 1),
            'bbox': [round(g['x0'] + min(xs) * res, 2), round(g['x0'] + (max(xs) + 1) * res, 2),
                     round(g['y0'] + min(ys) * res, 2), round(g['y0'] + (max(ys) + 1) * res, 2)],
            '_members': set(comp), '_mask': mask, '_veg': is_veg})
        out[-1]['kind'], out[-1]['kind_reason'] = label_structure(out[-1])
    return out


def label_structure(st):
    """Rule-based environment label from the sensed footprint and height. A guess, not a recognition."""
    area, h = st['area_m2'], st['max_height']
    if st.get('_veg'):
        return 'vegetation', f'{area} m2, {h} m high, foliage-coloured in the camera images'
    if h > 3.0 and area < 8.0:
        return 'spike_artifact', 'tall but tiny: probably depth noise at an edge'
    if area >= 100.0 and h >= 3.0:
        return 'building_ruin', f'{area} m2 footprint, {h} m high'
    if h > 4.5 and area < 100.0:
        return 'tall_object', f'{area} m2 footprint, {h} m high (tree, pole or a wall fragment)'
    if area >= 8.0 and h <= 4.5:
        return 'vehicle_or_large_object', f'{area} m2 footprint, {h} m high'
    return 'small_obstacle', f'{area} m2 footprint, {h} m high'


def public(struct):
    return {k: v for k, v in struct.items() if not k.startswith('_')}


def cell_centres(g, mask):
    ys, xs = np.nonzero(mask)
    return g['x0'] + (xs + 0.5) * g['resolution'], g['y0'] + (ys + 0.5) * g['resolution']


def hazards_from_world(path):
    out = []
    try:
        w = ET.parse(path).getroot().find('world')
    except (OSError, ET.ParseError):
        return out
    for m in w.findall('model'):
        n = m.get('name') or ''
        if n.startswith(('fire_', 'smoke_')):
            v = [float(t) for t in (m.findtext('pose') or '0 0 0').split()]
            out.append({'name': n, 'type': n.split('_')[0], 'x': v[0], 'y': v[1], 'source': 'scene description'})
    return out


def _nearest(items, x, y):
    if not items:
        return None
    it = min(items, key=lambda i: math.hypot(i['x'] - x, i['y'] - y))
    return round(math.hypot(it['x'] - x, it['y'] - y), 1)


def relations(person, cls, g, structs, hazards):
    x, y = person['x'], person['y']
    rel = {'inside_structure': None, 'inside_kind': None, 'nearest_structure': {'name': None, 'distance_m': None},
           'nearest_fire_m': _nearest([h for h in hazards if h['type'] == 'fire'], x, y),
           'nearest_smoke_m': _nearest([h for h in hazards if h['type'] == 'smoke'], x, y),
           'open_ground_m': None}
    if g is None:
        return rel
    structs = [s for s in structs if s['kind'] != 'spike_artifact']
    res = g['resolution']
    ix, iy = int((x - g['x0']) // res), int((y - g['y0']) // res)
    on_grid = 0 <= iy < cls.shape[0] and 0 <= ix < cls.shape[1]
    if on_grid and cls[iy, ix] >= 3:
        for s in structs:
            if (iy, ix) in s['_members']:
                rel['inside_structure'] = s['name']
                rel['inside_kind'] = s['kind']
    best = None
    for s in structs:
        mx, my = cell_centres(g, s['_mask'])
        d = float(np.min(np.hypot(mx - x, my - y)))
        if best is None or d < best[1]:
            best = (s['name'], d)
    if best:
        kind = next(s['kind'] for s in structs if s['name'] == best[0])
        rel['nearest_structure'] = {'name': best[0], 'kind': kind, 'distance_m': round(best[1], 1)}
    if on_grid and cls[iy, ix] == 1:
        rel['open_ground_m'] = 0.0
    else:
        fx, fy = cell_centres(g, cls == 1)
        if len(fx):
            rel['open_ground_m'] = round(float(np.min(np.hypot(fx - x, fy - y))), 1)
    return rel


def triage(person, rel, evac=None):
    """Explainable heuristic, not a medical assessment. Higher score = look at first."""
    score, why = 0, []
    if person['role'] == 'casualty':
        score += 3
        why.append('lying (possible casualty)')
    if rel['nearest_fire_m'] is not None and rel['nearest_fire_m'] <= HAZARD_NEAR:
        score += 3
        why.append(f"{rel['nearest_fire_m']} m from fire")
    if rel['inside_structure'] and rel.get('inside_kind') == 'building_ruin':
        score += 2
        why.append(f"inside {rel['inside_structure']} (building ruin)")
    if rel['open_ground_m'] is not None and rel['open_ground_m'] > 3.0:
        score += 1
        why.append(f"{rel['open_ground_m']} m to open ground")
    if evac is not None:
        if evac['status'] in ('unreachable', 'no_safe_area'):
            score += 3
            why.append('no passable route to a safe area on the sensed map')
        elif evac['status'] == 'ok' and evac['length_m'] > 25.0:
            score += 1
            why.append(f"long evacuation route ({evac['length_m']} m)")
    level = 'high' if score >= 4 else 'medium' if score >= 2 else 'low'
    return {'score': score, 'level': level, 'reasons': why}


def role_of(posture):
    return 'casualty' if posture == 'lying' else 'survivor' if posture == 'upright' else 'unknown'


# ---- evacuation routes
ROBOT_R_CELLS = 1          # obstacles are inflated by this many cells (0.5 m) so a person or stretcher fits
FIRE_BLOCK_M = 2.0         # never route closer than this to a fire
FIRE_COST_M = 5.0          # extra cost closer than this to a fire
STRUCT_COST_M = 2.0        # extra cost this close to a tall structure (falling debris)
SMOKE_COST_M = 3.0
SAFE_CLEARANCE_CELLS = 6   # a safe area has this many free cells (3 m) around it ...
SAFE_FIRE_M = 8.0          # ... and is at least this far from any fire
WALK_SPEED = 1.0           # m/s over open ground
RUBBLE_COST = 4.0          # rubble is this many times slower
RUBBLE_MAX_M = 1.5         # a heap up to this height can be climbed; anything taller is a wall
START_FREE_M = 1.5         # the sensor cannot see inside or under things: the area around the start is passable


def _dilate(mask, k):
    out = mask.copy()
    for _ in range(k):
        nxt = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nxt |= np.roll(np.roll(out, dy, 0), dx, 1)
        out = nxt
    return out


def _rdp(points, tol):
    """Douglas-Peucker simplification of a polyline."""
    if len(points) < 3:
        return points
    a, b = np.array(points[0]), np.array(points[-1])
    ab = b - a
    n = float(np.hypot(*ab))
    d = [float(np.hypot(*(np.array(p) - a))) if n == 0 else abs(ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])) / n
         for p in points]
    i = int(np.argmax(d))
    if d[i] > tol:
        return _rdp(points[:i + 1], tol)[:-1] + _rdp(points[i:], tol)
    return [points[0], points[-1]]


def plan_route(cls, g, start, hazards, goal=None, walk_speed=WALK_SPEED):
    """Cheapest route over the sensed grid from start (x, y) to the nearest safe area, or to `goal` (x, y).

    Passable: observed free ground and low obstacles (rubble, RUBBLE_COST times slower). Not passable:
    obstacles (inflated by 0.5 m), never-seen cells, and the area within FIRE_BLOCK_M of a fire. Extra cost
    near fire, smoke and tall structures. The area around the start is always passable."""
    import heapq
    res = g['resolution']
    ny, nx = cls.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    cx, cy = g['x0'] + (xs + 0.5) * res, g['y0'] + (ys + 0.5) * res
    fires = [(h['x'], h['y']) for h in hazards if h['type'] == 'fire']
    smokes = [(h['x'], h['y']) for h in hazards if h['type'] == 'smoke']

    def dist_to(points):
        d = np.full(cls.shape, np.inf)
        for px, py in points:
            d = np.minimum(d, np.hypot(cx - px, cy - py))
        return d
    d_fire, d_smoke = dist_to(fires), dist_to(smokes)
    z = g['z']
    unknown = np.isnan(z)
    wall = ~unknown & (z >= RUBBLE_MAX_M)
    low = ~unknown & (z >= 0.5) & (z < RUBBLE_MAX_M)
    tall = ~unknown & (z >= TALL)
    blocked = wall
    impass = _dilate(blocked, ROBOT_R_CELLS) | unknown | (d_fire < FIRE_BLOCK_M)
    sx, sy = start
    impass &= ~(np.hypot(cx - sx, cy - sy) <= START_FREE_M)
    tall_near = _dilate(tall, int(STRUCT_COST_M / res))
    cost = np.ones(cls.shape)
    cost[low] = RUBBLE_COST
    cost += 3.0 * tall_near
    cost += 4.0 * (d_fire < FIRE_COST_M)
    cost += 2.0 * (d_smoke < SMOKE_COST_M)

    ix, iy = int((sx - g['x0']) // res), int((sy - g['y0']) // res)
    out = {'status': 'outside_map', 'path': [], 'length_m': None, 'walk_time_s': None, 'goal': None,
           'min_fire_distance_m': None, 'rubble_cells': 0, 'near_tall_structure_m': 0.0,
           'starts_inside_obstacle': False, 'notes': []}
    if not (0 <= iy < ny and 0 <= ix < nx):
        return out
    out['starts_inside_obstacle'] = bool(blocked[iy, ix] and not np.isnan(z[iy, ix]))

    if goal is None:
        bad = blocked | low | unknown
        safe = ~unknown & (z < 0.5) & ~_dilate(bad, SAFE_CLEARANCE_CELLS) & (d_fire >= SAFE_FIRE_M)
    else:
        safe = np.zeros(cls.shape, dtype=bool)
        gx, gy = int((goal[0] - g['x0']) // res), int((goal[1] - g['y0']) // res)
        if 0 <= gy < ny and 0 <= gx < nx:
            safe[gy, gx] = True
            impass[gy, gx] = False
    if not safe.any():
        out['status'] = 'no_safe_area'
        out['notes'].append('the sensed map has no open area with enough clearance')
        return out

    dist = np.full(cls.shape, np.inf)
    prev = {}
    dist[iy, ix] = 0.0
    heap = [(0.0, iy, ix)]
    end = None
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        if safe[y, x]:
            end = (y, x)
            break
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < ny and 0 <= xx < nx and not impass[yy, xx]:
                    nd = d + (1.4142 if dy and dx else 1.0) * 0.5 * (cost[y, x] + cost[yy, xx])
                    if nd < dist[yy, xx]:
                        dist[yy, xx] = nd
                        prev[(yy, xx)] = (y, x)
                        heapq.heappush(heap, (nd, yy, xx))
    if end is None:
        out['status'] = 'unreachable'
        out['notes'].append('no passable route over the sensed map (blocked by obstacles, fire or unseen ground)')
        return out
    cells = [end]
    while cells[-1] != (iy, ix):
        cells.append(prev[cells[-1]])
    cells = cells[::-1]
    pts = [(float(cx[y, x]), float(cy[y, x])) for y, x in cells]
    seg = [float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(pts, pts[1:])]
    length = sum(seg)
    rubble = int(sum(low[y, x] for y, x in cells))
    near_tall = sum(l for l, (y, x) in zip(seg, cells[1:]) if tall_near[y, x])
    out.update({
        'status': 'ok', 'path': [[round(a, 2), round(b, 2)] for a, b in _rdp(pts, 0.4)],
        'length_m': round(length, 1),
        'walk_time_s': round((length + rubble * res * (RUBBLE_COST - 1)) / walk_speed),
        'goal': [round(pts[-1][0], 1), round(pts[-1][1], 1)],
        'min_fire_distance_m': round(float(min(d_fire[y, x] for y, x in cells)), 1) if fires else None,
        'rubble_cells': rubble, 'near_tall_structure_m': round(near_tall, 1)})
    if out['starts_inside_obstacle']:
        out['notes'].append('starts inside an obstacle: the way out of it is not visible to the sensor')
    if rubble:
        out['notes'].append(f'crosses {rubble} rubble cells')
    return out


def screen_observations(obs):
    """Split one flight's observations (dicts with x, y, z, n) into kept and rejected.
    Rejected: centre below ground, or a weak cluster next to a much stronger one."""
    kept, rejected = [], []
    for o in obs:
        if o['z'] < MIN_PERSON_Z:
            rejected.append((o, 'below_ground'))
        else:
            kept.append(o)
    final = []
    for o in kept:
        strong = [q for q in kept if q is not o and q['n'] >= DUP_RATIO * o['n']
                  and math.hypot(q['x'] - o['x'], q['y'] - o['y']) < DUP_RADIUS]
        if strong:
            rejected.append((o, 'duplicate_of_stronger_cluster'))
        else:
            final.append(o)
    return final, rejected


class Twin:
    def __init__(self, hazards=None, match_radius=MATCH_RADIUS, use_sensed_fire=False):
        self.match_radius = match_radius
        self.use_sensed_fire = use_sensed_fire   # off: colour-sensed fire was not reliable in the tests
        self.hazards = list(hazards or [])
        self.flights = []          
        self.people = {}           
        self.people_flights = set()  
        self.rejected = []
        self.next_id = 1
        self.res = None           
        self.z = None              
        self.z_flight = None       
        self.base_z = None        
        self.base_n = None        
        self.hi = self.veg = self.fire = None
        self.base_hi = self.base_veg = self.base_fire = None
        self.n = None           
        self.cur_n = None
        self.cur_z = None
        self.base_flight = None
        self.updated = None

    def _flight_empty(self, idx):
        name = self.flights[idx]['name']
        return (not any(o['flight'] == name for p in self.people.values() for o in p['obs'].values())
                and (self.z_flight is None or not (self.z_flight == idx).any()))

    def begin_flight(self, name=None, reuse_empty=False):
        if reuse_empty and self.flights and self._flight_empty(len(self.flights) - 1):
            return self.flights[-1]['name']
        name = name or f'flight_{len(self.flights) + 1}'
        self.flights.append({'name': name, 'started': time.strftime('%Y-%m-%d %H:%M:%S')})
        if self.z is not None:
            self.base_z, self.base_flight = self.z.copy(), self.z_flight.copy()
            self.base_n = self.n.copy() if self.n is not None else None
            for k in ('hi', 'veg', 'fire'):
                v = getattr(self, k)
                setattr(self, 'base_' + k, None if v is None else v.copy())
        return name

    @property
    def flight(self):
        if not self.flights:
            self.begin_flight()
        return self.flights[-1]['name']

    def update_people(self, observations):
      
        flight = self.flight
        self.people_flights.add(flight)
        cur, prev_owner = {}, {}
        for pid, p in self.people.items():
            for k, o in p['obs'].items():
                if o['flight'] == flight:
                    cur[k] = o
                    prev_owner[k] = pid
        for o in observations:
            if not o.get('confirmed', True):
                continue
            cur[f"{flight}:{o['id']}"] = {
                'flight': flight, 'x': float(o['x']), 'y': float(o['y']), 'z': float(o.get('z', 0.0)),
                'posture': o.get('posture', 'unknown'), 'n': int(o.get('detections') or 1),
                'confidence': float(o.get('confidence', 0.0)), 'z_extent': o.get('z_extent')}
        keys = list(cur)
        kept, rej = screen_observations([cur[k] for k in keys])
        rej_ids = {id(o): why for o, why in rej}
        self.rejected = [r for r in self.rejected if r['flight'] != flight]
        for k in keys:
            if id(cur[k]) in rej_ids:
                self.rejected.append({'key': k, 'reason': rej_ids[id(cur[k])], **cur[k]})
                del cur[k]
        for p in self.people.values():
            p['obs'] = {k: o for k, o in p['obs'].items() if o['flight'] != flight}

        pairs = []
        for k, o in cur.items():
            for pid, p in self.people.items():
                for sm in self._flight_summaries(p).values():
                    d = math.hypot(sm['x'] - o['x'], sm['y'] - o['y'])
                    if d < self.match_radius:
                        pairs.append((d, k, pid))
        used_keys, used_people = set(), set()
        for d, k, pid in sorted(pairs):
            if k in used_keys or pid in used_people:
                continue
            self.people[pid]['obs'][k] = cur[k]
            used_keys.add(k)
            used_people.add(pid)
        for k, o in cur.items():
            if k in used_keys:
                continue
            old = prev_owner.get(k)
            if old in self.people and not self.people[old]['obs'] and old not in used_people:
                pid = old
            else:
                pid = self.next_id
                self.next_id += 1
                self.people[pid] = {'id': pid, 'obs': {}}
            self.people[pid]['obs'][k] = o
            used_people.add(pid)
        self.people = {pid: p for pid, p in self.people.items() if p['obs']}
        self.updated = time.strftime('%Y-%m-%d %H:%M:%S')

    def clean_people(self):
        """Apply the screening rules to observations already stored (older states)."""
        for fl in {o['flight'] for p in self.people.values() for o in p['obs'].values()}:
            items = [(pid, k, o) for pid, p in self.people.items() for k, o in p['obs'].items() if o['flight'] == fl]
            _, rej = screen_observations([o for _, _, o in items])
            ids = {id(o): why for o, why in rej}
            for pid, k, o in items:
                if id(o) in ids:
                    self.rejected.append({'key': k, 'reason': ids[id(o)], **o})
                    del self.people[pid]['obs'][k]
        self.people = {pid: p for pid, p in self.people.items() if p['obs']}
        return len(self.rejected)

    def _flight_summaries(self, person):
        by = {}
        for o in person['obs'].values():
            by.setdefault(o['flight'], []).append(o)
        out = {}
        for f, obs in by.items():
            w = sum(o['n'] for o in obs)
            votes = {}
            for o in obs:
                votes[o['posture']] = votes.get(o['posture'], 0) + o['n']
            out[f] = {'flight': f, 'x': sum(o['x'] * o['n'] for o in obs) / w,
                      'y': sum(o['y'] * o['n'] for o in obs) / w,
                      'z': sum(o['z'] * o['n'] for o in obs) / w,
                      'posture': max(votes, key=votes.get), 'detections': w,
                      'confidence': max(o['confidence'] for o in obs)}
        return out

    # grid
    def update_grid(self, grid):
       
        self.flight
        meta = {k: grid[k] for k in ('resolution', 'x0', 'y0', 'nx', 'ny', 'blocked_height')}
        if self.res is None:
            self.res = meta
            self.z = np.full(grid['z'].shape, np.nan)
            self.z_flight = np.full(grid['z'].shape, -1, dtype=int)
            self.base_z, self.base_flight = self.z.copy(), self.z_flight.copy()
            self.base_n = np.zeros(grid['z'].shape, dtype=int)
            self.n = self.base_n.copy()
        elif any(abs(self.res[k] - meta[k]) > 1e-9 for k in ('resolution', 'x0', 'y0', 'nx', 'ny')):
            raise ValueError('grid layout differs from the stored one')
        new = grid['z']
        base = self.base_z
        seen = ~np.isnan(new)
        known = seen & ~np.isnan(base)
        merged = base.copy()
        fresh = seen & np.isnan(base)
        merged[fresh] = new[fresh]
        take = known & ((new >= base) | ((base >= 1.0) & (new <= 0.5)))
        merged[take] = new[take]
        n_new = grid.get('n')
        self.cur_z = new.copy()
        self.cur_n = n_new.copy() if n_new is not None else None
        self.n = (self.base_n + n_new) if (n_new is not None and self.base_n is not None) else self.base_n
        self.z = merged
        for k in ('hi', 'veg', 'fire'):
            a = grid.get(k)
            if a is None:
                continue
            base = getattr(self, 'base_' + k)
            if base is None:
                base = np.zeros(a.shape, dtype=int)
                setattr(self, 'base_' + k, base)
            setattr(self, k, base + a)
        self.z_flight = np.where(seen, len(self.flights) - 1, self.base_flight)
        self.updated = time.strftime('%Y-%m-%d %H:%M:%S')

    def _grid_dict(self):
        if self.res is None:
            return None
        g = dict(self.res)
        g['z'] = self.z
        g['veg_mask'] = self.veg_mask()
        return g

    def veg_mask(self):
        """Cells that are tall and whose upper points are mostly foliage-coloured."""
        if self.veg is None or self.hi is None or self.z is None:
            return None
        return veg_mask_from(self.hi, self.veg, self.z)

    def sensed_fire(self):
        """Fires seen by the camera: clusters of hot-coloured cells near the ground."""
        if self.fire is None or self.res is None:
            return []
        hot = self.fire >= FIRE_CELL_MIN
        out = []
        r = self.res
        for comp in _components(_dilate(hot, 1)):
            cells = [(y, x) for y, x in comp if hot[y, x]]
            pts = sum(int(self.fire[y, x]) for y, x in cells)
            if pts < FIRE_CLUSTER_MIN:
                continue
            w = np.array([self.fire[y, x] for y, x in cells], dtype=float)
            out.append({'type': 'fire', 'source': 'sensed', 'points': pts,
                        'x': round(float(r['x0'] + (np.average([x for _, x in cells], weights=w) + 0.5) * r['resolution']), 2),
                        'y': round(float(r['y0'] + (np.average([y for y, _ in cells], weights=w) + 0.5) * r['resolution']), 2)})
        return out

    def all_hazards(self):
        """Prior hazards (scene description) plus fires the camera saw; a sensed fire near a prior one confirms it."""
        prior = [dict(h, source='prior') for h in self.hazards]
        extra = []
        for s in (self.sensed_fire() if self.use_sensed_fire else []):
            near = [h for h in prior if h['type'] == 'fire' and math.hypot(h['x'] - s['x'], h['y'] - s['y']) < HAZARD_MATCH_M]
            if near:
                for h in near:
                    h['source'] = 'prior+sensed'
            else:
                extra.append(s)
        return prior + extra

    def hazard_check(self):
        """How the prior fires compare with what the camera saw (only meaningful where the drone looked)."""
        hz = self.all_hazards()
        unverified = 0 if self.use_sensed_fire else len(self.sensed_fire())
        res = {'sensed_fire_in_use': self.use_sensed_fire, 'sensed_clusters_not_used': unverified, 'prior_fires': 0, 'confirmed': 0, 'not_seen_in_observed_area': 0, 'area_not_observed': 0,
               'sensed_only': sum(h['source'] == 'sensed' for h in hz), 'fire_colour_available': self.fire is not None}
        r = self.res
        for h in hz:
            if h['type'] != 'fire' or h['source'] == 'sensed':
                continue
            res['prior_fires'] += 1
            if self.fire is None:
                res['no_colour_data'] = res.get('no_colour_data', 0) + 1
                continue
            if h['source'] == 'prior+sensed':
                res['confirmed'] += 1
                continue
            ix, iy = int((h['x'] - r['x0']) // r['resolution']), int((h['y'] - r['y0']) // r['resolution'])
            win = self.z[max(iy - 3, 0):iy + 4, max(ix - 3, 0):ix + 4] if self.z is not None else None
            seen = win is not None and win.size and bool((~np.isnan(win)).any())
            res['not_seen_in_observed_area' if seen else 'area_not_observed'] += 1
        return res

    def grid_changes(self):
      
        cur = getattr(self, 'cur_z', None)
        if cur is None or self.base_z is None or len(self.flights) < 2:
            return None
        both = ~np.isnan(cur) & ~np.isnan(self.base_z)
        if self.cur_n is not None and self.base_n is not None:
            both &= (self.cur_n >= CHANGE_MIN_POINTS) & (self.base_n >= CHANGE_MIN_POINTS)
        dz = np.where(both, cur - self.base_z, 0.0)
        added = both & (dz >= CHANGE_HEIGHT)
        removed = both & (self.base_z >= 1.0) & (cur <= 0.5) & (dz <= -CHANGE_HEIGHT)
        lower = both & (dz <= -CHANGE_HEIGHT) & ~removed
        changed = added | removed
        out = {'flight': self.flight, 'compared_cells': int(both.sum()),
               'evidence_filter': f'>= {CHANGE_MIN_POINTS} depth points per cell in both flights' if self.cur_n is not None else 'none (no counts)',
               'changed_cells': int(changed.sum()), 'added': int(added.sum()), 'removed': int(removed.sum()),
               'lower_reading_not_counted': int(lower.sum())}
        if changed.any():
            ys, xs = np.nonzero(changed)
            r = self.res
            out['bbox'] = [round(r['x0'] + xs.min() * r['resolution'], 1), round(r['x0'] + (xs.max() + 1) * r['resolution'], 1),
                           round(r['y0'] + ys.min() * r['resolution'], 1), round(r['y0'] + (ys.max() + 1) * r['resolution'], 1)]
            out['regions'] = self._change_regions(changed, dz)
        return out

    def _change_regions(self, changed, dz, min_cells=6, limit=12):
        r = self.res
        seen = np.zeros(changed.shape, dtype=bool)
        regions = []
        for sy, sx in zip(*np.nonzero(changed)):
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
                        if 0 <= yy < changed.shape[0] and 0 <= xx < changed.shape[1] and changed[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
            if len(comp) >= min_cells:
                ys, xs = zip(*comp)
                mean = float(np.mean([dz[y, x] for y, x in comp]))
                before = float(np.mean([self.base_z[y, x] for y, x in comp]))
                after = float(np.mean([self.cur_z[y, x] for y, x in comp]))
                ground = float(np.mean([self.base_z[y, x] < 0.6 for y, x in comp]))   # share that was open ground before
                regions.append({'x': round(r['x0'] + (np.mean(xs) + 0.5) * r['resolution'], 2),
                                'y': round(r['y0'] + (np.mean(ys) + 0.5) * r['resolution'], 2),
                                'cells': len(comp), 'area_m2': round(len(comp) * r['resolution'] ** 2, 1),
                                'mean_height_change': round(mean, 2), 'height_before_m': round(before, 1),
                                'height_after_m': round(after, 1), 'was_open_ground': round(ground, 2),
                                'kind': 'added (something appeared)' if mean > 0 else 'removed (ground visible where an obstacle was)'})
        return sorted(regions, key=lambda g: -g['cells'])[:limit]

    def snapshot(self):
        g = self._grid_dict()
        cur = self.flight
        cur_idx = len(self.flights) - 1
        ents = []
        for p in self.people.values():
            sums = self._flight_summaries(p)
            order = [f['name'] for f in self.flights if f['name'] in sums]
            last = sums[order[-1]]
            ents.append({'id': p['id'], 'x': round(last['x'], 2), 'y': round(last['y'], 2), 'z': round(last['z'], 2),
                         'posture': last['posture'], 'role': role_of(last['posture']),
                         'confidence': round(last['confidence'], 2), 'detections': last['detections'],
                         'confirmed': True, 'last_flight': order[-1], 'first_flight': order[0],
                         'flights_seen': order,
                         'history': [{'flight': f, 'x': round(sums[f]['x'], 2), 'y': round(sums[f]['y'], 2),
                                      'posture': sums[f]['posture'], 'detections': sums[f]['detections']} for f in order]})
        hz = self.all_hazards()
        cls = structs = None
        if g is not None:
            cls = classify(g, ents)
            structs = structures(cls, g)
        routes = self._routes(ents, cls, g, hz)
        for e in ents:
            seen_now = e['last_flight'] == cur
            if seen_now:
                e['status'] = 'present'
            elif g is not None and cur in self.people_flights and self._observed_in_flight(e, cur_idx):
                e['status'] = 'not_found_in_latest'
            else:
                e['status'] = 'not_reobserved'
            e['relations'] = relations(e, cls, g, structs or [], hz)
            e['evacuation'] = routes.get(e['id'])
            e['triage'] = triage(e, e['relations'], e['evacuation'])
            e['changes'] = self._person_changes(e)
        ents.sort(key=lambda e: (-e['triage']['score'], e['id']))
        static = {'hazards': hz, 'hazard_check': self.hazard_check(), 'buildings': [], 'debris_fields': [], 'vehicles': [], 'trees': [],
                  'building_cells': {}}
        if g is not None:
            static['grid'] = {k: self.res[k] for k in ('resolution', 'x0', 'y0', 'nx', 'ny', 'blocked_height')}
            static['grid']['rows'] = rows(cls)
            static['structures'] = [public(s) for s in structs]
        return {
            'meta': {'schema': SCHEMA, 'frame': 'map', 'updated': self.updated, 'current_flight': cur,
                     'flights': self.flights, 'world': 'sensed',
                     'static_layer': 'structures/ruins/open ground from the drone depth map; fire and smoke from the scene description',
                     'triage': 'heuristic: lying +3, fire <=3 m +3, inside structure +2, >3 m from open ground +1'},
            'summary': {'people': len(ents),
                        'casualties': sum(e['role'] == 'casualty' for e in ents),
                        'survivors': sum(e['role'] == 'survivor' for e in ents),
                        'high_priority': sum(e['triage']['level'] == 'high' for e in ents),
                        'present_in_latest': sum(e['status'] == 'present' for e in ents),
                        'structures': len(structs) if structs is not None else 0,
                        'observed_cells': int((cls > 0).sum()) if cls is not None else 0},
            'people': ents, 'static': static, 'grid_changes': self.grid_changes()}

    def _routes(self, ents, cls, g, hz):
        """Evacuation route per person; cached until the map, the people or the hazards change."""
        if g is None:
            return {}
        import hashlib
        key = (hashlib.md5(cls.tobytes()).hexdigest(), tuple(sorted((e['id'], round(e['x'] * 2), round(e['y'] * 2)) for e in ents)),
               tuple((h['type'], h['x'], h['y']) for h in hz))
        cache = getattr(self, '_route_cache', None)
        if cache and cache[0] == key:
            return cache[1]
        routes = {e['id']: plan_route(cls, g, (e['x'], e['y']), hz) for e in ents}
        self._route_cache = (key, routes)
        return routes

    def _observed_in_flight(self, e, flight_idx):
        r = self.res
        ix, iy = int((e['x'] - r['x0']) // r['resolution']), int((e['y'] - r['y0']) // r['resolution'])
        return (0 <= iy < self.z_flight.shape[0] and 0 <= ix < self.z_flight.shape[1]
                and self.z_flight[iy, ix] == flight_idx)

    @staticmethod
    def _person_changes(e):
        h = e['history']
        if len(h) < 2:
            return []
        a, b = h[-2], h[-1]
        out = []
        d = math.hypot(b['x'] - a['x'], b['y'] - a['y'])
        if d > MOVED_DIST:
            out.append(f"moved {d:.1f} m since {a['flight']}")
        if a['posture'] != b['posture']:
            out.append(f"posture {a['posture']} -> {b['posture']} since {a['flight']}")
        return out

    # ---- queries
    def query(self, req):
        """JSON request -> JSON result. ops: summary, people, near, near_hazard, priority, changes, structure."""
        snap = self.snapshot()
        op = req.get('op', 'summary')
        people = snap['people']
        if op == 'summary':
            return {'ok': True, 'summary': snap['summary'], 'grid_changes': snap['grid_changes']}
        if op == 'people':
            sel = [p for p in people
                   if (not req.get('role') or p['role'] == req['role'])
                   and (not req.get('posture') or p['posture'] == req['posture'])
                   and (not req.get('status') or p['status'] == req['status'])
                   and LEVELS[p['triage']['level']] >= LEVELS[req.get('min_level', 'low')]]
            return {'ok': True, 'count': len(sel), 'people': [self._brief(p) for p in sel]}
        if op == 'near':
            x, y, r = float(req['x']), float(req['y']), float(req.get('radius', 5.0))
            sel = [p for p in people if math.hypot(p['x'] - x, p['y'] - y) <= r]
            return {'ok': True, 'count': len(sel), 'people': [self._brief(p) for p in sel]}
        if op == 'near_hazard':
            r = float(req.get('radius', HAZARD_NEAR))
            sel = [p for p in people if p['relations']['nearest_fire_m'] is not None and p['relations']['nearest_fire_m'] <= r]
            return {'ok': True, 'count': len(sel), 'people': [self._brief(p) for p in sel]}
        if op == 'priority':
            n = int(req.get('n', 5))
            return {'ok': True, 'people': [self._brief(p) for p in people[:n]]}
        if op == 'changes':
            return {'ok': True, 'grid': snap['grid_changes'],
                    'people': [{'id': p['id'], 'status': p['status'], 'changes': p['changes']}
                               for p in people if p['changes'] or p['status'] != 'present']}
        if op == 'route':
            p = next((q for q in people if q['id'] == int(req.get('person', -1))), None)
            if p is None:
                return {'ok': False, 'error': f"unknown person {req.get('person')}"}
            if req.get('to'):
                g = self._grid_dict()
                cls = classify(g, people)
                return {'ok': True, 'person': p['id'], 'route': plan_route(cls, g, (p['x'], p['y']), self.all_hazards(), goal=tuple(req['to']))}
            return {'ok': True, 'person': p['id'], 'route': p['evacuation']}
        if op == 'unreachable':
            sel = [p for p in people if p['evacuation'] and p['evacuation']['status'] != 'ok']
            return {'ok': True, 'count': len(sel), 'people': [self._brief(p) for p in sel]}
        if op == 'structure':
            name = req.get('name')
            st = next((s for s in snap['static'].get('structures', []) if s['name'] == name), None)
            if st is None:
                return {'ok': False, 'error': f'unknown structure {name}'}
            b = st['bbox']
            inside = [p for p in people if p['relations']['inside_structure'] == name
                      or (b[0] - 1 <= p['x'] <= b[1] + 1 and b[2] - 1 <= p['y'] <= b[3] + 1)]
            return {'ok': True, 'structure': st, 'people_near': [self._brief(p) for p in inside]}
        return {'ok': False, 'error': f'unknown op {op}'}

    @staticmethod
    def _brief(p):
        ev = p.get('evacuation') or {}
        return {'id': p['id'], 'x': p['x'], 'y': p['y'], 'posture': p['posture'], 'role': p['role'],
                'status': p['status'], 'priority': p['triage']['level'], 'reasons': p['triage']['reasons'],
                'evacuation': {k: ev.get(k) for k in ('status', 'length_m', 'walk_time_s', 'goal')}}

    # ---- persistence
    def to_state(self):
        def z2l(a):
            return None if a is None else [[None if np.isnan(v) else round(float(v), 2) for v in r] for r in a]
        return {'schema': SCHEMA + '-state', 'flights': self.flights, 'people': list(self.people.values()),
                'next_id': self.next_id, 'people_flights': sorted(self.people_flights), 'rejected': self.rejected, 'colour': {k: (None if getattr(self, k) is None else getattr(self, k).tolist()) for k in ('hi', 'veg', 'fire')}, 'base_colour': {k: (None if getattr(self, 'base_' + k) is None else getattr(self, 'base_' + k).tolist()) for k in ('hi', 'veg', 'fire')}, 'res': self.res, 'z': z2l(self.z), 'base_z': z2l(self.base_z),
                'cur_z': z2l(getattr(self, 'cur_z', None)), 'n': None if self.n is None else self.n.tolist(), 'base_n': None if self.base_n is None else self.base_n.tolist(), 'cur_n': None if self.cur_n is None else self.cur_n.tolist(), 'z_flight': None if self.z_flight is None else self.z_flight.tolist(),
                'base_flight': None if self.base_flight is None else self.base_flight.tolist(),
                'hazards': self.hazards, 'updated': self.updated}

    @classmethod
    def from_state(cls, s):
        t = cls(hazards=s.get('hazards'))
        t.flights, t.next_id, t.res, t.updated = s['flights'], s['next_id'], s['res'], s.get('updated')
        t.people = {p['id']: p for p in s['people']}
        t.people_flights = set(s.get('people_flights', []))
        t.rejected = s.get('rejected', [])
        for src, pre in ((s.get('colour') or {}, ''), (s.get('base_colour') or {}, 'base_')):
            for k, v in src.items():
                setattr(t, pre + k, None if v is None else np.array(v, dtype=int))

        def l2z(l):
            return None if l is None else np.array([[np.nan if v is None else v for v in r] for r in l], dtype=float)
        t.z, t.base_z = l2z(s['z']), l2z(s['base_z'])
        t.cur_z = l2z(s.get('cur_z'))
        for k in ('n', 'base_n', 'cur_n'):
            setattr(t, k, None if s.get(k) is None else np.array(s[k], dtype=int))
        t.z_flight = None if s['z_flight'] is None else np.array(s['z_flight'], dtype=int)
        t.base_flight = None if s['base_flight'] is None else np.array(s['base_flight'], dtype=int)
        return t

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_state(), f)

    @classmethod
    def load(cls, path):
        return cls.from_state(json.load(open(path)))
