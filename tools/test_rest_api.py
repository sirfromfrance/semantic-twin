#!/usr/bin/env python3
"""Tests the REST API (rest_api.py) on a saved twin, without ROS.

usage: python3 tools/test_rest_api.py [STATE.json]     (default: results/series4/twin_state.json)
"""
import importlib.util
import json
import sys

HERE = __file__.rsplit('/tools/', 1)[0]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    tc = load(f'{HERE}/src/semantic_twin/semantic_twin/twin_core.py', 'twin_core')
    api = load(f'{HERE}/src/semantic_twin/semantic_twin/rest_api.py', 'rest_api')
    state = sys.argv[1] if len(sys.argv) > 1 else f'{HERE}/results/series4/twin_state.json'
    t = tc.Twin.load(state)
    snap = json.loads(json.dumps(t.snapshot()))          # as the operator node sees it (JSON round trip)
    ask = lambda q: json.loads(json.dumps(t.query(q)))
    pid = snap['people'][0]['id']
    sname = snap['static']['structures'][0]['name']
    p0 = snap['people'][0]

    cases = [  # method, path, query, body, expected code, check
        ('GET', '/api/v1', '', None, 200, lambda b: len(b['endpoints']) >= 15),
        ('GET', '/api/v1/', '', None, 200, None),
        ('GET', '/api/v1/twin', '', None, 200, lambda b: 'people' in b and 'meta' in b),
        ('GET', '/api/v1/summary', '', None, 200, lambda b: b['summary']['people'] == len(snap['people'])),
        ('GET', '/api/v1/flights', '', None, 200, lambda b: len(b['flights']) == len(snap['meta']['flights'])),
        ('GET', '/api/v1/people', '', None, 200, lambda b: b['count'] == len(snap['people'])),
        ('GET', '/api/v1/people', 'posture=lying', None, 200,
         lambda b: b['count'] == sum(p['posture'] == 'lying' for p in snap['people'])),
        ('GET', '/api/v1/people', 'min_level=high', None, 200, lambda b: all(p['priority'] == 'high' for p in b['people'])),
        ('GET', '/api/v1/people', 'min_level=urgent', None, 400, None),
        ('GET', '/api/v1/people', 'colour=red', None, 400, None),
        ('GET', '/api/v1/people/priority', 'n=3', None, 200, lambda b: len(b['people']) == min(3, len(snap['people']))),
        ('GET', '/api/v1/people/priority', 'n=0', None, 400, None),
        ('GET', '/api/v1/people/priority', 'n=abc', None, 400, None),
        ('GET', '/api/v1/people/unreachable', '', None, 200, lambda b: all(p['evacuation']['status'] != 'ok' for p in b['people'])),
        ('GET', '/api/v1/people/near', f"x={p0['x']}&y={p0['y']}&radius=1", None, 200,
         lambda b: any(p['id'] == pid for p in b['people'])),
        ('GET', '/api/v1/people/near', 'x=1', None, 400, None),
        ('GET', '/api/v1/people/near', 'x=nan&y=0', None, 400, None),
        ('GET', '/api/v1/people/near-fire', 'radius=6', None, 200, None),
        ('GET', f'/api/v1/people/{pid}', '', None, 200, lambda b: b['person']['id'] == pid and 'history' in b['person']),
        ('GET', '/api/v1/people/99999', '', None, 404, None),
        ('GET', '/api/v1/people/abc', '', None, 400, None),
        ('GET', f'/api/v1/people/{pid}/route', '', None, 200, lambda b: 'status' in b['route']),
        ('GET', '/api/v1/people/99999/route', '', None, 404, None),
        ('POST', f'/api/v1/people/{pid}/route', '', json.dumps({'to': [0.0, -15.0]}).encode(), 200,
         lambda b: b['route']['status'] in ('ok', 'unreachable', 'outside_map')),
        ('POST', f'/api/v1/people/{pid}/route', '', b'{"to": "here"}', 400, None),
        ('POST', f'/api/v1/people/{pid}/route', '', b'not json', 400, None),
        ('POST', '/api/v1/people', '', b'{}', 404, None),
        ('GET', '/api/v1/changes', '', None, 200, lambda b: 'grid' in b and 'people' in b),
        ('GET', '/api/v1/structures', '', None, 200, lambda b: b['count'] == len(snap['static']['structures'])),
        ('GET', f'/api/v1/structures/{sname}', '', None, 200, lambda b: b['structure']['name'] == sname),
        ('GET', '/api/v1/structures/nope', '', None, 404, None),
        ('GET', '/api/v1/hazards', '', None, 200, lambda b: b['count'] == len(snap['static']['hazards'])),
        ('GET', '/api/v1/nothing', '', None, 404, None),
        ('GET', '/api/v2/people', '', None, 404, None),
        ('DELETE', '/api/v1/people', '', None, 404, None),
    ]
    failed = 0
    for method, path, qs, body, want, check in cases:
        code, obj = api.handle(method, path, qs, body, ask, lambda: snap)
        json.dumps(obj)                                   # every answer must be valid JSON
        ok = code == want and (check is None or code != 200 or check(obj))
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {method:6s} {path + ('?' + qs if qs else ''):55s} -> {code} (expected {want})"
              + ('' if ok else f"  {str(obj)[:150]}"))
    # no twin yet / twin node silent
    code, _ = api.handle('GET', '/api/v1/twin', '', None, ask, lambda: None)
    failed += code != 503
    print(f"{'ok  ' if code == 503 else 'FAIL'} GET    /api/v1/twin without a twin                        -> {code} (expected 503)")
    code, _ = api.handle('GET', '/api/v1/summary', '', None,
                         lambda q: {'ok': False, 'error': 'the twin node did not answer'}, lambda: snap)
    failed += code != 504
    print(f"{'ok  ' if code == 504 else 'FAIL'} GET    /api/v1/summary, twin node silent                 -> {code} (expected 504)")
    print(f'\n{len(cases) + 2 - failed}/{len(cases) + 2} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
