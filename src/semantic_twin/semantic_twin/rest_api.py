"""REST API, version 1, for the semantic twin (ROS-free, so it can be tested on a saved twin).

Every endpoint maps onto the twin's existing query operations (Twin.query) or onto the latest
twin snapshot; nothing here changes the twin. All responses are JSON.

  GET  /api/v1                          this list of endpoints
  GET  /api/v1/twin                     the full latest snapshot
  GET  /api/v1/summary                  counts and the map-change report of the current flight
  GET  /api/v1/flights                  the flights in the twin
  GET  /api/v1/people                   people; filters: status, posture, role, min_level (low|medium|high)
  GET  /api/v1/people/priority?n=5      the n people with the highest triage priority
  GET  /api/v1/people/unreachable       people without a passable evacuation route
  GET  /api/v1/people/near?x=&y=&radius=5        people within radius (m) of a point
  GET  /api/v1/people/near-fire?radius=          people near a fire
  GET  /api/v1/people/{id}              the full record of one person (history, relations, triage, route)
  GET  /api/v1/people/{id}/route        that person's evacuation route
  POST /api/v1/people/{id}/route        body {"to": [x, y]}: a route to a chosen point (computed, not stored)
  GET  /api/v1/changes                  people changes and map changes of the current flight
  GET  /api/v1/structures               sensed structures
  GET  /api/v1/structures/{name}        one structure and the people in or next to it
  GET  /api/v1/hazards                  fire and smoke (scene description)

Status codes: 200 ok, 400 bad parameter, 404 unknown endpoint/person/structure,
503 no twin received yet, 504 the twin node did not answer.
"""
import json
import math
from urllib.parse import parse_qs

PREFIX = '/api/v1'
LEVELS = ('low', 'medium', 'high')
PEOPLE_FILTERS = ('status', 'posture', 'role', 'min_level')


class ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def _num(params, name, default=None):
    if name not in params:
        if default is None:
            raise ApiError(400, f'missing parameter {name}')
        return default
    try:
        v = float(params[name])
    except ValueError:
        raise ApiError(400, f'parameter {name} must be a number')
    if not math.isfinite(v):
        raise ApiError(400, f'parameter {name} must be finite')
    return v


def _int_id(text, what):
    try:
        return int(text)
    except ValueError:
        raise ApiError(400, f'{what} must be an integer')


def _answer(res):
    """A Twin.query result -> (status code, body)."""
    if res.get('ok'):
        return 200, res
    err = str(res.get('error', 'error'))
    if err.startswith(('unknown person', 'unknown structure')):
        return 404, res
    if 'did not answer' in err:
        return 504, res
    return 400, res


def handle(method, path, query_string, body, ask, snapshot):
    """One HTTP request -> (status code, JSON-serialisable body).

    ask(req: dict) -> dict      sends a query to the twin (Twin.query semantics)
    snapshot() -> dict | None   the latest twin snapshot
    """
    try:
        return _handle(method, path.rstrip('/') or '/', query_string or '', body, ask, snapshot)
    except ApiError as e:
        return e.code, {'ok': False, 'error': e.message}


def _handle(method, path, qs, body, ask, snapshot):
    if not (path == PREFIX or path.startswith(PREFIX + '/')):
        raise ApiError(404, 'unknown endpoint')
    parts = [p for p in path[len(PREFIX):].split('/') if p]
    params = {k: v[-1] for k, v in parse_qs(qs).items()}

    if method == 'POST':
        if len(parts) == 3 and parts[0] == 'people' and parts[2] == 'route':
            pid = _int_id(parts[1], 'person id')
            try:
                to = json.loads(body or b'{}').get('to')
                to = [float(to[0]), float(to[1])]
                if not all(math.isfinite(v) for v in to):
                    raise ValueError
            except (ValueError, TypeError, IndexError, KeyError, AttributeError):
                raise ApiError(400, 'body must be {"to": [x, y]}')
            return _answer(ask({'op': 'route', 'person': pid, 'to': to}))
        raise ApiError(404, 'unknown endpoint')
    if method != 'GET':
        raise ApiError(404, 'unknown endpoint')

    if not parts:
        return 200, {'ok': True, 'version': 1, 'endpoints': [l.strip() for l in __doc__.splitlines()
                                                             if l.strip().startswith(('GET ', 'POST '))]}

    def snap():
        s = snapshot()
        if s is None:
            raise ApiError(503, 'no twin received yet')
        return s

    head = parts[0]
    if head == 'twin' and len(parts) == 1:
        return 200, snap()
    if head == 'summary' and len(parts) == 1:
        return _answer(ask({'op': 'summary'}))
    if head == 'flights' and len(parts) == 1:
        m = snap()['meta']
        return 200, {'ok': True, 'current_flight': m.get('current_flight'), 'flights': m.get('flights', [])}
    if head == 'changes' and len(parts) == 1:
        return _answer(ask({'op': 'changes'}))
    if head == 'hazards' and len(parts) == 1:
        hz = snap()['static'].get('hazards', [])
        return 200, {'ok': True, 'count': len(hz), 'hazards': hz}

    if head == 'structures':
        if len(parts) == 1:
            st = snap()['static'].get('structures', [])
            return 200, {'ok': True, 'count': len(st), 'structures': st}
        if len(parts) == 2:
            return _answer(ask({'op': 'structure', 'name': parts[1]}))

    if head == 'people':
        if len(parts) == 1:
            unknown = set(params) - set(PEOPLE_FILTERS)
            if unknown:
                raise ApiError(400, f'unknown filter(s): {", ".join(sorted(unknown))}')
            if 'min_level' in params and params['min_level'] not in LEVELS:
                raise ApiError(400, 'min_level must be low, medium or high')
            return _answer(ask({'op': 'people', **params}))
        sub = parts[1]
        if len(parts) == 2 and sub == 'priority':
            n = _num(params, 'n', 5)
            if n < 1 or n != int(n):
                raise ApiError(400, 'n must be a positive integer')
            return _answer(ask({'op': 'priority', 'n': int(n)}))
        if len(parts) == 2 and sub == 'unreachable':
            return _answer(ask({'op': 'unreachable'}))
        if len(parts) == 2 and sub == 'near':
            r = _num(params, 'radius', 5.0)
            if r <= 0:
                raise ApiError(400, 'radius must be positive')
            return _answer(ask({'op': 'near', 'x': _num(params, 'x'), 'y': _num(params, 'y'), 'radius': r}))
        if len(parts) == 2 and sub == 'near-fire':
            req = {'op': 'near_hazard'}
            if 'radius' in params:
                req['radius'] = _num(params, 'radius')
            return _answer(ask(req))
        pid = _int_id(sub, 'person id')
        if len(parts) == 2:
            p = next((q for q in snap()['people'] if q['id'] == pid), None)
            if p is None:
                raise ApiError(404, f'unknown person {pid}')
            return 200, {'ok': True, 'person': p}
        if len(parts) == 3 and parts[2] == 'route':
            return _answer(ask({'op': 'route', 'person': pid}))

    raise ApiError(404, 'unknown endpoint')
