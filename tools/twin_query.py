#!/usr/bin/env python3

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.join(__file__.rsplit('/tools/', 1)[0], 'src/semantic_twin/semantic_twin'))
tc = importlib.import_module('twin_core')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-s', '--state', default=os.path.expanduser('~/ROS/thesis_ws/results/twin_state.json'))
    ap.add_argument('op')
    ap.add_argument('args', nargs='*')
    a = ap.parse_args()
    req = {'op': a.op}
    for kv in a.args:
        k, v = kv.split('=', 1)
        try:
            req[k] = float(v) if '.' in v or v.lstrip('-').isdigit() else v
        except ValueError:
            req[k] = v
    print(json.dumps(tc.Twin.load(a.state).query(req), indent=1))


if __name__ == '__main__':
    main()
