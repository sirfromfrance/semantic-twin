#!/usr/bin/env python3

import argparse
import json

import importlib.util

spec = importlib.util.spec_from_file_location(
    'operator_page', __file__.rsplit('/tools/', 1)[0] + '/src/semantic_twin/semantic_twin/operator_page.py')
_m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_m)
PAGE = _m.PAGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('twin')
    ap.add_argument('-o', '--out', default='/tmp/sdt_twin.html')
    a = ap.parse_args()
    twin = json.load(open(a.twin))
    open(a.out, 'w').write(PAGE.replace('__TWIN__', json.dumps(twin)))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
