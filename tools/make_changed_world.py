#!/usr/bin/env python3

import re

HERE = __file__.rsplit('/tools/', 1)[0]
SRC = HERE + '/src/semantic_twin/worlds/disaster_scene.world'
DST = HERE + '/src/semantic_twin/worlds/disaster_scene_changed.world'

MOVES = {'person_standing_3': (3.25, 5.35478), 'person_standing_2': (-2.2, 0.3)}
REMOVE = ['person_walking']
NEW_PERSON = ('person_new', -0.5, 5.8)
NEW_OBJECT = ('dumpster_new', 'dumpster', 0.6, 1.0)


def model_block(s, name):
    m = re.search(r'    <model name="%s">.*?\n    </model>\n' % re.escape(name), s, re.S)
    if not m:
        raise SystemExit(f'model {name} not found')
    return m


def set_xy(block, x, y):
    out, n = re.subn(r'(\n      <pose>)([-\d.]+) ([-\d.]+)', lambda g: f'{g.group(1)}{x} {y}', block, count=1)
    if n != 1:
        raise SystemExit('model-level pose not found')
    return out


def main():
    s = open(SRC).read()
    for name, (x, y) in MOVES.items():
        m = model_block(s, name)
        blk = set_xy(m.group(0), x, y)
        s = s.replace(m.group(0), blk)
    for name in REMOVE:
        s = s.replace(model_block(s, name).group(0), '')
    base = model_block(s, 'person_standing_2')
    name, x, y = NEW_PERSON
    new = re.sub(r'<model name="person_standing_2">', f'<model name="{name}">', base.group(0), count=1)
    new = set_xy(new, x, y)
    s = s.replace(base.group(0), base.group(0) + new)
    name, mesh, x, y = NEW_OBJECT
    inc = (f'    <include>\n      <name>{name}</name>\n      <uri>model://{mesh}</uri>\n      <static>true</static>\n'
           f'      <pose>{x} {y} 0 0 0 0.3</pose>\n    </include>\n')
    s = s.replace('  </world>', inc + '  </world>')
    open(DST, 'w').write(s)
    print('wrote', DST)


if __name__ == '__main__':
    main()
