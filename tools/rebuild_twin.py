#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(__file__.rsplit('/tools/', 1)[0], 'src/semantic_twin'))
from semantic_twin import twin_core as tc

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('survey_grid')
    ap.add_argument('people')
    ap.add_argument('old_state')
    ap.add_argument('out_state')
    ap.add_argument('--from-flight', type=int, default=1)
    a = ap.parse_args()

    survey = tc.load_grid(a.survey_grid)
    old = tc.Twin.load(a.old_state)
    raw = survey.copy()
    z = np.where(old.z_flight >= a.from_flight, old.z, np.nan)
    raw['z'] = z
    raw['n'] = None            # the inspection run's per-cell point counts were not saved
    people = json.load(open(a.people))

    t = tc.Twin(hazards=old.hazards)
    t.begin_flight('flight_1')
    t.update_grid(survey)
    t.begin_flight('flight_2')
    t.update_people(people)
    t.update_grid(raw)
    t.save(a.out_state)
    s = t.snapshot()
    print('rebuilt:', s['summary'])
    print('grid changes flight_2 vs survey:', {k: v for k, v in (s['grid_changes'] or {}).items() if k != 'regions'})


if __name__ == '__main__':
    main()
