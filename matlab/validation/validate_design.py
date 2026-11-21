"""Independently verify one exported C24 benchmark design using NumPy.

Usage: python validation/validate_design.py outputs/sizing/<run-folder>
"""
from pathlib import Path
import argparse
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
from independent_frame_core import independent_frame, panel_loads
from verify_exported_design import audit


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    model = read(folder/'frame_model.json')
    cases = read(folder/'frame_cases.json')
    definitions = read(folder/'case_reports.json')
    loads = read(folder/'load_components.json')
    records = []

    def close(name, actual, expected, tolerance):
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)), initial=0))
        records.append(dict(check=name, passes=error<=tolerance, maximum_error=error, tolerance=tolerance))
        assert error<=tolerance, f'{name}: {error} exceeds {tolerance}'

    displacement, force, reaction, actions, residual = independent_frame(model, cases)
    roof = np.asarray(model['originalNodes'])
    panels = loads['panels']
    qd = loads['qD_kPa']; qw = loads['qW_kPa']
    if all(len(p)==4 for p in panels):
        gravity, wind = panel_loads(roof, panels, qd, qw, order=20)
    else:
        gravity = np.zeros_like(roof); wind = np.zeros_like(roof)
        for corners, dead, pressure in zip(panels, qd, qw):
            ids = np.asarray(corners)-1
            assert len(ids)==3, 'This independent checker supports the packaged T3/Q4 benchmark panels.'
            triangle = roof[ids]
            area = np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])/2
            gravity[ids,2] -= dead*np.linalg.norm(area)/3
            wind[ids,2] -= pressure*abs(area[2])/3
    close('panel gravity integration', gravity, loads['unfactoredPanelDeadNodal'], 1e-8)
    close('signed wind integration', wind, loads['unfactoredPanelWindReferenceNodal'], 1e-8)
    for c, (case, definition) in enumerate(zip(cases, definitions)):
        name = case['name']
        close(name+' displacements', displacement[:,c].reshape(-1,6), case['displacements'], 1e-8)
        close(name+' reactions', reaction[:,c].reshape(-1,6), case['reactions'], 1e-5)
        close(name+' end-I actions', actions[:,0,:,c], case['sectionForcesI'], 1e-5)
        close(name+' end-J actions', actions[:,1,:,c], case['sectionForcesJ'], 1e-5)
        close(name+' consistent beam loads', force[:,c].reshape(-1,6), case['appliedNodalForces'], 1e-9)
        close(name+' free-node equilibrium', residual[:,c], 0, case['residualTolerance'])
        prescribed = definition['gammaG']*gravity + definition['gammaW']*definition['windSign']*wind
        close(name+' panel load combination', np.asarray(case['prescribedNodalForces'])[:len(roof),:3], prescribed, 1e-8)
        assert np.all(np.asarray(case['lineLoadsGlobal'])[:,2]<0), 'Member weight must remain downward.'
    member_report = audit(folder)
    assert member_report['passed_independent_member_audit']
    result = dict(passes=True, elastic_and_load_checks=len(records), checks=records,
                  member_checks=member_report,
                  scope='Independent verification of the exported first-order frame response, load combinations and specified C24 member checks; no additional design checks are implied.')
    (folder/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(records)} elastic/load comparisons and the independent N-mm-MPa member audit.')


if __name__=='__main__':
    main()
