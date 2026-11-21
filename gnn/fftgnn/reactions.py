"""Verify and load the existing MATLAB support-reaction labels.

The original FFT export already stores Rx/Ry/Rz_support_kN. These are the
forces exerted by the supports on the structure, R = f_internal-f_external
at restrained joints. Free-node residuals are NOT support reactions.
All source files are read-only. Tolerances below cover CSV round-off only;
the original case equilibrium tolerance is retained for free nodes.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def _max(value):
    return float(np.max(np.abs(value))) if np.size(value) else 0.


def _numbers(rows, columns):
    values=np.array([[float(r[c]) for c in columns] for r in rows],dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values in {columns}")
    return values


def _roundoff(a):
    # Conservative allowance for MATLAB's 15-significant-digit CSV values,
    # independent of and far below the physical equilibrium tolerance.
    return 2e-13 + 2e-14*np.abs(a)


def _cross_abs(a,b):
    a,b=np.abs(a),np.abs(b)
    return np.stack((a[...,1]*b[...,2]+a[...,2]*b[...,1],
                     a[...,2]*b[...,0]+a[...,0]*b[...,2],
                     a[...,0]*b[...,1]+a[...,1]*b[...,0]),axis=-1)


def _read_verified_case(case_dir):
    case=Path(case_dir)
    manifest=json.loads((case/'training/label_manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('status')!='independently_verified_fft_labels':
        raise ValueError('Only independently accepted FFT labels may supply reactions')
    buffers={}
    for name,folder,hash_key in (('final_joint.csv',case,'result_sha256'),
                               ('summary.json',case,'result_sha256'),
                               ('joint.csv',case/'training','training_csv_sha256')):
        raw=(folder/name).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest != manifest.get(hash_key,{}).get(name):
            raise ValueError(f'Original reaction source hash mismatch: {folder/name}')
        buffers[name]=raw
    rows=list(csv.DictReader(io.StringIO(buffers['final_joint.csv'].decode('utf-8-sig'))))
    target_rows=list(csv.DictReader(io.StringIO(buffers['joint.csv'].decode('utf-8-sig'))))
    if not rows or not target_rows:
        raise ValueError('Empty reaction/target CSV')
    def order(items):
        pairs=[(int(r['joint_id']),r) for r in items]
        if len(set(i for i,_ in pairs))!=len(pairs):
            raise ValueError('Duplicate reaction/target joint IDs')
        return sorted(pairs,key=lambda t:t[0])
    pairs,target_pairs=order(rows),order(target_rows)
    ids=np.array([i for i,_ in pairs],dtype=np.int64)
    if not np.array_equal(ids,[i for i,_ in target_pairs]):
        raise ValueError('Reaction export and training labels have different joint IDs')
    rows=[r for _,r in pairs]; target_rows=[r for _,r in target_pairs]
    summary=json.loads(buffers['summary.json'].decode('utf-8-sig'))
    if summary.get('sample_id')!=manifest.get('sample_id'):
        raise ValueError('Reaction source sample identity mismatch')
    if not summary.get('converged'):
        raise ValueError('A nonconverged FFT result cannot supply accepted reactions')
    flags=np.array([int(r['support']) for r in rows])
    target_flags=np.array([int(r['support']) for r in target_rows])
    if not np.isin(flags,[0,1]).all() or not np.array_equal(flags,target_flags):
        raise ValueError('Reaction support flags differ from the training graph')
    return rows,target_rows,ids,flags.astype(bool),summary,manifest


def verify_reaction_case(case_dir, expected_joint_ids=None, expected_free_mask=None):
    """Check saved reaction labels, including force AND moment equilibrium.

    Returns a JSON-compatible report. Global force/moment balance is assessed
    against bounds implied by the actual free-node residual tolerance, and
    its closure against the actual residual is checked to CSV precision.
    Thus small solver residuals are acknowledged rather than silently zeroed.
    """
    rows,target_rows,ids,support,summary,manifest=_read_verified_case(case_dir)
    free=~support
    if not free.any() or not support.any():
        raise ValueError('Reaction supervision requires supported and free joints')
    if expected_joint_ids is not None:
        expected=np.asarray(expected_joint_ids,dtype=np.int64)
        if not np.array_equal(ids,expected):
            raise ValueError('Reaction IDs are not aligned with the requested training graph')
    if expected_free_mask is not None and not np.array_equal(free,np.asarray(expected_free_mask,dtype=bool)):
        raise ValueError('Reaction support mask is not aligned with the requested training graph')
    xyz=_numbers(rows,('x_ff_m','y_ff_m','z_ff_m'))
    x0=_numbers(target_rows,('x0_m','y0_m','z0_m'))
    exported_u=_numbers(rows,('ux_ff_m','uy_ff_m','uz_ff_m'))
    targets=_numbers(target_rows,('ux_ff_m','uy_ff_m','uz_ff_m'))
    fint=_numbers(rows,('Fx_internal_kN','Fy_internal_kN','Fz_internal_kN'))
    fext=_numbers(rows,('Fx_final_kN','Fy_final_kN','Fz_final_kN'))
    residual_saved=_numbers(rows,('Rx_residual_kN','Ry_residual_kN','Rz_residual_kN'))
    reactions=_numbers(rows,('Rx_support_kN','Ry_support_kN','Rz_support_kN'))
    residual=fint-fext
    tolerance=float(summary['force_acceptance_threshold_kN'])
    if not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Original FFT tolerance must be finite and positive')
    rb=_roundoff(fint)+_roundoff(fext)+_roundoff(residual_saved)
    reaction_bound=_roundoff(fint)+_roundoff(fext)+_roundoff(reactions)
    fint_sum=fint.sum(0)
    fint_force_bound=_roundoff(fint).sum(0)
    fint_moment=np.cross(xyz,fint).sum(0)
    fint_moment_bound=(_cross_abs(xyz,_roundoff(fint))+
                      _cross_abs(_roundoff(xyz),fint)).sum(0)+1e-12
    balance=(fext+reactions).sum(0)
    moment=np.cross(xyz,fext+reactions).sum(0)
    free_sum=residual[free].sum(0)
    free_moment=np.cross(xyz[free],residual[free]).sum(0)
    closure_bound=(_roundoff(fext)+_roundoff(reactions)+_roundoff(fint)+_roundoff(fext)).sum(0)
    moment_closure_bound=(_cross_abs(xyz,2*_roundoff(fext)+_roundoff(reactions)+_roundoff(fint))+
                          _cross_abs(_roundoff(xyz),2*np.abs(fext)+np.abs(reactions)+np.abs(fint))).sum(0)+1e-12
    force_bound=np.full(3,free.sum()*tolerance)+closure_bound
    moment_bound=_cross_abs(xyz[free],np.full_like(xyz[free],tolerance)).sum(0)+moment_closure_bound
    checks={
        'supported_labels_zero':_max(targets[support])<=1e-10 and _max(exported_u[support])<=1e-10,
        'support_coordinates_preserved':_max(xyz[support]-x0[support])<=1e-10,
        'all_label_coordinates_match':bool(np.all(np.abs(xyz-(x0+targets))<=_roundoff(xyz)+_roundoff(x0)+_roundoff(targets))),
        'all_residuals_match_internal_minus_external':bool(np.all(np.abs(residual_saved-residual)<=rb)),
        'support_reactions_match_internal_minus_external':bool(np.all(np.abs(reactions[support]-residual[support])<=reaction_bound[support])),
        'free_reaction_labels_zero':_max(reactions[free])<=1e-12,
        'free_node_equilibrium_original_tolerance':bool(np.all(np.abs(residual[free])<=tolerance+rb[free])),
        'internal_force_resultant_zero':bool(np.all(np.abs(fint_sum)<=fint_force_bound)),
        'internal_moment_resultant_zero':bool(np.all(np.abs(fint_moment)<=fint_moment_bound)),
        'global_force_matches_free_residual':bool(np.all(np.abs(balance+free_sum)<=closure_bound)),
        'global_moment_matches_free_residual':bool(np.all(np.abs(moment+free_moment)<=moment_closure_bound)),
        'global_force_within_solver_bound':bool(np.all(np.abs(balance)<=force_bound)),
        'global_moment_within_solver_bound':bool(np.all(np.abs(moment)<=moment_bound)),
    }
    force_scale=max(float(np.linalg.norm(fext,axis=1).sum()+np.linalg.norm(reactions,axis=1).sum()),1e-12)
    moment_scale=max(float(np.linalg.norm(np.cross(xyz,fext),axis=1).sum()+np.linalg.norm(np.cross(xyz,reactions),axis=1).sum()),1e-12)
    return {'sample_id':summary['sample_id'],'passed':all(checks.values()),'checks':checks,
            'joints':len(ids),'supported_joints':int(support.sum()),'free_joints':int(free.sum()),
            'original_force_tolerance_kN':tolerance,
            'max_free_residual_kN':_max(residual[free]),
            'max_support_reaction_identity_error_kN':_max(reactions[support]-residual[support]),
            'external_force_resultant_kN':fext.sum(0).tolist(),
            'support_reaction_resultant_kN':reactions.sum(0).tolist(),
            'global_force_imbalance_kN':balance.tolist(),
            'global_moment_imbalance_kNm':moment.tolist(),
            'global_force_bound_kN':force_bound.tolist(),
            'global_moment_bound_kNm':moment_bound.tolist(),
            'relative_global_force_imbalance':float(np.linalg.norm(balance)/force_scale),
            'relative_global_moment_imbalance':float(np.linalg.norm(moment)/moment_scale),
            'max_global_force_imbalance_kN':_max(balance),
            'max_global_moment_imbalance_kNm':_max(moment),
            'source_final_joint_sha256':manifest['result_sha256']['final_joint.csv'],
            'reaction_convention':'support force on structure; R_s=f_internal_s-f_external_s; R_free=0',
            'global_balance_note':'Nonzero total imbalance equals minus the remaining free-node residual; assessed against unchanged FFT tolerances.'}


def read_verified_reactions(case_dir,joint_ids,free_mask):
    """Return N×3 float64 saved reaction labels, including zero free entries."""
    import torch
    ids=joint_ids.detach().cpu().numpy() if torch.is_tensor(joint_ids) else joint_ids
    free=free_mask.detach().cpu().numpy() if torch.is_tensor(free_mask) else free_mask
    report=verify_reaction_case(case_dir,ids,free)
    if not report['passed']:
        raise ValueError(f"Reaction balance verification failed for {report['sample_id']}: "+
                         ', '.join(k for k,v in report['checks'].items() if not v))
    rows,_,_,_,_,_=_read_verified_case(case_dir)
    return torch.tensor(_numbers(rows,('Rx_support_kN','Ry_support_kN','Rz_support_kN')),dtype=torch.float64)
