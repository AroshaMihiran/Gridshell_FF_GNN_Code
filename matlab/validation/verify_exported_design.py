"""Independent read-only audit of exported member demands and EC5 ratios.

Does not call MATLAB or copy its checker. Reconstructs section-resultant
polynomials from equilibrium and evaluates the first-generation EC5 formulas
in an independent N-mm-MPa calculation. The audit returns a report without changing the supplied files.
"""
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np



def utilization(N_kN, M2_kNm, M3_kNm, side_m, length_m, kmod):
    """Independent units: N, mm, MPa; square section and stated K=1."""
    side = np.asarray(side_m)*1000
    area = side**2
    modulus = side**3/6
    radius = side/math.sqrt(12)
    slender = np.asarray(length_m)*1000/(math.pi*radius)*math.sqrt(21/7400)
    phi = (1+0.2*(slender-0.3)+slender**2)/2
    kc = np.ones_like(slender, dtype=float)
    active = slender > 0.3
    kc[active] = 1/(phi[active]+np.sqrt(phi[active]**2-slender[active]**2))
    nc = np.maximum(-np.asarray(N_kN)*1000, 0)/(area*(21*kmod/1.3))
    nt = np.maximum(np.asarray(N_kN)*1000, 0)/(area*(14.5*kmod/1.3))
    b2 = np.abs(np.asarray(M2_kNm))*1e6/(modulus*(24*kmod/1.3))
    b3 = np.abs(np.asarray(M3_kNm))*1e6/(modulus*(24*kmod/1.3))
    cross = nc**2+nt+np.maximum(b2+.7*b3, .7*b2+b3)
    column = np.where((np.asarray(N_kN)<0)&active,
                      nc/kc+np.maximum(b2+.7*b3, .7*b2+b3), 0)
    return np.maximum.reduce([nc, nt, cross, column])


def audit(folder):
    folder = Path(folder)
    name = folder.name
    model = json.loads((folder/'frame_model.json').read_text())
    cases = json.loads((folder/'frame_cases.json').read_text())
    definitions = json.loads((folder/'case_reports.json').read_text())
    member_table = list(csv.DictReader((folder/'design_members.csv').open()))
    nodes = np.array(model['nodes'])
    bars = np.array(model['bars'], dtype=int)-1
    member = np.array(model['frameMemberRows'], dtype=int)-1
    side = np.array(model['sides'])
    length = np.array(model['originalMemberLengths'])
    seg_length = np.linalg.norm(nodes[bars[:,1]]-nodes[bars[:,0]], axis=1)
    nm = len(side)
    ne = len(bars)
    assert nm in (112,280)
    assert set(member) == set(range(nm))
    assert not model['initialStressTransferred']
    assert np.allclose(model['A'], side[member]**2, rtol=1e-13, atol=0)
    assert np.allclose(model['E'], 11e6, rtol=0, atol=0)
    assert np.allclose(model['G'], .69e6, rtol=0, atol=0)
    assert model['material']['fc0k']==21000
    assert model['material']['ft0k']==14500
    assert model['material']['E05']==7.4e6
    assert model['designOptions']==dict(betaC=.2,km=.7,EffectiveLengthFactor2=1,EffectiveLengthFactor3=1)
    assert np.allclose(np.bincount(member, weights=seg_length), length, rtol=1e-13)
    group = np.array(model['groups'])
    group_mismatch = max(float(np.ptp(side[group==g])) for g in set(group))
    assert group_mismatch < 1e-12
    out = []
    member_case = np.zeros((nm,len(cases)))
    G = np.asarray(cases[0]['prescribedNodalForces'])/1.35
    W = (np.asarray(cases[1]['prescribedNodalForces'])-1.35*G)/1.5
    expected_factors = [(1.35,0,0,.6),(1.35,1.5,1,.9),(1.,1.5,1,.9),
                        (1.35,1.5,-1,.9),(1.,1.5,-1,.9)]
    assert len(cases)==len(expected_factors)
    for c, (case, definition, factors) in enumerate(zip(cases, definitions, expected_factors)):
        gammaG, gammaW, sign, kmod = factors
        assert all(abs(definition[k]-v)<1e-14 for k,v in
                   zip(['gammaG','gammaW','windSign','kmod'],factors))
        nodal = np.asarray(case['prescribedNodalForces'])
        assert np.allclose(nodal, gammaG*G+gammaW*sign*W,rtol=1e-13,atol=1e-10)
        body = np.asarray(case['lineLoadsGlobal'])
        assert np.max(abs(body[:,:2]))==0
        assert np.allclose(body[:,2],-gammaG*4.1202*side[member]**2,rtol=1e-13,atol=0)
        local_q = np.asarray(case['lineLoadsLocal'])
        start = np.asarray(case['sectionForcesI'])
        finish = np.asarray(case['sectionForcesJ'])
        actual_envelope = np.asarray(case['envelopeActions'])
        reconstructed = np.zeros((2*ne,6))
        dense_max = np.zeros(ne)
        endpoint_error = 0.
        for e in range(ne):
            L = seg_length[e]
            a = start[e]
            q = local_q[e]
            # F(x)=F(0)-qx; M'(x)=-e1 cross F(x).
            def section(x):
                x = np.atleast_1d(x)
                force = a[:3]-x[:,None]*q
                torque = np.full_like(x,a[3])
                m2 = a[4]+a[2]*x-q[2]*x*x/2
                m3 = a[5]-a[1]*x+q[1]*x*x/2
                return np.column_stack([force,torque,m2,m3])
            endpoint_error=max(endpoint_error,float(np.max(abs(section(L)[0]-finish[e]))))
            candidates = [0.,L]
            for force_value, load_value in [(a[2],q[2]),(a[1],q[1])]:
                if load_value!=0 and 0<force_value/load_value<L:
                    candidates.append(force_value/load_value)
            values = section(candidates)
            extremes = np.max(abs(values[:,1:]),axis=0)
            reconstructed[e] = np.r_[np.min(values[:,0]),extremes]
            reconstructed[ne+e] = np.r_[np.max(values[:,0]),extremes]
            dense = section(np.linspace(0,L,101))
            dense_max[e]=np.max(utilization(dense[:,0],dense[:,4],dense[:,5],
                        np.full(101,side[member[e]]),np.full(101,length[member[e]]),kmod))
        envelope_error=float(np.max(abs(reconstructed-actual_envelope)))
        assert envelope_error<1e-7
        assert endpoint_error<1e-7
        station_parent=np.r_[member,member]
        station_ratio=utilization(reconstructed[:,0],reconstructed[:,4],reconstructed[:,5],
                                  side[station_parent],length[station_parent],kmod)
        for e in range(nm): member_case[e,c]=max(station_ratio[station_parent==e])
        seg_bound=np.maximum(station_ratio[:ne],station_ratio[ne:])
        assert np.all(dense_max<=seg_bound+1e-12)
        # External equilibrium using analytic body resultants at centroids,
        # independently of the exported equivalent nodal vectors.
        reaction=np.asarray(case['reactions'])
        body_force=body*seg_length[:,None]
        centroid=(nodes[bars[:,0]]+nodes[bars[:,1]])/2
        force_close=np.sum(nodal[:,:3]+reaction[:,:3],axis=0)+np.sum(body_force,axis=0)
        moment_close=(np.sum(nodal[:,3:]+reaction[:,3:],axis=0)
                      +np.sum(np.cross(nodes,nodal[:,:3]+reaction[:,:3]),axis=0)
                      +np.sum(np.cross(centroid,body_force),axis=0))
        assert np.linalg.norm(force_close,np.inf)<1e-6
        assert np.linalg.norm(moment_close,np.inf)<1e-5
        reported=definition['maximumUtilization']
        assert abs(np.max(station_ratio)-reported)<1e-10
        out.append(dict(case=case['name'],max_ratio=float(np.max(station_ratio)),
                        max_dense_concurrent_ratio=float(np.max(dense_max)),
                        nonconcurrent_envelope_bounds_all_sampled_sections=True,
                        endpoint_recovery_error_kN_or_kNm=endpoint_error,
                        envelope_recovery_error_kN_or_kNm=envelope_error,
                        global_force_closure_kN=force_close.tolist(),
                        global_moment_closure_kNm=moment_close.tolist(),
                        selfweight_kN=float(np.sum(-body_force[:,2])),
                        reported_ratio_difference=float(np.max(station_ratio)-reported)))
    expected_ratios=np.max(member_case,axis=1)
    reported_ratios=np.array([float(x['Utilization']) for x in member_table])
    ratio_error=float(np.max(abs(expected_ratios-reported_ratios)))
    assert ratio_error<1e-10
    volume=float(np.sum(side**2*length))
    sls=json.loads((folder/'serviceability_displacements.json').read_text())
    assert not sls['complianceAssessed'] and not sls['displacementLimitSpecified']
    assert sls['kdef']==.8 and sls['psi2Wind']==0
    uG=np.asarray(cases[0]['displacements'])/1.35
    uW=(np.asarray(cases[1]['displacements'])-1.35*uG)/1.5
    sls_error=0.
    for field,expected in [('instantaneousReference',uG+uW),('instantaneousReversed',uG-uW),
                           ('finalReference',1.8*uG+uW),('finalReversed',1.8*uG-uW)]:
        exported=np.asarray(sls[field])
        expected=expected[:exported.shape[0],:exported.shape[1]]
        error=float(np.max(abs(exported-expected)))
        sls_error=max(sls_error,error)
        assert error<1e-10
    # Conditional screening, not a passed LTB check: L_eff=L and rectangular
    # softwood formula, no continuous edge-restraint assumption.
    lambda_ltb=np.sqrt(24*(length*1000)/(.78*(side*1000)*7400))
    return dict(scenario=name,passed_independent_member_audit=True,
        physical_members=nm,numerical_segments=ne,retained_all_members=True,
        initial_stress_transferred=False,max_member_ratio=float(np.max(expected_ratios)),
        max_ratio_difference=ratio_error,volume_m3=volume,
        unfactored_member_selfweight_kN=volume*4.1202,
        SLS_signed_superposition_max_error=sls_error,
        group_section_max_difference_m=group_mismatch,cases=out,
        conditional_LTB_screen={"assumption":"L_eff=L, square softwood, not a verification of restraint or combined LTB",
            "status":"UNVERIFIED: conditional slenderness screen only; neither an LTB pass nor a resistance failure",
            "max_lambda_rel_m":float(np.max(lambda_ltb)),
            "number_above_0_75":int(np.sum(lambda_ltb>.75)),
            "full_LTB_check_passed":False},
        scope="Independent force recovery, combination/selfweight checks and EC5 normal-stress/assumed-K=1 column ratios. Does not establish roof/global/LTB/shear/torsion/connection/SLS adequacy.")


