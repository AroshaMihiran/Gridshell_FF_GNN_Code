"""Independent audit of loads, elastic response and sequential-design invariants.

Frame stiffness is assembled from end-rotation flexibilities, independently of
the MATLAB closed-form matrix. Uniform body-load vectors are integrated with
Hermite/linear functions and Gaussian quadrature rather than copied formulas.
This audit is not a global stability or full timber design certification.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
from independent_frame_reference import bending_by_flexibility

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def norm(a): return float(np.max(np.abs(a),initial=0))
def vec(value,n):
    a=np.asarray(value,float).reshape(-1)
    return np.full(n,a[0]) if len(a)==1 else a

records=[]
def check(name,passed,**details):
    item={'check':name,'passes':bool(passed),**details};records.append(item)
    print(('PASS ' if passed else 'FAIL ')+name,flush=True)
    return item
def close(name,a,b,tolerance):
    error=norm(np.asarray(a)-np.asarray(b));return check(name,error<=tolerance,max_abs_error=error,tolerance=tolerance)

def panel_loads(nodes,panels,qd,qw,order=20):
    ab,w=np.polynomial.legendre.leggauss(order);dead=np.zeros_like(nodes);wind=np.zeros_like(nodes)
    for corners,d,p in zip(panels,qd,qw):
        ids=np.asarray(corners,int)-1;v=nodes[ids];assert len(ids)==4
        for i,xi in enumerate(ab):
            for j,eta in enumerate(ab):
                N=.25*np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])
                d1=.25*np.array([-(1-eta),1-eta,1+eta,-(1+eta)])
                d2=.25*np.array([-(1-xi),-(1+xi),1+xi,1-xi])
                a=np.cross(d1@v,d2@v);weight=w[i]*w[j]
                dead[ids,2]-=d*N*np.linalg.norm(a)*weight
                wind[ids,2]-=p*N*abs(a[2])*weight
    return dead,wind

def independent_frame(model,cases):
    xyz=np.asarray(model['nodes'],float);bars=np.asarray(model['bars'],int)-1
    nn,ne,nc=len(xyz),len(bars),len(cases);nd=6*nn
    props={key:vec(model[key],ne) for key in ('E','G','A','I2','I3','J','As2','As3')}
    K=np.zeros((nd,nd));P=np.column_stack([np.asarray(c['prescribedNodalForces']).reshape(-1) for c in cases])
    elements=[];g,gw=np.polynomial.legendre.leggauss(3)
    for e,(i,j) in enumerate(bars):
        chord=xyz[j]-xyz[i];L=np.linalg.norm(chord);ex=chord/L;ref=np.array([0.,0.,1.])
        if np.linalg.norm(np.cross(ex,ref))<1e-8:ref=np.array([1.,0.,0.])
        ey=ref-np.dot(ref,ex)*ex;ey/=np.linalg.norm(ey);R=np.array([ex,ey,np.cross(ex,ey)]);T=np.kron(np.eye(4),R)
        k=np.zeros((12,12));pair=np.array([[1.,-1.],[-1.,1.]])
        k[np.ix_([0,6],[0,6])]=props['E'][e]*props['A'][e]/L*pair
        k[np.ix_([3,9],[3,9])]=props['G'][e]*props['J'][e]/L*pair
        k[np.ix_([1,5,7,11],[1,5,7,11])]=bending_by_flexibility(props['E'][e]*props['I3'][e],props['G'][e]*props['As2'][e],L)
        S=np.diag([1.,-1.,1.,-1.])
        k[np.ix_([2,4,8,10],[2,4,8,10])]=S@bending_by_flexibility(props['E'][e]*props['I2'][e],props['G'][e]*props['As3'][e],L)@S
        dof=np.r_[np.arange(6*i,6*i+6),np.arange(6*j,6*j+6)];K[np.ix_(dof,dof)]+=T.T@k@T
        equivalent=np.zeros((12,nc))
        for c,case in enumerate(cases):
            q=R@np.asarray(case['lineLoadsGlobal'])[e]
            for value,weight in zip(g,gw):
                t=(value+1)/2;dx=weight*L/2
                H=np.array([1-3*t*t+2*t**3,L*(t-2*t*t+t**3),3*t*t-2*t**3,L*(-t*t+t**3)])
                equivalent[[0,6],c]+=q[0]*np.array([1-t,t])*dx
                equivalent[[1,5,7,11],c]+=q[1]*H*dx
                equivalent[[2,4,8,10],c]+=q[2]*(S@H)*dx
            P[dof,c]+=T.T@equivalent[:,c]
        elements.append((k,T,dof,L,equivalent))
    free=~np.asarray(model['restraints'],bool).reshape(-1);U=np.zeros((nd,nc));Kff=K[np.ix_(free,free)]
    scale=1/np.sqrt(np.diag(Kff));scaled=scale[:,None]*Kff*scale[None,:]
    U[free]=scale[:,None]*np.linalg.solve(scaled,scale[:,None]*P[free])
    residual=K@U-P;reaction=np.zeros_like(U);reaction[~free]=residual[~free]
    actions=np.zeros((ne,2,6,nc))
    for e,(k,T,dof,L,f) in enumerate(elements):
        ends=k@T@U[dof]-f;actions[e,0]=-ends[:6];actions[e,1]=ends[6:]
    return U,P,reaction,actions,residual[free]

