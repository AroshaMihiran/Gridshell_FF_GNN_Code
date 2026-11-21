function result = frame3d_linear(model,options)
%FRAME3D_LINEAR Small-displacement 3D prismatic Timoshenko frame analysis.
% Six DOFs per node: UX UY UZ RX RY RZ, global right-hand rotations.
% Required fields: nodes(N,3), bars(M,2), E,G,A,I2,I3,J,As2,As3 (scalar
% or M-vectors), restraints(N,6), loads(N,6). Units are kN and metres.
% Optional local2(M,3): vector whose projection normal to each member defines
% the local 2 axis. Default: projected global Z, with global X for verticals.
% Zero imposed support displacement, rigid member connections, no offsets,
% joint springs, distributed member loads, geometric stiffness, prestress or
% nonlinear effects. Loads must already be assembled as nodal loads.
% E,G are independently supplied: timber need not be isotropic.
% As2/As3=Inf explicitly suppresses shear deformation; zero is NOT accepted.
% endForces are local element nodal actions k_local*u_local.
% sectionForcesI=-endForces(:,1:6), sectionForcesJ=endForces(:,7:12).
% Section-force order N,V2,V3,T,M2,M3; positive N is tension.
% Member equations follow the standard two-node shear-flexible frame matrix.
% Shear conventions and axis definitions: CSI Analysis Reference Manual,
% chapter VII, https://docs.csiamerica.com/manuals/etabs/Analysis%20Reference.pdf
if nargin<2,options=struct();end
if ~isstruct(model)||~isscalar(model)||~isstruct(options)||~isscalar(options)
    error('frame3d:Input','model and options must be scalar structures.');
end
defaults=struct('RelativeResidualTolerance',1e-8,'AbsoluteResidualTolerance',1e-8, ...
    'MinimumReciprocalCondition',1e-14);
names=fieldnames(options);
for k=1:numel(names)
    if ~isfield(defaults,names{k}),error('frame3d:Option','Unknown option %s.',names{k});end
    defaults.(names{k})=options.(names{k});
end
options=defaults;
for k=1:numel(fieldnames(options))
    names=fieldnames(options);validateattributes(options.(names{k}),{'numeric'}, ...
        {'real','finite','scalar','positive'},mfilename,names{k});
end
required={'nodes','bars','E','G','A','I2','I3','J','As2','As3','restraints','loads'};
for k=1:numel(required)
    if ~isfield(model,required{k}),error('frame3d:Input','Missing model.%s.',required{k});end
end
validateattributes(model.nodes,{'numeric'},{'2d','ncols',3,'real','finite','nonempty'});
nodes=double(model.nodes);nn=size(nodes,1);nd=6*nn;
validateattributes(model.bars,{'numeric'},{'2d','ncols',2,'integer','positive','nonempty'});
bars=double(model.bars);nm=size(bars,1);
if any(bars(:)>nn)||any(bars(:,1)==bars(:,2))
    error('frame3d:Connectivity','Invalid member end node.');
end
if ~isequal(size(model.restraints),[nn,6]) || ...
        ~(islogical(model.restraints)||isnumeric(model.restraints)) || ...
        any(~ismember(model.restraints(:),[0,1]))
    error('frame3d:Restraints','restraints must be an N by 6 logical/zero-one matrix.');
end
validateattributes(model.loads,{'numeric'},{'size',[nn,6],'real','finite'});
restraints=logical(model.restraints);F=reshape(double(model.loads)',[],1);
fixed=find(reshape(restraints',[],1));free=find(~reshape(restraints',[],1));
fields={'E','G','A','I2','I3','J','As2','As3'};p=struct();
for k=1:numel(fields)
    name=fields{k};p.(name)=memberVector(model.(name),nm,name,ismember(name,{'As2','As3'}));
end
if isfield(model,'local2')
    validateattributes(model.local2,{'numeric'},{'size',[nm,3],'real','finite'});
end
row=zeros(144*nm,1);col=row;val=row;R=zeros(3,3,nm);
T=zeros(12,12,nm);ke=zeros(12,12,nm);dof=zeros(nm,12);L=zeros(nm,1);
geometryScale=max(1,max(max(nodes)-min(nodes)));
for m=1:nm
    delta=nodes(bars(m,2),:)-nodes(bars(m,1),:);L(m)=norm(delta);
    if L(m)<=100*eps(geometryScale)
        error('frame3d:ZeroLength','Member %d has zero or numerically negligible length.',m);
    end
    e1=delta/L(m);
    if isfield(model,'local2')
        ref=model.local2(m,:);
    else
        ref=[0,0,1];
        if norm(cross(e1,ref))<1e-8,ref=[1,0,0];end
    end
    e2=ref-dot(ref,e1)*e1;
    if norm(e2)<=1e-12*max(1,norm(ref))
        error('frame3d:LocalAxes','Member %d local2 is parallel to local1 or zero.',m);
    end
    e2=e2/norm(e2);e3=cross(e1,e2);e3=e3/norm(e3);
    R(:,:,m)=[e1;e2;e3];T(:,:,m)=kron(eye(4),R(:,:,m));
    k=zeros(12);
    k([1,7],[1,7])=p.E(m)*p.A(m)/L(m)*[1,-1;-1,1];
    k([4,10],[4,10])=p.G(m)*p.J(m)/L(m)*[1,-1;-1,1];
    k([2,6,8,12],[2,6,8,12])=bending(p.E(m)*p.I3(m),p.G(m)*p.As2(m),L(m));
    signW=diag([1,-1,1,-1]);
    k([3,5,9,11],[3,5,9,11])=signW*bending(p.E(m)*p.I2(m),p.G(m)*p.As3(m),L(m))*signW;
    ke(:,:,m)=k;
    dof(m,:)=[6*(bars(m,1)-1)+(1:6),6*(bars(m,2)-1)+(1:6)];
    kg=T(:,:,m)'*k*T(:,:,m);
    [rr,cc]=ndgrid(dof(m,:),dof(m,:));idx=(m-1)*144+(1:144);
    row(idx)=rr(:);col(idx)=cc(:);val(idx)=kg(:);
end
K=sparse(row,col,val,nd,nd);K=(K+K')/2;
u=zeros(nd,1);reciprocalCondition=1;reason='linear_equilibrium';success=true;
if ~isempty(free)
    Kff=K(free,free);diagK=full(diag(Kff));
    if any(~isfinite(diagK)|diagK<=0)
        success=false;reason='unrestrained_or_zero_stiffness_dof';
    else
        scale=1./sqrt(diagK);D=spdiags(scale,0,numel(free),numel(free));
        Ks=D*Kff*D;Ks=(Ks+Ks')/2;
        [C,flag]=chol(Ks,'lower');
        if flag~=0
            success=false;reason='singular_or_nonpositive_stiffness';
        else
            reciprocalCondition=1/condest(Ks);
            if ~isfinite(reciprocalCondition)||reciprocalCondition<options.MinimumReciprocalCondition
                success=false;reason='numerically_singular_stiffness';
            else
                us=C'\(C\(scale.*F(free)));u(free)=scale.*us;
            end
        end
    end
end
if ~success
    u(:)=NaN;u(fixed)=0;
end
residual=K*u-F;
if isempty(free),residualNorm=0;forceScale=0;
else,residualNorm=norm(residual(free),Inf);forceScale=norm(F(free),Inf);end
limit=options.AbsoluteResidualTolerance+options.RelativeResidualTolerance*forceScale;
if success && (~all(isfinite(u)) || residualNorm>limit)
    success=false;reason='free_equilibrium_residual_exceeded';
end
reactions=zeros(nd,1);reactions(fixed)=residual(fixed);
endForces=zeros(nm,12);localDisplacements=zeros(nm,12);
for m=1:nm
    ul=T(:,:,m)*u(dof(m,:));localDisplacements(m,:)=ul';
    endForces(m,:)=(ke(:,:,m)*ul)';
end
nodalU=reshape(u,6,[])';nodalR=reshape(reactions,6,[])';
netForce=sum(nodalR(:,1:3)+model.loads(:,1:3),1);
netMoment=sum(nodalR(:,4:6)+model.loads(:,4:6)+cross(nodes,nodalR(:,1:3)+model.loads(:,1:3),2),1);
result=struct('success',success,'reason',reason,'displacements',nodalU,'displacement',nodalU, ...
    'displacementVector',u,'reactions',nodalR,'reactionVector',reactions, ...
    'endForces',endForces,'forceLocalI',endForces(:,1:6),'forceLocalJ',endForces(:,7:12), ...
    'sectionForcesI',-endForces(:,1:6), ...
    'sectionForcesJ',endForces(:,7:12),'localDisplacements',localDisplacements, ...
    'sectionForceOrder',{{'N','V2','V3','T','M2','M3'}},'localAxes',R,'lengths',L, ...
    'K',K,'freeDofs',free,'fixedDofs',fixed,'elementDofs',dof, ...
    'elementLocalStiffness',ke,'elementTransform',T,'residual',residual, ...
    'residualNorm',residualNorm,'residualTolerance',limit,'reciprocalConditionScaled',reciprocalCondition, ...
    'globalForceClosure',netForce,'globalMomentClosure',netMoment, ...
    'strainEnergy',0.5*u'*(K*u),'options',options, ...
    'scope','Linear elastic frame; fixed unstressed construction geometry, no geometric stiffness or nonlinear stability verification');
end

function k=bending(EI,GA,L)
phi=12*EI/(GA*L^2);
k=EI/(L^3*(1+phi))*[12,6*L,-12,6*L; ...
    6*L,(4+phi)*L^2,-6*L,(2-phi)*L^2; ...
    -12,-6*L,12,-6*L;6*L,(2-phi)*L^2,-6*L,(4+phi)*L^2];
end

function v=memberVector(v,n,name,allowInf)
validateattributes(v,{'numeric'},{'vector','real','positive','nonempty'},mfilename,name);
v=double(v(:));if isscalar(v),v=repmat(v,n,1);end
if numel(v)~=n||any(isnan(v))||(~allowInf&&any(~isfinite(v)))
    error('frame3d:Property','%s must contain a positive scalar or one value per member.',name);
end
end
