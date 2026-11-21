function out=st_frame_batch(model,nodalCases,lineLoadsGlobal,caseNames)
%ST_FRAME_BATCH Linear rigid-joint frame cases with uniform member body loads.
% nodalCases: N x 6 x C; lineLoadsGlobal: E x 3 x C, force per metre.
% Consistent end forces include the fixed-end moments of transverse body load.
% Uses the preserved Timoshenko stiffness, one factorization and multiple RHS.
% All physical members and rigid joints remain; subdivisions are numerical only.
nn=size(model.nodes,1);ne=size(model.bars,1);nc=size(nodalCases,3);nd=6*nn;
validateattributes(nodalCases,{'numeric'},{'real','finite','size',[nn,6,nc]});
assert(isequal(size(lineLoadsGlobal,1),ne)&&size(lineLoadsGlobal,2)==3&&size(lineLoadsGlobal,3)==nc);
assert(numel(caseNames)==nc&&all(isfinite(lineLoadsGlobal(:))));
model.loads=zeros(nn,6);template=frame3d_linear(model);
assert(template.success,'sequential:Frame','Frame stiffness failed: %s',template.reason);
P=reshape(permute(nodalCases,[2,1,3]),nd,nc);
localLine=zeros(ne,3,nc);equivalent=zeros(ne,12,nc);
for e=1:ne
    L=template.lengths(e);R=template.localAxes(:,:,e);T=template.elementTransform(:,:,e);
    for c=1:nc
        q=R*lineLoadsGlobal(e,:,c)';localLine(e,:,c)=q';
        f=[q(1)*L/2;q(2)*L/2;q(3)*L/2;0;-q(3)*L^2/12;q(2)*L^2/12; ...
           q(1)*L/2;q(2)*L/2;q(3)*L/2;0;q(3)*L^2/12;-q(2)*L^2/12];
        equivalent(e,:,c)=f';ix=template.elementDofs(e,:);
        P(ix,c)=P(ix,c)+T'*f;
    end
end
free=template.freeDofs;fixed=template.fixedDofs;K=template.K;
U=zeros(nd,nc);
if ~isempty(free)
    Kff=K(free,free);scale=1./sqrt(full(diag(Kff)));D=spdiags(scale,0,numel(free),numel(free));
    C=chol((D*Kff*D+(D*Kff*D)')/2,'lower');
    U(free,:)=scale.*(C'\(C\(scale.*P(free,:))));
end
cases=cell(nc,1);
for c=1:nc
    residual=K*U(:,c)-P(:,c);reaction=zeros(nd,1);reaction(fixed)=residual(fixed);
    limit=template.options.AbsoluteResidualTolerance+template.options.RelativeResidualTolerance*norm(P(free,c),Inf);
    rnorm=norm(residual(free),Inf);assert(rnorm<=limit,'sequential:Residual','%s residual %.9g exceeds %.9g.',caseNames{c},rnorm,limit);
    endActions=zeros(ne,12);
    for e=1:ne
        T=template.elementTransform(:,:,e);ke=template.elementLocalStiffness(:,:,e);
        endActions(e,:)=(ke*T*U(template.elementDofs(e,:),c)-equivalent(e,:,c)')';
    end
    fi=-endActions(:,1:6);fj=endActions(:,7:12);
    % Conservative NONCOINCIDENT envelopes per numerical segment: exact extrema
    % of N and each moment are combined. This bounds the EC5 interactions.
    envelopes=zeros(2*ne,6);locations=zeros(ne,2);
    for e=1:ne
        L=template.lengths(e);q=localLine(e,:,c);a=fi(e,:);
        loc2=[0,L];if abs(q(3))>0,xc=a(3)/q(3);if xc>0&&xc<L,loc2(end+1)=xc;end,end
        loc3=[0,L];if abs(q(2))>0,xc=a(2)/q(2);if xc>0&&xc<L,loc3(end+1)=xc;end,end
        m2=a(5)+a(3)*loc2-.5*q(3)*loc2.^2;
        m3=a(6)-a(2)*loc3+.5*q(2)*loc3.^2;
        [M2,j2]=max(abs(m2));[M3,j3]=max(abs(m3));locations(e,:)=[loc2(j2),loc3(j3)];
        n=[a(1),a(1)-q(1)*L];v2=max(abs([a(2),a(2)-q(2)*L]));v3=max(abs([a(3),a(3)-q(3)*L]));
        envelopes([e,ne+e],:)=[min(n),v2,v3,abs(a(4)),M2,M3;max(n),v2,v3,abs(a(4)),M2,M3];
        predicted=[a(1:3)-q*L,a(4),a(5)+a(3)*L-.5*q(3)*L^2,a(6)-a(2)*L+.5*q(2)*L^2];
        assert(max(abs(predicted-fj(e,:)))<1e-7*max(1,max(abs(fj(e,:)))),'sequential:ElementEquilibrium','Distributed member recovery failed.');
    end
    applied=reshape(P(:,c),6,[])';Rnodal=reshape(reaction,6,[])';
    forceClosure=sum(applied(:,1:3)+Rnodal(:,1:3),1);
    momentClosure=sum(applied(:,4:6)+Rnodal(:,4:6)+cross(model.nodes,applied(:,1:3)+Rnodal(:,1:3),2),1);
    Unodal=reshape(U(:,c),6,[])';
    cases{c}=struct('name',caseNames{c},'success',true,'displacements',Unodal, ...
        'reactions',Rnodal,'appliedNodalForces',applied,'prescribedNodalForces',nodalCases(:,:,c), ...
        'lineLoadsGlobal',lineLoadsGlobal(:,:,c),'sectionForcesI',fi,'sectionForcesJ',fj, ...
        'envelopeActions',envelopes,'momentExtremeLocations',locations,'lineLoadsLocal',localLine(:,:,c), ...
        'equivalentLocalLoads',equivalent(:,:,c),'residualNorm',rnorm,'residualTolerance',limit, ...
        'globalForceClosure',forceClosure,'globalMomentClosure',momentClosure, ...
        'maximumAbsUZ',max(abs(U(3:6:end,c))),'maximumTranslation',max(vecnorm(Unodal(:,1:3),2,2)));
end
out=struct('success',true,'cases',{cases},'template',template,'loads',P,'displacements',U, ...
    'scope','First-order Timoshenko frame with exact uniform body-load resultants; no geometric stiffness or stability test');
end
