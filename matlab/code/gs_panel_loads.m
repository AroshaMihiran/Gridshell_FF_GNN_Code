function [F,info]=gs_panel_loads(nodes,panels,q1,q2,options)
%GS_PANEL_LOADS Geometry-aware load-only T3, Q4 and convex polygon surfaces.
% Units: coordinates m, q1/q2 kN/m^2, returned force kN. No panel stiffness,
% new structural nodes or bars are introduced. Existing panel identities stay
% fixed. q1 is downward load per CURRENT surface area; q2 is a NONNEGATIVE
% magnitude per horizontal projected area. WindDirection is 'downward'
% (default) or 'upward'. Isolated round2 extension: WindDirection='signed'
% accepts signed q2, positive downward and negative upward per panel.
% Only vertical wind is supported by this release.
%
% T3: barycentric surface. Q4: bilinear surface, no preferred diagonal.
% 5+ nodes: mean-value shape functions on a fixed convex planar reference
% polygon, mapped to the current spatial corner coordinates. Collinear
% boundary nodes are retained. Polygon integration uses a centroid fan only
% as numerical quadrature; load uses ALL original polygon shape functions.
% Concave, self-intersecting, collapsed or folded panels are rejected.
% A concave roof footprint may still contain valid convex individual faces.
%
% ReferenceNodes must be retained from the first geometry. IntegrationOrder
% defaults to 8 and remains fixed during equilibrium iterations. Cache
% info.PanelIntegration in options.PanelIntegration for faster repeated calls.
% Increasing the order is an independent final quadrature accuracy check;
% force/moment conservation alone does not prove quadrature accuracy.
if nargin<5||isempty(options),options=struct();end
if ~isstruct(options)||~isscalar(options),error('gs:LoadOptions','Options must be a scalar structure.');end
validateattributes(nodes,{'numeric'},{'real','finite','2d','ncols',3,'nonempty'});
nodes=double(nodes);nn=size(nodes,1);
if ~iscell(panels)||isempty(panels)||~isvector(panels),error('gs:LoadPanels','Panels must be a nonempty cell vector.');end
panels=panels(:);np=numel(panels);
q1=panelVector(q1,np,'q1');q2=panelVector(q2,np,'q2');
if any(q1<0),error('gs:LoadMagnitude','q1 must be nonnegative.');end
if any(q2<0) && ~strcmpi(char(option(options,'WindDirection','downward')),'signed')
    error('gs:LoadMagnitude','Negative q2 requires WindDirection=signed.');
end
deadFactor=panelVector(option(options,'DeadLoadFactor',1),np,'DeadLoadFactor');
windFactor=panelVector(option(options,'WindLoadFactor',1),np,'WindLoadFactor');
if any(deadFactor<0)||any(windFactor<0),error('gs:LoadFactor','Load factors must be nonnegative.');end
windMode=lower(char(option(options,'WindMode','vertical')));
if ~strcmp(windMode,'vertical'),error('gs:WindMode','This revision implements vertical wind only, with a projected-area basis.');end
direction=lower(char(option(options,'WindDirection','downward')));
if ~ismember(direction,{'downward','upward','signed'}),error('gs:WindDirection','WindDirection must be downward, upward or signed.');end
windSign=-1;if strcmp(direction,'upward'),windSign=1;end
ref=option(options,'ReferenceNodes',nodes);if isempty(ref),ref=nodes;end
validateattributes(ref,{'numeric'},{'real','finite','size',size(nodes)});ref=double(ref);
order=option(options,'IntegrationOrder',8);
validateattributes(order,{'numeric'},{'scalar','integer','>=',2,'<=',128,'finite'});
tol=option(options,'GeometryRelativeTolerance',1e-10);
validateattributes(tol,{'numeric'},{'scalar','positive','finite','<',.01});
cache=option(options,'PanelIntegration',[]);
if ~isempty(cache)&&(~iscell(cache)||numel(cache)~=np),error('gs:IntegrationCache','Cache must have one entry per panel.');end
rules=cell(np,1);method=cell(np,1);nodal=zeros(nn,3);
areas=zeros(np,1);projected=zeros(np,1);resultants=zeros(np,3);moments=zeros(np,3);
panelNodal=cell(np,1);surfaceWeights=cell(np,1);projectedWeights=cell(np,1);
minimumJacobian=zeros(np,1);quadratureCount=zeros(np,1);
for p=1:np
    ids=panels{p};
    validateattributes(ids,{'numeric'},{'vector','integer','positive','finite'});
    ids=double(ids(:)');if numel(ids)>1&&ids(1)==ids(end),ids(end)=[];end
    if numel(ids)<3||numel(unique(ids))~=numel(ids)||any(ids>nn),error('gs:PanelConnectivity','Panel %d needs distinct existing corner nodes.',p);end
    panels{p}=ids;v=nodes(ids,:);vr=ref(ids,:);
    if isempty(cache)
        rules{p}=integrationRule(vr,ids,order,tol,p);
    else
        c=cache{p};
        if ~isstruct(c)||~isfield(c,'schemaVersion')||~strcmp(c.schemaVersion,'load-surface-1')|| ...
                ~isequal(c.ids,ids)||~isequal(c.referenceCoordinates,vr)||c.order~=order
            error('gs:IntegrationCache','Panel %d cache does not match reference geometry, connectivity or order.',p);
        end
        rules{p}=c;
    end
    r=rules{p};method{p}=r.method;
    [~,normal,~]=validPolygon(v,tol,p,false);
    dx=r.dN1*v;dy=r.dN2*v;av=cross(dx,dy,2);
    js=sqrt(sum(av.^2,2));jh=abs(av(:,3));
    if any(~isfinite([js;jh]))
        error('gs:LoadOverflow','Panel %d surface Jacobian overflow; check geometry and units.',p);
    end
    orient=av*normal';
    if any(js<=tol*max(js))||any(orient<=tol*js)
        error('gs:FoldedSurface','Panel %d has a degenerate or folded interpolated surface.',p);
    end
    ztol=tol*max(js);
    if any(av(:,3)>ztol)&&any(av(:,3)<-ztol)
        error('gs:FoldedProjection','Panel %d has a folded horizontal projection.',p);
    end
    ws=r.N'*(r.w.*js);wh=r.N'*(r.w.*jh);
    if any(ws < -tol*sum(ws))||any(wh < -tol*max(sum(wh),eps))
        error('gs:NegativeLoadWeight','Panel %d has negative tributary weights.',p);
    end
    f=zeros(numel(ids),3);
    f(:,3)=-deadFactor(p)*q1(p)*ws+windSign*windFactor(p)*q2(p)*wh;
    nodal(ids,:)=nodal(ids,:)+f;
    qf=zeros(size(r.N,1),3);
    qf(:,3)=r.w.*(-deadFactor(p)*q1(p)*js+windSign*windFactor(p)*q2(p)*jh);
    if any(~isfinite([f(:);qf(:)]))
        error('gs:LoadOverflow','Panel %d load overflow; check load magnitudes and units.',p);
    end
    xyz=r.N*v;
    areas(p)=sum(r.w.*js);projected(p)=sum(r.w.*jh);
    resultants(p,:)=sum(qf,1);moments(p,:)=sum(cross(xyz,qf,2),1);
    panelNodal{p}=f;surfaceWeights{p}=ws;projectedWeights{p}=wh;
    minimumJacobian(p)=min(js);quadratureCount(p)=numel(r.w);
end
F=reshape(nodal',[],1);
totalForce=sum(nodal,1);totalMoment=sum(cross(nodes,nodal,2),1);
if any(~isfinite([F;resultants(:);moments(:);totalForce(:);totalMoment(:)]))
    error('gs:LoadOverflow','Assembled load or moment overflow; check inputs and units.');
end
info=struct('Panels',{panels},'PanelIntegration',{rules},'panelIDs',(1:np)', ...
    'panelAreas',areas,'panelProjectedAreas',projected,'panelResultants',resultants, ...
    'panelMoments',moments,'nodalForces',nodal,'panelNodalForces',{panelNodal}, ...
    'surfaceNodalAreas',{surfaceWeights},'projectedNodalAreas',{projectedWeights}, ...
    'totalForce',totalForce,'totalMoment',totalMoment, ...
    'forceResidual',totalForce-sum(resultants,1), ...
    'momentResidual',totalMoment-sum(moments,1), ...
    'methods',{method},'WindMode',windMode,'WindDirection',direction, ...
    'DeadLoadFactor',deadFactor,'WindLoadFactor',windFactor,'q1',q1,'q2',q2, ...
    'IntegrationOrder',order,'quadraturePointCounts',quadratureCount, ...
    'minimumSurfaceJacobian',minimumJacobian,'integrationCacheReused',~isempty(cache), ...
    'surfaceDefinition','T3 barycentric; Q4 bilinear; convex 5+ mean-value reference surface', ...
    'massBasis','q1 is prescribed per current candidate surface area, not conserved cladding mass');
end

function r=integrationRule(v,ids,order,tol,p)
n=size(v,1);[xy,~,~]=validPolygon(v,tol,p,n>4);
[g,w]=gaussRule(order);
if n==3
    N=[1/3,1/3,1/3];d1=[-1,1,0];d2=[-1,0,1];wt=.5;name='T3 barycentric';
elseif n==4
    [a,b]=ndgrid(g,g);[wa,wb]=ndgrid(w,w);a=a(:);b=b(:);wt=wa(:).*wb(:);
    sx=[-1,1,1,-1];sy=[-1,-1,1,1];
    N=(1+a*sx).*(1+b*sy)/4;
    d1=repmat(sx,numel(a),1).*(1+b*sy)/4;
    d2=repmat(sy,numel(b),1).*(1+a*sx)/4;name='Q4 bilinear';
else
    % Convex reference domain, centroid fan for QUADRATURE ONLY. The load
    % basis is polygon-wide mean-value interpolation, retaining collinear
    % boundary nodes and avoiding arbitrary diagonal load assignment.
    center=mean(xy,1);u=(g+1)/2;wu=w/2;
    [uu,vv]=ndgrid(u,u);[ww1,ww2]=ndgrid(wu,wu);
    uu=uu(:);vv=vv(:);basew=ww1(:).*ww2(:);N=[];d1=[];d2=[];wt=[];
    for j=1:n
        a=xy(j,:)-center;b=xy(mod(j,n)+1,:)-center;
        % Average the two equivalent Duffy parameterizations so reversing a
        % panel's winding cannot alter its finite-order quadrature weights.
        for orientation=1:2
            if orientation==1,aa=a;bb=b;else,aa=b;bb=a;end
            points=center+uu*aa+((1-uu).*vv)*bb;
            tw=basew.*(1-uu)*abs(cross2(a,b))/2;
            [Ni,dxx,dyy]=gs_load_polygon_coordinates(xy,points);
            N=[N;Ni];d1=[d1;dxx];d2=[d2;dyy];wt=[wt;tw]; %#ok<AGROW>
        end
    end
    name='Polygon mean-value';
end
if max(abs(sum(N,2)-1))>1e-11|| ...
        any(abs(sum(d1,2))>1e-10*max(1,sum(abs(d1),2)))|| ...
        any(abs(sum(d2,2))>1e-10*max(1,sum(abs(d2),2)))
    error('gs:ShapeFunction','Panel %d violates interpolation identities.',p);
end
r=struct('schemaVersion','load-surface-1','ids',ids,'referenceCoordinates',v, ...
    'order',order,'method',name,'N',N,'dN1',d1,'dN2',d2,'w',wt);
end

function [xy,normal,scale]=validPolygon(v,tol,p,requirePlanar)
n=size(v,1);center=mean(v,1);vc=v-center;
scale=max(sqrt(sum(vc.^2,2)));if scale==0,error('gs:DegeneratePanel','Panel %d collapsed.',p);end
edges=v([2:n,1],:)-v;
if any(sqrt(sum(edges.^2,2))<=tol*scale),error('gs:DegeneratePanel','Panel %d has a collapsed edge.',p);end
vec=sum(cross(vc,vc([2:n,1],:),2),1)/2;
if norm(vec)<=tol*scale^2,error('gs:DegeneratePanel','Panel %d has zero vector area or intersecting boundaries.',p);end
normal=vec/norm(vec);
if requirePlanar&&max(abs(vc*normal'))>tol*scale
    error('gs:ReferencePlanarity','Reference polygon %d with 5+ corners must be planar; retain its initial reference nodes.',p);
end
[~,axis]=min(abs(normal));unit=zeros(1,3);unit(axis)=1;
t1=cross(unit,normal);t1=t1/norm(t1);t2=cross(normal,t1);
xy=[vc*t1',vc*t2'];ee=xy([2:n,1],:)-xy;
turn=ee(:,1).*ee([2:n,1],2)-ee(:,2).*ee([2:n,1],1);
if any(turn < -tol*scale^2)||sum(turn>tol*scale^2)<3
    error('gs:ConvexPanel','Panel %d must project to a convex polygon with at least three noncollinear corners.',p);
end
for a=1:n
    for b=a+1:n
        if b==a+1||(a==1&&b==n),continue;end
        x1=xy(a,:);x2=xy(mod(a,n)+1,:);y1=xy(b,:);y2=xy(mod(b,n)+1,:);
        if any(max(min(x1,x2),min(y1,y2))>min(max(x1,x2),max(y1,y2))+tol*scale),continue;end
        s1=cross2(x2-x1,y1-x1);s2=cross2(x2-x1,y2-x1);
        t1c=cross2(y2-y1,x1-y1);t2c=cross2(y2-y1,x2-y1);
        if s1*s2<=0&&t1c*t2c<=0,error('gs:PanelIntersection','Panel %d boundary self-intersects.',p);end
    end
end
end

function z=cross2(a,b),z=a(1)*b(2)-a(2)*b(1);end
function [x,w]=gaussRule(n)
persistent cacheN cacheX cacheW
if ~isempty(cacheN)&&cacheN==n,x=cacheX;w=cacheW;return;end
k=(1:n-1)';beta=k./sqrt(4*k.^2-1);[V,D]=eig(diag(beta,1)+diag(beta,-1));
[x,ix]=sort(diag(D));w=2*(V(1,ix)').^2;cacheN=n;cacheX=x;cacheW=w;
end
function v=panelVector(v,n,name)
validateattributes(v,{'numeric'},{'vector','real','finite'},mfilename,name);v=double(v(:));
if isscalar(v),v=repmat(v,n,1);end
if numel(v)~=n,error('gs:LoadSize','%s must be scalar or have one entry per panel.',name);end
end
function v=option(o,name,fallback)
if isfield(o,name),v=o.(name);else,v=fallback;end
end
