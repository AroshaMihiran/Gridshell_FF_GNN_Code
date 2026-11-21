function [panels,info]=gs_identify_panels(nodes,bars,boundaryLoops)
%GS_IDENTIFY_PANELS Planar faces with explicitly declared load-only boundaries.
% boundaryLoops is a cell array of ORDERED NODE ROW indices. One enclosing
% loop and optional disjoint interior hole loops are allowed, in either
% orientation. Missing outline edges are virtual load boundaries only: the
% input BARS is never changed and no stiffness/member is created.
% Long boundary edges are split at existing collinear nodes. Physical bars
% crossing without a shared end node, or passing through another node, are
% rejected. Nonplanar initial geometry requires explicit panel connectivity
% and is not supported by this planar discovery routine.
if nargin<3,boundaryLoops={};end
validateattributes(nodes,{'numeric'},{'2d','ncols',3,'real','finite','nonempty'});
validateattributes(bars,{'numeric'},{'2d','ncols',2,'integer','positive','finite'});
nodes=double(nodes);bars=double(bars);nn=size(nodes,1);
if nn<3||isempty(bars)||any(bars(:)>nn)||any(bars(:,1)==bars(:,2))
    error('GridShell:PanelTopology','At least three nodes and valid nonempty bars are required.');
end
if ~iscell(boundaryLoops),boundaryLoops={boundaryLoops};end
origin=mean(nodes,1);centered=nodes-origin;
[~,~,v]=svd(centered,0);normal=v(:,end);
[~,axis]=max(abs(normal));if normal(axis)<0,normal=-normal;end
[~,axis]=min(abs(normal));basis=zeros(3,1);basis(axis)=1;
u=basis-normal*(normal'*basis);u=u/norm(u);w=cross(normal,u);
xy=[centered*u,centered*w];scale=max(vecnorm(centered,2,2));
if ~isfinite(scale)||scale==0,error('GridShell:PanelGeometry','Degenerate node geometry.');end
tol=1e-9*scale;areaTol=1e-10*scale^2;
if max(abs(centered*normal))>tol
    error('GridShell:PanelPlanarity','Panel discovery requires a planar reference layout.');
end
for a=1:nn-1
    if any(vecnorm(xy(a+1:end,:)-xy(a,:),2,2)<=tol)
        error('GridShell:PanelGeometry','Coincident reference nodes are not allowed.');
    end
end
physical=sort(bars,2);
if size(unique(physical,'rows'),1)~=size(physical,1)
    error('GridShell:PanelTopology','Repeated physical member endpoint pairs.');
end
for e=1:size(physical,1)
    a=physical(e,1);b=physical(e,2);
    on=pointsOnSegment(xy,xy(a,:),xy(b,:),tol);
    on([a,b])=false;
    if any(on)
        error('GridShell:MemberThroughNode','A physical member passes through an existing node; split the member explicitly.');
    end
end
expanded=cell(numel(boundaryLoops),1);boundaryEdges=zeros(0,2);loopAreas=zeros(numel(boundaryLoops),1);
for k=1:numel(boundaryLoops)
    ids=double(boundaryLoops{k}(:)');
    if numel(ids)>1&&ids(end)==ids(1),ids(end)=[];end
    if numel(ids)<3||any(~isfinite(ids))||any(ids~=fix(ids))||any(ids<1|ids>nn)||numel(unique(ids))~=numel(ids)
        error('GridShell:BoundaryNodes','Boundary loops need at least three distinct existing node rows.');
    end
    out=[];
    for j=1:numel(ids)
        a=ids(j);b=ids(mod(j,numel(ids))+1);d=xy(b,:)-xy(a,:);
        on=find(pointsOnSegment(xy,xy(a,:),xy(b,:),tol));
        alpha=(xy(on,:)-xy(a,:))*d'/sum(d.^2);[~,ord]=sort(alpha);on=on(ord);
        out=[out,on(1:end-1)']; %#ok<AGROW>
    end
    if numel(unique(out))~=numel(out)
        error('GridShell:BoundarySelfIntersection','Boundary loops repeat a node or overlap themselves.');
    end
    expanded{k}=out;ed=sort([out(:),circshift(out(:),-1)],2);
    assertNoCrossings(xy,ed,tol,areaTol);
    loopAreas(k)=abs(signedArea(xy(out,:)));
    if loopAreas(k)<=areaTol,error('GridShell:BoundaryArea','Boundary loop has negligible area.');end
    boundaryEdges=[boundaryEdges;ed]; %#ok<AGROW>
end
outer=0;holes=[];domainArea=NaN;
if ~isempty(expanded)
    [~,outer]=max(loopAreas);holes=setdiff(1:numel(expanded),outer);
    outerXY=xy(expanded{outer},:);
    for k=holes
        [in,on]=inpolygon(xy(expanded{k},1),xy(expanded{k},2),outerXY(:,1),outerXY(:,2));
        if ~all(in&~on),error('GridShell:BoundaryContainment','All hole loops must lie strictly inside the outer boundary.');end
    end
    for ia=1:numel(holes)
        a=holes(ia);va=xy(expanded{a},:);
        for ib=ia+1:numel(holes)
            b=holes(ib);vb=xy(expanded{b},:);
            if any(inpolygon(va(:,1),va(:,2),vb(:,1),vb(:,2)))||any(inpolygon(vb(:,1),vb(:,2),va(:,1),va(:,2)))
                error('GridShell:BoundaryContainment','Hole loops must be disjoint and may not contain each other.');
            end
        end
    end
    assertNoCrossings(xy,boundaryEdges,tol,areaTol);
    for e=1:size(physical,1)
        pts=[xy(physical(e,:),:);mean(xy(physical(e,:),:),1)];
        if ~all(inDomain(pts,xy,expanded,outer,holes,true))
            error('GridShell:MemberOutsideBoundary','A member lies outside the declared loaded boundary or inside an opening.');
        end
    end
    domainArea=loopAreas(outer)-sum(loopAreas(holes));
end
boundaryEdges=unique(boundaryEdges,'rows');
virtual=setdiff(boundaryEdges,physical,'rows');
edges=unique([physical;boundaryEdges],'rows');
assertNoCrossings(xy,edges,tol,areaTol);
ne=size(edges,1);from=[edges(:,1);edges(:,2)];to=[edges(:,2);edges(:,1)];
twin=[(ne+1:2*ne)';(1:ne)'];outgoing=cell(nn,1);
for a=1:nn
    hs=find(from==a);d=xy(to(hs),:)-xy(a,:);[~,ord]=sort(atan2(d(:,2),d(:,1)));
    outgoing{a}=hs(ord);
end
visited=false(2*ne,1);panels={};areas=[];
for first=1:2*ne
    if visited(first),continue;end
    h=first;cycle=[];
    for step=1:2*ne+1
        if visited(h)
            if h~=first,error('GridShell:PanelTopology','Face walk encountered an inconsistent cycle.');end
            break;
        end
        visited(h)=true;cycle(end+1)=from(h); %#ok<AGROW>
        hs=outgoing{to(h)};pos=find(hs==twin(h),1);
        h=hs(mod(pos-2,numel(hs))+1);
    end
    a=signedArea(xy(cycle,:));
    if a<=areaTol,continue;end % Clockwise exterior/unbounded face.
    if numel(unique(cycle))~=numel(cycle)
        error('GridShell:OpenInternalEdges','A bounded face contains a dangling member or bridge; specify a complete panel topology.');
    end
    point=interiorPoint(xy(cycle,:),tol);
    if outer>0
        if ~inDomain(point,xy,expanded,outer,holes,false),continue;end
        for k=holes
            holePoint=interiorPoint(xy(expanded{k},:),tol);
            [in,on]=inpolygon(holePoint(1),holePoint(2),xy(cycle,1),xy(cycle,2));
            if in&&~on
                error('GridShell:PanelWithHole','A single panel encloses an opening; add connecting panel boundaries or provide a supported partition.');
            end
        end
    end
    [~,j]=min(cycle);cycle=circshift(cycle,[0,1-j]);
    panels{end+1,1}=cycle;areas(end+1,1)=a; %#ok<AGROW>
end
if isempty(panels)
    error('GridShell:NoPanels','No closed loaded panels found. Supply an explicit boundary_nodes.csv for missing perimeter members.');
end
% Deterministic IDs independent of bar ordering and boundary-loop direction.
keys=zeros(numel(panels),max(cellfun(@numel,panels)));
for k=1:numel(panels),keys(k,1:numel(panels{k}))=panels{k};end
[~,order]=sortrows(keys);panels=panels(order);areas=areas(order);
if outer>0&&abs(sum(areas)-domainArea)>1e-8*max(scale^2,domainArea)
    error('GridShell:BoundaryCoverage','Discovered panels do not cover the declared outer boundary minus holes.');
end
info=struct('method','planar face walk with explicit load-only boundary edges', ...
    'boundaryLoops',{expanded},'outerBoundaryIndex',outer,'holeBoundaryIndices',holes, ...
    'boundaryEdges',boundaryEdges,'virtualEdges',virtual,'topologyEdges',edges, ...
    'physicalMemberCount',size(bars,1),'virtualEdgeCount',size(virtual,1), ...
    'physicalMembersUnchanged',true,'panelAreas',areas,'totalPanelArea',sum(areas), ...
    'declaredDomainArea',domainArea,'panelCount',numel(panels), ...
    'planeOrigin',origin,'planeBasis',[u,w],'planeNormal',normal', ...
    'boundaryWasExplicit',~isempty(expanded));
end

function a=signedArea(v)
w=v([2:end,1],:);a=.5*sum(v(:,1).*w(:,2)-w(:,1).*v(:,2));
end
function on=pointsOnSegment(p,a,b,tol)
d=b-a;len=norm(d);alpha=(p-a)*d'/len^2;
distance=abs((p(:,1)-a(1))*d(2)-(p(:,2)-a(2))*d(1))/len;
on=distance<=tol & alpha>=-tol/len & alpha<=1+tol/len;
end
function assertNoCrossings(xy,edges,tol,areaTol)
for i=1:size(edges,1)-1
    ids=edges(i,:);a=xy(ids(1),:);b=xy(ids(2),:);
    for j=i+1:size(edges,1)
        js=edges(j,:);c=xy(js(1),:);d=xy(js(2),:);common=intersect(ids,js);
        if numel(common)==2,error('GridShell:BoundaryOverlap','Repeated boundary edge.');end
        if numel(common)==1
            o=xy(common,:);p=xy(setdiff(ids,common),:)-o;q=xy(setdiff(js,common),:)-o;
            if abs(cross2(p,q))<=areaTol&&dot(p,q)>tol^2
                error('GridShell:BoundaryOverlap','Collinear graph segments overlap.');
            end
            continue;
        end
        ab1=cross2(b-a,c-a);ab2=cross2(b-a,d-a);cd1=cross2(d-c,a-c);cd2=cross2(d-c,b-c);
        proper=((ab1>areaTol&&ab2<-areaTol)||(ab1<-areaTol&&ab2>areaTol))&& ...
            ((cd1>areaTol&&cd2<-areaTol)||(cd1<-areaTol&&cd2>areaTol));
        touching=pointsOnSegment(c,a,b,tol)||pointsOnSegment(d,a,b,tol)|| ...
            pointsOnSegment(a,c,d,tol)||pointsOnSegment(b,c,d,tol);
        if proper||touching
            error('GridShell:BoundaryCrossing','Graph segments intersect without a shared endpoint; provide an explicit connected topology.');
        end
    end
end
end
function v=cross2(a,b),v=a(1)*b(2)-a(2)*b(1);end
function good=inDomain(p,xy,loops,outer,holes,allowBoundary)
v=xy(loops{outer},:);[inside,on]=inpolygon(p(:,1),p(:,2),v(:,1),v(:,2));
if allowBoundary,good=inside;else,good=inside&~on;end
for k=holes
    v=xy(loops{k},:);[inside,on]=inpolygon(p(:,1),p(:,2),v(:,1),v(:,2));
    if allowBoundary,good=good&(~inside|on);else,good=good&~inside;end
end
end
function p=interiorPoint(v,tol)
p=mean(v,1);[in,on]=inpolygon(p(1),p(2),v(:,1),v(:,2));if in&&~on,return;end
sgn=sign(signedArea(v));
for k=1:size(v,1)
    a=v(k,:);b=v(mod(k,size(v,1))+1,:);d=b-a;normal=sgn*[-d(2),d(1)]/norm(d);
    for factor=[10,100,1000]
        p=(a+b)/2+factor*tol*normal;[in,on]=inpolygon(p(1),p(2),v(:,1),v(:,2));
        if in&&~on,return;end
    end
end
error('GridShell:PanelGeometry','Could not find an interior point for a panel.');
end
