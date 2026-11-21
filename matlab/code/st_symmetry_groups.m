function [groups,info]=st_symmetry_groups(nodes,bars,restraints,mode)
%ST_SYMMETRY_GROUPS Section groups from verified initial-layout reflections.
% Loads need not be symmetric: all members in each group use the worst demand.
if nargin<4,mode='xy';end
mode=validatestring(mode,{'none','x','y','xy'});n=size(nodes,1);m=size(bars,1);
groups=(1:m)';maps=zeros(n,0);axesToUse=[];
if contains(mode,'x'),axesToUse(end+1)=1;end
if contains(mode,'y'),axesToUse(end+1)=2;end
center=(max(nodes,[],1)+min(nodes,[],1))/2;tol=1e-8*max(1,norm(max(nodes)-min(nodes)));
for axis=axesToUse
    reflected=nodes;reflected(:,axis)=2*center(axis)-nodes(:,axis);map=zeros(n,1);
    for i=1:n
        [distance,map(i)]=min(vecnorm(nodes-reflected(i,:),2,2));
        assert(distance<=tol,'sequential:Symmetry','Requested reflection does not map the initial nodes.');
    end
    assert(numel(unique(map))==n&&isequal(logical(restraints),logical(restraints(map,:))), ...
        'sequential:Symmetry','Reflection does not preserve nodes and supports.');
    sortedBars=sort(bars,2);
    for e=1:m
        pair=sort(map(bars(e,:)));j=find(all(sortedBars==pair(:)',2));
        assert(numel(j)==1,'sequential:Symmetry','Reflection does not preserve physical-member topology.');
        a=groups(e);b=groups(j);groups(groups==max(a,b))=min(a,b);
    end
    maps(:,end+1)=map; %#ok<AGROW>
end
[~,~,groups]=unique(groups);
info=struct('mode',mode,'numberOfGroups',max(groups),'groups',groups,'nodeReflectionMaps',maps, ...
    'referenceCenter',center,'geometryTolerance_m',tol,'loadsAreNotSymmetrized',true, ...
    'rule','Identical section in each reflection group; governing demand envelopes all group members and all design cases');
end
