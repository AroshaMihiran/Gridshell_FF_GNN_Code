function [groups,info]=rt_plan_groups(nodes,bars,restraints)
% Verified plan isometries; never mirror/average panel loads or actions.
n=size(nodes,1);m=size(bars,1);center=(max(nodes)+min(nodes))/2;
tol=1e-8*max(1,norm(max(nodes)-min(nodes)));sorted=sort(bars,2);
names={'reflect_x','reflect_y','rotate_180','reflect_diagonal','reflect_antidiagonal'};
transforms={diag([-1 1 1]),diag([1 -1 1]),diag([-1 -1 1]),[0 1 0;1 0 0;0 0 1],[0 -1 0;-1 0 0;0 0 1]};
groups=(1:m)';accepted={};maps=[];rejected={};
for a=1:numel(names)
    trial=(nodes-center)*transforms{a}'+center;map=zeros(n,1);okay=true;
    for i=1:n,[distance,map(i)]=min(vecnorm(nodes-trial(i,:),2,2));okay=okay&&distance<=tol;end
    okay=okay&&numel(unique(map))==n&&isequal(logical(restraints),logical(restraints(map,:)));
    [exists,memberMap]=ismember(sort(map(bars),2),sorted,'rows');okay=okay&&all(exists);
    if ~okay,rejected{end+1}=names{a};continue;end %#ok<AGROW>
    for e=1:m,j=memberMap(e);u=groups(e);v=groups(j);groups(groups==max(u,v))=min(u,v);end
    maps(:,end+1)=map;accepted{end+1}=names{a}; %#ok<AGROW>
end
[~,~,groups]=unique(groups);
info=struct('mode','verified_plan_isometries','numberOfGroups',max(groups),'groups',groups, ...
    'acceptedTransformations',{accepted},'rejectedTransformations',{rejected},'nodeMaps',maps, ...
    'referenceCenter',center,'geometryTolerance_m',tol,'loadsAreNotSymmetrized',true, ...
    'rule','Common square side within each orbit of valid initial-plan transformations; physical nodes, supports and member topology verified.');
end
