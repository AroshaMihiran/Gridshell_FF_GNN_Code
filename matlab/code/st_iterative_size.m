function design=st_iterative_size(base,material,designOptions,cases,groups,overrides)
%ST_ITERATIVE_SIZE Grouped feasible sizing, not a global volume optimum.
if nargin<6,overrides=struct();end
o=struct('MinimumSide',.05,'MaximumSide',1.5,'TargetRatio',.98,'Damping',.35, ...
    'MaximumRelativeStep',.25,'RelativeSideTolerance',1e-4,'MaximumIterations',200, ...
    'RoundingIncrement',.005,'MaximumRepairIterations',50);
for f=fieldnames(overrides)',assert(isfield(o,f{1}));o.(f{1})=overrides.(f{1});end
for f={'MinimumSide','MaximumSide','TargetRatio','Damping','MaximumRelativeStep','RelativeSideTolerance','RoundingIncrement'}
    validateattributes(o.(f{1}),{'numeric'},{'scalar','real','finite','positive'});
end
assert(o.MinimumSide<=o.MaximumSide&&o.TargetRatio<=1&&o.Damping<=1,'sequential:Options','Invalid sizing bounds or factors.');
validateattributes(o.MaximumIterations,{'numeric'},{'scalar','finite','integer','positive'});
validateattributes(o.MaximumRepairIterations,{'numeric'},{'scalar','finite','integer','positive'});
n=numel(base.originalA);groups=groups(:);validateattributes(groups,{'numeric'},{'numel',n,'integer','positive','finite'});ng=max(groups);
assert(isequal(unique(groups),(1:ng)'),'sequential:Groups','Group numbers must be consecutive.');
sizes=max(o.MinimumSide,sqrt(base.originalA(:)));groupSides=accumarray(groups,sizes,[ng,1],@max);sides=groupSides(groups);
assert(all(sides<=o.MaximumSide),'sequential:Options','Initial section exceeds maximum side.');
initial=st_design_evaluate(base,sides,material,designOptions,cases);current=initial;
history=zeros(0,7);sizeHistory=zeros(0,n);converged=false;reason='iteration_limit';started=tic;
for iteration=1:o.MaximumIterations
    [target,possible]=targets(current);
    if ~all(possible),reason='maximum_section_insufficient';break;end
    targetGroup=accumarray(groups,target,[ng,1],@max);target=targetGroup(groups);
    step=o.Damping*log(target./sides);step=max(-log(1+o.MaximumRelativeStep),min(log(1+o.MaximumRelativeStep),step));
    next=max(o.MinimumSide,min(o.MaximumSide,sides.*exp(step)));change=max(abs(next-sides)./sides);
    history(end+1,:)=[iteration,current.maxRatio,current.volume,current.maxAbsUZ,change,current.unfactoredTimberSelfWeight_kN,toc(started)]; %#ok<AGROW>
    sizeHistory(end+1,:)=sides'; %#ok<AGROW>
    if mod(iteration,10)==1,fprintf('Grouped sizing %d: U %.6f, volume %.5g m3, change %.3g\n',iteration,current.maxRatio,current.volume,change);end
    if current.memberChecksPass&&change<=o.RelativeSideTolerance,converged=true;reason='continuous_sections_converged';break;end
    sides=next;current=st_design_evaluate(base,sides,material,designOptions,cases);
end
continuous=current;repairHistory=zeros(0,4);
if converged
    sides=ceil((sides-1e-12)/o.RoundingIncrement)*o.RoundingIncrement;
    for repair=0:o.MaximumRepairIterations
        assert(all(sides<=o.MaximumSide+1e-12),'sequential:Bound','Rounded section exceeds bound.');
        current=st_design_evaluate(base,sides,material,designOptions,cases);
        repairHistory(end+1,:)=[repair,current.maxRatio,current.volume,current.maxAbsUZ]; %#ok<AGROW>
        if current.memberChecksPass,reason='rounded_sections_freshly_reanalysed';break;end
        if repair==o.MaximumRepairIterations,converged=false;reason='rounded_repair_limit';break;end
        failing=unique(groups(current.memberRatios>1));sides(ismember(groups,failing))=sides(ismember(groups,failing))+o.RoundingIncrement;
    end
end
for g=1:ng,assert(max(sides(groups==g))-min(sides(groups==g))<1e-12);end
design=struct('success',converged&&current.memberChecksPass,'reason',reason,'initial',initial, ...
    'continuous',continuous,'final',current,'history',history,'sizeHistory',sizeHistory, ...
    'repairHistory',repairHistory,'groups',groups,'options',o,'seconds',toc(started), ...
    'geometryNeverUpdatedDuringSizing',true,'fftCallsDuringSizing',0, ...
    'scope','Feasible grouped sections for the listed normal-stress/member-buckling checks; no exact-unity or global-optimum claim');

    function [target,possible]=targets(ev)
        rows=ev.stationMemberRows;L=base.originalMemberLengths(rows);mat=material;mat.kmod=ev.stationKmod;
        lo=repmat(o.MinimumSide,n,1);hi=repmat(o.MaximumSide,n,1);
        umin=ratios(lo);possible=ratios(hi)<=o.TargetRatio;
        for j=1:35
            mid=(lo+hi)/2;okay=ratios(mid)<=o.TargetRatio;hi(okay)=mid(okay);lo(~okay)=mid(~okay);
        end
        target=hi;target(umin<=o.TargetRatio)=o.MinimumSide;
        function u=ratios(d)
            a=d(rows);s=struct('A',a.^2,'I2',a.^4/12,'I3',a.^4/12,'W2',a.^3/6,'W3',a.^3/6);
            check=timber_frame_checks(ev.stationActions,s,L,mat,designOptions);
            u=accumarray(rows,check.ratio,[n,1],@max);
        end
    end
end
