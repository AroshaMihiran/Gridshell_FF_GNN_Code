function report=run_load_comparison()
root=fileparts(mfilename('fullpath'));addpath(root,fullfile(root,'code'),fullfile(root,'legacy_loads'));
output=fullfile(root,'..','..','outputs','load_comparison');if ~isfolder(output),mkdir(output);end
o=gs_options(fullfile(root,'inputs'),struct('ElasticModulusFactor',.84,'WindDirection','upward', ...
    'DeadLoadFactor',1,'WindLoadFactor',1,'MakePlots',false,'FailOnNonconvergence',false,'IntegrationOrder',8));
m=gs_read_model(fullfile(root,'inputs'),o);o.ReferenceNodes=m.nodes;
[~,cache]=gs_panel_loads(m.nodes,m.panels,m.q1,m.q2,o);o.PanelIntegration=cache.PanelIntegration;
Lref=sqrt(sum((m.nodes(m.bars(:,2),:)-m.nodes(m.bars(:,1),:)).^2,2));
assert(size(m.nodes,1)==64&&size(m.bars,1)==112&&numel(m.panels)==49);
assert(all(m.q1==4.5)&&all(m.q2==.5)&&all(m.nodes(:,3)==0));
E=m.E*o.ElasticModulusFactor;
methods={'integrated_q4','equal_integrated_area','angle_integrated_area','equal_newell_fixed_panels','jiang_centroid_fan','jiang_weights_integrated_area'};
runs=cell(numel(methods),1);summaries=cell(size(runs));
settings=o;settings.PanelIntegration=[];
gs_write_json(fullfile(output,'settings.json'),struct('options',settings,'E_input_kN_m2',m.E, ...
    'E_effective_kN_m2',E,'A_m2',m.A,'Lref_m',Lref,'nodeIds',m.nodeIds, ...
    'memberIds',m.memberIds,'bars',m.bars,'panels',{m.panels},'fixedDofs',m.fixedDofs,'freeDofs',m.freeDofs, ...
    'referenceNodes',m.nodes,'sameCurrentEquilibriumSolverForAll',true,'legacyDefaultShareRule','equal'));
for a=1:numel(methods)
    name=methods{a};folder=fullfile(output,name);if ~isfolder(folder),mkdir(folder);end
    fprintf('\nCONTROLLED FFT RUN %d/%d: %s\n',a,numel(methods),name);
    loadFunction=@(x)lc_load(name,x,m,o);started=tic;
    s=gs_equilibrium(m.nodes,m.bars,m.freeDofs,E,m.A,Lref,loadFunction,m.nodes,o);seconds=toc(started);
    [F,info]=lc_load(name,s.nodes,m,o);res=s.internalForce-F;
    threshold=o.AbsoluteForceTolerance+o.ForceTolerance*norm(F(m.freeDofs),Inf);
    fine=o;fine.IntegrationOrder=2*o.IntegrationOrder;fine.PanelIntegration=[];
    [Ffine,~]=lc_load(name,s.nodes,m,fine);
    qdiff=norm(Ffine-F,Inf);fineResidual=norm(s.internalForce(m.freeDofs)-Ffine(m.freeDofs),Inf);
    qApplicable=ismember(name,methods([1:3,6]));
    success=s.converged&&s.exitflag>0&&norm(res(m.freeDofs),Inf)<=threshold&& ...
        (~qApplicable||(qdiff<=.1*threshold&&fineResidual<=threshold));
    summary=struct('method',name,'success',success,'reason',s.reason,'outerIterations',s.iterations, ...
        'innerExitFlag',s.exitflag,'relativeDisplacement',s.relativeDisplacement,'freeForceResidual_kN',norm(res(m.freeDofs),Inf), ...
        'forceThreshold_kN',threshold,'quadratureApplicable',qApplicable,'quadratureOrder',o.IntegrationOrder, ...
        'referenceOrder',fine.IntegrationOrder,'maximumNodalQuadratureDifference_kN',qdiff,'fineLoadResidual_kN',fineResidual, ...
        'seconds',seconds,'minimumZ_m',min(s.nodes(:,3)),'maximumZ_m',max(s.nodes(:,3)), ...
        'surfaceArea_m2',sum(info.panelAreas),'projectedArea_m2',sum(info.panelProjectedAreas), ...
        'totalVerticalForce_kN',sum(F(3:3:end)),'maximumMemberTension_kN',max(s.force), ...
        'minimumMemberForce_kN',min(s.force),'initialEmin_kN_m2',min(E),'initialEmax_kN_m2',max(E));
    r=struct('summary',summary,'state',s,'loadInfo',info);runs{a}=r;summaries{a}=summary;
    save(fullfile(folder,'result.mat'),'r','-v7');gs_write_json(fullfile(folder,'summary.json'),summary);
    gs_write_json(fullfile(folder,'history.json'),s.history);gs_write_json(fullfile(folder,'inner_history.json'),s.innerHistory);gs_write_json(fullfile(folder,'loads.json'),info);
    gs_write_json(fullfile(folder,'coordinates.json'),struct('nodeIds',m.nodeIds,'initial',m.nodes,'final',s.nodes));
    disp=s.nodes-m.nodes;R=reshape(res,3,[])';
    lc_write_csv(fullfile(folder,'nodes.csv'),{'Node_ID','X0_m','Y0_m','Z0_m','X_m','Y_m','Z_m', ...
        'UX_m','UY_m','UZ_m','Fx_kN','Fy_kN','Fz_kN','FintX_kN','FintY_kN','FintZ_kN','Rx_kN','Ry_kN','Rz_kN','FullySupported'}, ...
        [m.nodeIds,m.nodes,s.nodes,disp,reshape(F,3,[])',s.nodalInternalForce,R,double(m.fixedNodes)]);
    lc_write_csv(fullfile(folder,'members.csv'),{'Member_ID','Node1_ID','Node2_ID','E_kN_m2','A_m2','L0_m','L_m','Strain','Stress_kN_m2','Axial_kN'}, ...
        [m.memberIds,m.nodeIds(m.bars),E,m.A,Lref,s.lengths,s.strain,s.stress,s.force]);
    pc=cell2mat(m.panels);sw=cell2mat(cellfun(@(x)x(:)',info.surfaceNodalAreas,'UniformOutput',false));
    pw=cell2mat(cellfun(@(x)x(:)',info.projectedNodalAreas,'UniformOutput',false));
    lc_write_csv(fullfile(folder,'panels.csv'),{'Panel_ID','Node1','Node2','Node3','Node4','SurfaceArea_m2','ProjectedArea_m2', ...
        'SurfaceWeight1_m2','SurfaceWeight2_m2','SurfaceWeight3_m2','SurfaceWeight4_m2', ...
        'ProjectedWeight1_m2','ProjectedWeight2_m2','ProjectedWeight3_m2','ProjectedWeight4_m2'},[m.panelIds,pc,info.panelAreas,info.panelProjectedAreas,sw,pw]);
    if ~isempty(s.history),t=struct2table(s.history);lc_write_csv(fullfile(folder,'history.csv'),t.Properties.VariableNames,table2array(t));end
    fprintf('%s: PASS=%d, residual %.10g <= %.10g kN, minZ %.12g m, totalFz %.12g kN\n',name,success,summary.freeForceResidual_kN,threshold,summary.minimumZ_m,summary.totalVerticalForce_kN);
end
summaries=vertcat(summaries{:});gs_write_json(fullfile(output,'run_summary.json'),summaries);

% Coordinate differences are evaluated node by node without a fitted alignment.
pairs={};freeNodes=find(~m.fixedNodes);
for a=1:numel(methods)
    for b=a+1:numel(methods)
        d=runs{b}.state.nodes-runs{a}.state.nodes;df=d(freeNodes,:);mag=sqrt(sum(d.^2,2));
        item=struct('reference',methods{a},'comparison',methods{b},'bothConverged',summaries(a).success&&summaries(b).success, ...
            'maxAbsoluteXYZ_m',max(abs(d),[],1),'rmsXYZ_free_m',sqrt(mean(df.^2,1)), ...
            'maxNodeDistance_m',max(mag),'rmsNodeDistance_free_m',sqrt(mean(sum(df.^2,2))), ...
            'supportCoordinateDifference_m',max(abs(d(m.fixedNodes,:)),[],'all'));
        pairs{end+1}=item; %#ok<AGROW>
        lc_write_csv(fullfile(output,[methods{a} '__vs__' methods{b} '_coordinates.csv']), ...
            {'Node_ID','DX_comparison_minus_reference_m','DY_m','DZ_m','Distance_m','FullySupported'}, ...
            [m.nodeIds,d,mag,double(m.fixedNodes)]);
    end
end
gs_write_json(fullfile(output,'coordinate_comparisons.json'),[pairs{:}]);

% Apply every load method on the SAME initial and each converged geometry.
same=cell(numel(methods)+1,1);geometries=[{m.nodes};cellfun(@(x)x.state.nodes,runs,'UniformOutput',false)];
geometryNames=[{'initial_flat'};methods'];
for a=1:numel(geometries)
    X=geometries{a};[Fref,iref]=lc_load('integrated_q4',X,m,o);items=cell(numel(methods),1);
    for b=1:numel(methods)
        [F,inf]=lc_load(methods{b},X,m,o);merr=zeros(numel(m.panels),3);ferr=merr;
        for p=1:numel(m.panels)
            ids=m.panels{p};ff=inf.panelNodalForces{p};
            merr(p,:)=sum(cross(X(ids,:),ff,2),1)-iref.integratedReferencePanelMoments(p,:);
            ferr(p,:)=sum(ff,1)-iref.integratedReferencePanelResultants(p,:);
        end
        items{b}=struct('method',methods{b},'maxNodalDifferenceFromQ4_kN',norm(F-Fref,Inf), ...
            'totalVerticalForce_kN',sum(F(3:3:end)),'maxSurfaceAreaDifferenceFromQ4_m2',max(abs(inf.panelAreas-iref.panelAreas)), ...
            'sumSurfaceAreaDifferenceFromQ4_m2',sum(inf.panelAreas-iref.panelAreas), ...
            'maxProjectedAreaDifferenceFromQ4_m2',max(abs(inf.panelProjectedAreas-iref.panelProjectedAreas)), ...
            'maxPanelResultantErrorAgainstQ4_kN',max(sqrt(sum(ferr.^2,2))), ...
            'maxPanelMomentErrorAgainstQ4_kNm',max(sqrt(sum(merr.^2,2))));
        lf=fullfile(output,'same_geometry_loads');if ~isfolder(lf),mkdir(lf);end
        lc_write_csv(fullfile(lf,[geometryNames{a} '__' methods{b} '.csv']), ...
            {'Node_ID','Fx_kN','Fy_kN','Fz_kN','DifferenceXfromQ4_kN','DifferenceYfromQ4_kN','DifferenceZfromQ4_kN'}, ...
            [m.nodeIds,reshape(F,3,[])',reshape(F-Fref,3,[])']);
    end
    same{a}=struct('geometry',geometryNames{a},'comparisons',vertcat(items{:}));
end
gs_write_json(fullfile(output,'same_geometry_comparisons.json'),vertcat(same{:}));
report=struct('allRunsPass',all([summaries.success]),'runCount',numel(methods),'summaries',summaries, ...
    'comparisons',[pairs{:}],'settings',settings,'effectiveE_GPa',unique(E)/1e6, ...
    'sameSolverSameInputs',true,'sameStartingGeometry',true,'tolerancesRelaxed',false, ...
    'meaning','Controlled load-law comparison using the current equilibrium solver; not a rerun of the complete original numerical algorithm.');
gs_write_json(fullfile(output,'experiment_report.json'),report);save(fullfile(output,'all_runs.mat'),'runs','m','o','Lref','methods','report','-v7');
fprintf('\nCompleted %d matched FFT runs; all accepted = %d\n',numel(methods),report.allRunsPass);
end
