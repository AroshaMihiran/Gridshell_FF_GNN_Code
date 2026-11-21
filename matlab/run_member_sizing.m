function [report,design]=run_member_sizing(name,variant,handoffFile,overrides)
%RUN_MEMBER_SIZING Fixed-geometry C24 frame sizing, after one reflection.
% Default uses the included accepted FFT handoff. Supply run_fft(...).handoffFile
% as the third argument to size a newly computed form.
if nargin<1,name='benchmark';end
if nargin<2,variant='default';end
if nargin<4,overrides=struct();end
root=setup_paths();name=validatestring(name,{'benchmark','nonuniform'});
variant=validatestring(variant,{'default','grouped','independent','common','omit_bending'});
if nargin<3||isempty(handoffFile),handoffFile=fullfile(root,'inputs','handoffs',[name '.json']);end
h=jsondecode(fileread(handoffFile));assert(h.success,'A converged FFT handoff is required.');
if ~iscell(h.panels),h.panels=num2cell(h.panels,2);end
assert(strcmp(h.windConvention,'signed pressure; positive downward'),'Handoff qW must use the documented signed convention.');
[material,checks]=st_material();
base=frame3d_from_handoff(h,material.Gmean,struct('MirrorZ',true,'Segments',4));
base.originalE=repmat(material.Emean,numel(base.originalA),1);
base.originalG=repmat(material.Gmean,numel(base.originalA),1);
if strcmp(variant,'common')
    groups=ones(size(h.bars,1),1);symmetry=struct('mode','common_section','numberOfGroups',1,'groups',groups,'loadsAreNotSymmetrized',true);
elseif strcmp(variant,'independent')||(strcmp(name,'nonuniform')&&ismember(variant,{'default','omit_bending'}))
    groups=(1:size(h.bars,1))';symmetry=struct('mode','independent_members','numberOfGroups',numel(groups),'groups',groups,'loadsAreNotSymmetrized',true);
else
    [groups,symmetry]=rt_plan_groups(h.referenceNodes,h.bars,h.translationalRestraints);
end
roof=h.nodes;roof(:,3)=-roof(:,3);
[cases,loadInfo]=rt_signed_load_cases(roof,h.referenceNodes,h.panels,h.qD,h.qW);
loadInfo.panels=h.panels;loadInfo.panelIds=h.panelIds;loadInfo.referenceNodes=h.referenceNodes;
loadInfo.uniformC24DesignMaterial=true;loadInfo.FFTareasAreInitialSizingGuessOnly=true;
out=fullfile(root,'outputs','sizing',[name '_' variant '_' char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'))]);mkdir(out);
if strcmp(variant,'omit_bending')
    ablation=aa_iterative_size(base,material,checks,cases,groups,overrides,'omit_bending');
    design=ablation;design.final=st_design_evaluate(base,ablation.final.sides,material,checks,cases);
    design.success=ablation.success&&design.final.memberChecksPass;
    design.reason='omitted_bending_sections_freshly_reassessed';
    save(fullfile(out,'ablation_result.mat'),'ablation','-v7');
else
    design=st_iterative_size(base,material,checks,cases,groups,overrides);
end
save(fullfile(out,'design_result.mat'),'design','base','material','checks','loadInfo','symmetry','-v7');
report=st_export_design(design,base,material,checks,loadInfo,symmetry,out);
report.scenario=name;report.variant=variant;report.outputDirectory=out;
if strcmp(variant,'omit_bending')
    report.ablationSizingConverged=ablation.success;
    report.fullReassessmentFailingMembers=sum(design.final.memberRatios>1);
    assert(ablation.success,'The reduced-check sizing did not converge.');
else
    assert(design.success,'Sizing did not satisfy the unchanged acceptance criteria: %s',design.reason);
end
gs_write_json(fullfile(out,'summary.json'),report);
plot_sizing_result(h,design,fullfile(out,'sized_roof.png'));
fprintf('Sizing %s/%s: accepted=%d, Umax=%.9g, volume=%.9g m3, iterations=%d.\n', ...
    name,variant,report.success,report.maximumUtilization,report.volume_m3,report.iterations);
end
