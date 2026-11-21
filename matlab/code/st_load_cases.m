function [cases,info]=st_load_cases(roof,referenceNodes,panels,qD,qW)
%ST_LOAD_CASES G-only and G+/-W STR cases, recommended first-generation basis.
% qD is SUPERIMPOSED dead pressure; timber member self-weight is added later.
% One wind variable action, leading; no accompanying variable action or psi.
o=struct('ReferenceNodes',referenceNodes,'WindDirection','downward','IntegrationOrder',8);
[g,gi]=gs_panel_loads(roof,panels,qD,zeros(size(qD)),o);
[w,wi]=gs_panel_loads(roof,panels,zeros(size(qD)),qW,o);
G=reshape(g,3,[])';W=reshape(w,3,[])';
names={'G_only','Gadverse_Wdown','Gfavourable_Wdown','Gadverse_Wup','Gfavourable_Wup'};
gammaG=[1.35,1.35,1.0,1.35,1.0];gammaW=[0,1.5,1.5,1.5,1.5];signW=[0,1,1,-1,-1];
kmod=[.6,.9,.9,.9,.9];
cases=repmat(struct('name','','gammaG',0,'gammaW',0,'windSign',0,'kmod',0, ...
    'panelDeadLoads',G,'panelWindDownLoads',W),numel(names),1);
for i=1:numel(names)
    cases(i)=struct('name',names{i},'gammaG',gammaG(i),'gammaW',gammaW(i), ...
        'windSign',signW(i),'kmod',kmod(i),'panelDeadLoads',G,'panelWindDownLoads',W); %#ok<AGROW>
end
info=struct('qD_kPa',qD,'qW_kPa',qW,'unfactoredPanelDeadNodal',G,'unfactoredPanelWindDownNodal',W, ...
    'surfaceAreas_m2',gi.panelAreas,'projectedAreas_m2',wi.panelProjectedAreas, ...
    'roofGeometryFixed',true,'panelDeadExcludesTimberSelfWeight',true, ...
    'combinationBasis','Persistent/transient STR, EN1990 6.10 recommended factors; no National Annex; wind is sole variable action', ...
    'scope','Vertical wind pressure scenarios only; not wind zoning, internal pressure or complete project action assessment');
end
