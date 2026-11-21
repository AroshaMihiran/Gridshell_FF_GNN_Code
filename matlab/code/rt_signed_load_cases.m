function [cases,info]=rt_signed_load_cases(roof,ref,panels,qD,qW)
% qW is signed: positive downward, negative upward. Only wind is reversed.
o=struct('ReferenceNodes',ref,'WindDirection','downward','IntegrationOrder',8);
[g,gi]=gs_panel_loads(roof,panels,qD,zeros(size(qD)),o);
[wp,wi]=gs_panel_loads(roof,panels,zeros(size(qD)),max(qW,0),o);
wn=gs_panel_loads(roof,panels,zeros(size(qD)),max(-qW,0),o);
G=reshape(g,3,[])';W=reshape(wp-wn,3,[])';
names={'G_only','Gadverse_Wreference','Gfavourable_Wreference','Gadverse_Wreversed','Gfavourable_Wreversed'};
gammaG=[1.35 1.35 1 1.35 1];gammaW=[0 1.5 1.5 1.5 1.5];signW=[0 1 1 -1 -1];kmod=[.6 .9 .9 .9 .9];
cases=repmat(struct('name','','gammaG',0,'gammaW',0,'windSign',0,'kmod',0,'panelDeadLoads',G,'panelWindDownLoads',W),5,1);
for i=1:5,cases(i)=struct('name',names{i},'gammaG',gammaG(i),'gammaW',gammaW(i),'windSign',signW(i),'kmod',kmod(i),'panelDeadLoads',G,'panelWindDownLoads',W);end
info=struct('qD_kPa',qD,'qW_kPa',qW,'windConvention','qW positive downward; negative upward; reference field reversed as one variable action', ...
    'unfactoredPanelDeadNodal',G,'unfactoredPanelWindReferenceNodal',W,'unfactoredPanelWindDownNodal',W, ...
    'legacyFieldMeaning','panelWindDownLoads and unfactoredPanelWindDownNodal are retained API names containing the SIGNED REFERENCE field, not an all-downward field', ...
    'surfaceAreas_m2',gi.panelAreas,'projectedAreas_m2',wi.panelProjectedAreas,'roofGeometryFixed',true, ...
    'panelDeadExcludesTimberSelfWeight',true,'combinationBasis','EN1990 6.10 recommended factors; single permanent field and leading reference signed wind field; no National Annex');
end
