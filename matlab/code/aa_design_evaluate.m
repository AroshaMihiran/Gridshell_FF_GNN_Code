function ev=aa_design_evaluate(base,sides,material,designOptions,cases,checkMode)
%AA_DESIGN_EVALUATE Full unchanged frame analysis; explicit check dispatch.
if nargin<6,checkMode='full';end
checkMode=validatestring(checkMode,{'full','omit_bending'});
sides=sides(:);nm=numel(base.originalMemberLengths);rows=base.frameMemberRows(:);ne=numel(rows);nc=numel(cases);
assert(numel(sides)==nm&&all(sides>0));section=frame3d_section(sides,sides);model=base;
for f={'A','I2','I3','J','As2','As3'},name=f{1};model.(name)=section.(name)(rows);end
model.E=repmat(material.Emean,ne,1);model.G=repmat(material.Gmean,ne,1);
nn=size(model.nodes,1);n0=numel(base.originalNodeRows);nodal=zeros(nn,6,nc);line=zeros(ne,3,nc);
for c=1:nc
    nodal(1:n0,1:3,c)=cases(c).gammaG*cases(c).panelDeadLoads+cases(c).gammaW*cases(c).windSign*cases(c).panelWindDownLoads;
    line(:,3,c)=-cases(c).gammaG*material.unitWeight*model.A;
end
batch=st_frame_batch(model,nodal,line,{cases.name});
actions=zeros(2*ne*nc,6);stationRows=repmat([rows;rows],nc,1);stationKmod=zeros(2*ne*nc,1);
caseRatios=zeros(nm,nc);checksByCase=cell(nc,1);allChecks=cell(nc,1);maxDisplacements=zeros(nc,1);maxMoments=zeros(nc,1);
memberNmin=Inf(nm,nc);memberNmax=-Inf(nm,nc);memberM2=zeros(nm,nc);memberM3=zeros(nm,nc);
for c=1:nc
    index=(c-1)*2*ne+(1:2*ne);actions(index,:)=batch.cases{c}.envelopeActions;stationKmod(index)=cases(c).kmod;
    s=frame3d_section(sides([rows;rows]),sides([rows;rows]));mat=material;mat.kmod=cases(c).kmod;
    ch=aa_resistance_checks(actions(index,:),s,base.originalMemberLengths([rows;rows]),mat,designOptions,checkMode);
    caseRatios(:,c)=accumarray([rows;rows],ch.ratio,[nm,1],@max);checksByCase{c}=ch;allChecks{c}=ch;
    maxDisplacements(c)=batch.cases{c}.maximumAbsUZ;maxMoments(c)=max(hypot(actions(index,5),actions(index,6)));
    for e=1:nm
        ix=find([rows;rows]==e);a=actions(index(ix),:);
        memberNmin(e,c)=min(a(:,1));memberNmax(e,c)=max(a(:,1));memberM2(e,c)=max(abs(a(:,5)));memberM3(e,c)=max(abs(a(:,6)));
    end
end
[ratios,governingCase]=max(caseRatios,[],2);modes=cell(nm,1);
for e=1:nm
    c=governingCase(e);ch=checksByCase{c};ix=find([rows;rows]==e);[~,j]=max(ch.ratio(ix));modes{e}=ch.governingMode{ix(j)};
end
ev=struct('resistanceCheckMode',checkMode,'success',batch.success,'model',model,'sides',sides,'section',section,'cases',{batch.cases}, ...
    'caseDefinitions',cases,'caseMemberRatios',caseRatios,'memberRatios',ratios,'maxRatio',max(ratios), ...
    'memberChecksPass',all(ratios<=1),'governingCase',governingCase,'governingMode',{modes}, ...
    'stationActions',actions,'stationMemberRows',stationRows,'stationKmod',stationKmod,'checksByCase',{allChecks}, ...
    'memberNmin',memberNmin,'memberNmax',memberNmax,'memberM2',memberM2,'memberM3',memberM3, ...
    'volume',sum(section.A.*base.originalMemberLengths),'unfactoredTimberSelfWeight_kN',material.unitWeight*sum(section.A.*base.originalMemberLengths), ...
    'maxAbsUZ',max(maxDisplacements),'caseMaximumAbsUZ',maxDisplacements,'maxMoment',max(maxMoments), ...
    'allPhysicalMembersRetained',numel(sides)==size(base.originalBars,1),'initialStressTransferred',false, ...
    'fullDesignVerified',false,'scope','Signed normal-stress and assumed-effective-length member column checks; no global/second-order stability, LTB, shear/torsion, joints or SLS design pass');
end
