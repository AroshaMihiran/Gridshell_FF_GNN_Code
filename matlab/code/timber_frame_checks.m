function checks=timber_frame_checks(actions,section,lengths,material,options)
%TIMBER_FRAME_CHECKS Preliminary EC5 normal-stress/member-buckling checks.
% actions: N-by-6 [N,V2,V3,T,M2,M3], kN and kN m; N>0 is tension.
% section: A,I2,I3,W2,W3 in m units; scalar or one entry per action row.
% lengths: FULL physical member lengths, never numerical element lengths.
% material: explicit fc0k,ft0k,fmk,E05 [kN/m^2], kmod,gammaM.
% The same supplied bending strength is used about each section axis. Input
% strengths must already correspond to the product/section being checked;
% no size, system, volume or other favourable strength factors are inferred.
% options: betaC=.2 (solid timber), km=.7 (rectangle),
% EffectiveLengthFactor2=1, EffectiveLengthFactor3=1; each scalar/N-vector.
%
% First-generation EN1995-1-1: 6.2.3 (6.17/18), 6.2.4 (6.19/20),
% 6.3.2 (6.21-28). For compression with either lambdaRel>0.3, BOTH
% column-buckling interactions apply; each axis uses its own kc.
% The squared compression term is used only for cross-section interaction.
% Select the resistance branch from the CURRENT signed frame axial force.
% Reflection of geometry does not itself change this sign. Tensile members
% use ft,design and the linear tension+bending interactions; kc is never
% applied to a tensile force. Positive N means tension at either station.
% References checked:
% https://eurocodes.jrc.ec.europa.eu/sites/default/files/2022-06/EN1995_2_Winter.pdf
% (slides81-85, buckling and biaxial column interaction)
% https://www.swedishwood.com/siteassets/5-publikationer/pdfer/sw-design-of-timber-structures-vol2-2022.pdf
% (printed pages23,26-27)
%
% passes/ratio refer ONLY to the normal-stress checks implemented here.
% Shear/torsion resistance, lateral-torsional buckling, global stability,
% serviceability, connections, net-section effects, curved/tapered members,
% creep and load-combination selection are NOT verified by this function.
if nargin<5,options=struct();end
validateattributes(actions,{'numeric'},{'real','finite','2d','ncols',6,'nonempty'},mfilename,'actions');
actions=double(actions);n=size(actions,1);
if ~isstruct(section)||~isscalar(section)||~isstruct(material)||~isscalar(material)
    error('timberFrame:Input','section and material must be scalar structures.');
end
if ~isstruct(options)||~isscalar(options)
    error('timberFrame:Options','options must be a scalar structure.');
end
defaults=struct('betaC',.2,'km',.7,'EffectiveLengthFactor2',1,'EffectiveLengthFactor3',1);
names=fieldnames(options);
for k=1:numel(names)
    if ~isfield(defaults,names{k}),error('timberFrame:Options','Unknown option %s.',names{k});end
    defaults.(names{k})=options.(names{k});
end
options=defaults;s=struct();m=struct();o=struct();
names={'A','I2','I3','W2','W3'};
for k=1:numel(names)
    name=names{k};if ~isfield(section,name),error('timberFrame:Section','Missing section.%s.',name);end
    s.(name)=memberVector(section.(name),n,['section.' name],true);
end
names={'fc0k','ft0k','fmk','E05','kmod','gammaM'};
for k=1:numel(names)
    name=names{k};if ~isfield(material,name),error('timberFrame:Material','Missing material.%s.',name);end
    m.(name)=memberVector(material.(name),n,['material.' name],true);
end
L=memberVector(lengths,n,'lengths',true);names=fieldnames(options);
for k=1:numel(names)
    name=names{k};o.(name)=memberVector(options.(name),n,['options.' name],~strcmp(name,'betaC'));
end
if any(o.betaC<0)||any(o.km>1)
    error('timberFrame:Options','betaC must be nonnegative and km must be in (0,1].');
end
radius2=sqrt(s.I2./s.A);radius3=sqrt(s.I3./s.A);
Le2=L.*o.EffectiveLengthFactor2;Le3=L.*o.EffectiveLengthFactor3;
lambda2=Le2./radius2/pi.*sqrt(m.fc0k./m.E05);
lambda3=Le3./radius3/pi.*sqrt(m.fc0k./m.E05);
if any(~isfinite([lambda2;lambda3]))
    error('timberFrame:Overflow','Nonfinite slenderness; check input units/ranges.');
end
kc2=bucklingReduction(lambda2,o.betaC);kc3=bucklingReduction(lambda3,o.betaC);
fc=m.kmod.*m.fc0k./m.gammaM;ft=m.kmod.*m.ft0k./m.gammaM;fm=m.kmod.*m.fmk./m.gammaM;
if any(~isfinite([fc;ft;fm])|[fc;ft;fm]<=0)
    error('timberFrame:Overflow','Invalid design strengths; check input units/ranges.');
end
sigmaC=max(-actions(:,1),0)./s.A;sigmaT=max(actions(:,1),0)./s.A;
sigmaM2=abs(actions(:,5))./s.W2;sigmaM3=abs(actions(:,6))./s.W3;
c=sigmaC./fc;t=sigmaT./ft;m2=sigmaM2./fm;m3=sigmaM3./fm;
cross2=c.^2+t+m2+o.km.*m3;cross3=c.^2+t+o.km.*m2+m3;
compression=actions(:,1)<0;
tension=actions(:,1)>0;
zeroAxial=actions(:,1)==0;
% These branch-specific fields make the selected equations auditable. The
% historical cross2/cross3 fields above are retained for compatibility.
tensionBending2=zeros(n,1);tensionBending3=zeros(n,1);
tensionBending2(tension)=t(tension)+m2(tension)+o.km(tension).*m3(tension);
tensionBending3(tension)=t(tension)+o.km(tension).*m2(tension)+m3(tension);
compressionBending2=zeros(n,1);compressionBending3=zeros(n,1);
compressionBending2(compression)=c(compression).^2+m2(compression)+o.km(compression).*m3(compression);
compressionBending3(compression)=c(compression).^2+o.km(compression).*m2(compression)+m3(compression);
axialRegime=repmat({'zero_axial'},n,1);
axialRegime(tension)={'tension'};axialRegime(compression)={'compression'};
columnActive=compression & (lambda2>.3|lambda3>.3);
buck2=zeros(n,1);buck3=zeros(n,1);
buck2(columnActive)=c(columnActive)./kc2(columnActive)+m2(columnActive)+o.km(columnActive).*m3(columnActive);
buck3(columnActive)=c(columnActive)./kc3(columnActive)+o.km(columnActive).*m2(columnActive)+m3(columnActive);
% Including pure axial strength makes the reported utilization intuitive for
% stocky pure compression; it does not change any acceptance boundary.
allRatios=[c,t,cross2,cross3,buck2,buck3];
if any(~isfinite(allRatios(:)))
    error('timberFrame:Overflow','Nonfinite utilization; check input units/ranges.');
end
[ratio,which]=max(allRatios,[],2);
labels={'compression_strength','tension_strength','section_interaction_axis2', ...
    'section_interaction_axis3','column_buckling_interaction_axis2','column_buckling_interaction_axis3'};
mode=reshape(labels(which),[],1);
mode(which==3&tension)={'tension_bending_axis2'};
mode(which==4&tension)={'tension_bending_axis3'};
mode(which==3&compression)={'compression_bending_axis2'};
mode(which==4&compression)={'compression_bending_axis3'};
mode(which==3&zeroAxial)={'bending_axis2'};
mode(which==4&zeroAxial)={'bending_axis3'};
mode(all(actions==0,2))={'zero_demand'};
checks=struct('ratio',ratio,'passes',ratio<=1,'passesNormalStressChecks',ratio<=1, ...
    'governingMode',{mode},'utilizationCompression',c,'utilizationTension',t, ...
    'utilizationBending2',m2,'utilizationBending3',m3, ...
    'utilizationCrossSection2',cross2,'utilizationCrossSection3',cross3, ...
    'utilizationTensionBending2',tensionBending2,'utilizationTensionBending3',tensionBending3, ...
    'utilizationCompressionBending2',compressionBending2,'utilizationCompressionBending3',compressionBending3, ...
    'utilizationBuckling2',buck2,'utilizationBuckling3',buck3, ...
    'lambdaRel2',lambda2,'lambdaRel3',lambda3,'kc2',kc2,'kc3',kc3, ...
    'radiusOfGyration2',radius2,'radiusOfGyration3',radius3, ...
    'effectiveLength2',Le2,'effectiveLength3',Le3,'physicalLength',L, ...
    'fcDesign',fc,'ftDesign',ft,'fmDesign',fm,'E05',m.E05, ...
    'sigmaCompression',sigmaC,'sigmaTension',sigmaT,'sigmaBending2',sigmaM2,'sigmaBending3',sigmaM3, ...
    'isCompression',compression,'isTension',tension,'isZeroAxial',zeroAxial, ...
    'axialRegime',{axialRegime},'columnBucklingActive',columnActive, ...
    'axialForceSignConvention','N>0 tension; N<0 compression; from current frame analysis', ...
    'tensionUsesBucklingReduction',false, ...
    'hasShearDemand',any(actions(:,2:3)~=0,2),'hasTorsionDemand',actions(:,4)~=0, ...
    'shearResistanceVerified',false,'torsionResistanceVerified',false, ...
    'lateralTorsionalBucklingVerified',false,'globalStabilityVerified',false, ...
    'fullDesignVerified',false,'options',o, ...
    'scope','Normal-stress and assumed-effective-length column checks only; not full structural design verification');
end

function value=memberVector(value,n,name,positive)
validateattributes(value,{'numeric'},{'real','finite','vector','nonempty'},mfilename,name);
value=double(value(:));if isscalar(value),value=repmat(value,n,1);end
if numel(value)~=n,error('timberFrame:Size','%s must be scalar or have one entry per action row.',name);end
if positive&&any(value<=0),error('timberFrame:Positive','%s must be positive.',name);end
end

function kc=bucklingReduction(lambda,beta)
kc=ones(size(lambda));idx=lambda>.3;l=lambda(idx);
k=.5*(1+beta(idx).*(l-.3)+l.^2);disc=k.^2-l.^2;
if any(~isfinite([k;disc]))
    error('timberFrame:Overflow','Buckling expression overflow; check units/ranges.');
end
roundoff=100*eps(max(1,max([k.^2;1])));
if any(disc < -roundoff),error('timberFrame:Buckling','Invalid buckling discriminant.');end
kc(idx)=min(1,1./(k+sqrt(max(disc,0))));
if any(~isfinite(kc)|kc<=0),error('timberFrame:Buckling','Invalid buckling reduction factor.');end
end
