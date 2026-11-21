function model=frame3d_from_handoff(handoff,G,options)
%FRAME3D_FROM_HANDOFF Build an unstressed frame from a saved FFT/CAT state.
% model=frame3d_from_handoff(handoff,G,options)
% HANDOFF is an etabs_input JSON path or scalar structure, containing nodes,
% bars, A, E_effective, translationalRestraints and allFinalForces. Optional
% nodeIds/memberIds are preserved. Units: metres, kN, kN/m^2.
% G is REQUIRED explicitly, as a positive scalar or one value per original
% member in kN/m^2. No isotropic timber relationship is assumed from E.
% OPTIONS: MirrorZ=true, Segments=4. The construction geometry is copied from
% handoff.nodes and, by default, Z is reflected. Its current geometric lengths
% define an initially unstressed frame. Hanging strain, force, stress, Lref and
% other reference-length fields are NEVER transferred as initial conditions.
% Final global nodal force vectors are copied unchanged, including loads on
% restrained nodes. Original translation restraints are preserved. Rotations
% at physical member nodes and all subdivision DOFs are free. A fully
% translation-restrained original node with NO incident member may serve as
% a load-only boundary point: its three inactive rotations are constrained
% solely to remove empty matrix rows. They add no stiffness or frame restraint.
% Such a node with any unrestrained translation is rejected. Member joints
% are rigid.
% Solid-square properties come from frame3d_section (Saint-Venant torsion
% series). ETABS automatic J and shear areas can differ slightly; use its
% exported actual properties for exact software-to-software validation.
if nargin<2 || isempty(G)
    error('frame3d:GRequired','Supply G explicitly; no timber shear modulus is inferred from E.');
end
if nargin<3,options=struct();end
if ~isstruct(options)||~isscalar(options)
    error('frame3d:HandoffOptions','options must be a scalar structure.');
end
defaults=struct('MirrorZ',true,'Segments',4);names=fieldnames(options);
for k=1:numel(names)
    if ~isfield(defaults,names{k}),error('frame3d:HandoffOptions','Unknown option %s.',names{k});end
    defaults.(names{k})=options.(names{k});
end
options=defaults;
validateattributes(options.MirrorZ,{'logical','numeric'},{'scalar','binary'});
validateattributes(options.Segments,{'numeric'},{'real','finite','scalar','integer','positive'});
options.MirrorZ=logical(options.MirrorZ);
sourcePath='';
if ischar(handoff)||(isstring(handoff)&&isscalar(handoff))
    sourcePath=char(handoff);
    if ~isfile(sourcePath),error('frame3d:HandoffFile','Handoff file does not exist: %s',sourcePath);end
    handoff=jsondecode(fileread(sourcePath));
end
if ~isstruct(handoff)||~isscalar(handoff)
    error('frame3d:Handoff','handoff must be a scalar structure or a JSON path.');
end
required={'nodes','bars','A','E_effective','translationalRestraints','allFinalForces'};
for k=1:numel(required)
    if ~isfield(handoff,required{k}),error('frame3d:Handoff','Missing handoff.%s.',required{k});end
end
if isfield(handoff,'success'),requireTrue(handoff.success,'success');end
if isfield(handoff,'residualCheck')
    check=handoff.residualCheck;
    if ~isstruct(check)||~isscalar(check)
        error('frame3d:HandoffConvergence','residualCheck must be a scalar structure.');
    end
    flags={'passes','stateConverged'};
    for k=1:numel(flags)
        if isfield(check,flags{k}),requireTrue(check.(flags{k}),['residualCheck.' flags{k}]);end
    end
    if isfield(check,'freeForceResidual_kN')&&isfield(check,'forceTolerance_kN')
        validateattributes(check.freeForceResidual_kN,{'numeric'},{'real','finite','scalar','nonnegative'});
        validateattributes(check.forceTolerance_kN,{'numeric'},{'real','finite','scalar','positive'});
        if check.freeForceResidual_kN>check.forceTolerance_kN
            error('frame3d:HandoffConvergence','The exported free-force residual exceeds its tolerance.');
        end
    end
end
validateattributes(handoff.nodes,{'numeric'},{'2d','ncols',3,'real','finite','nonempty'});
validateattributes(handoff.bars,{'numeric'},{'2d','ncols',2,'real','integer','positive','nonempty'});
sourceNodes=double(handoff.nodes);bars=double(handoff.bars);
n0=size(sourceNodes,1);m0=size(bars,1);s=double(options.Segments);
if any(bars(:)>n0)||any(bars(:,1)==bars(:,2))
    error('frame3d:HandoffConnectivity','Member endpoints must be distinct valid node-row indices.');
end
if ~isequal(size(handoff.translationalRestraints),[n0,3]) || ...
        ~(isnumeric(handoff.translationalRestraints)||islogical(handoff.translationalRestraints)) || ...
        any(~ismember(handoff.translationalRestraints(:),[0,1]))
    error('frame3d:HandoffRestraints','Translation restraints must be an N by 3 logical/zero-one matrix.');
end
validateattributes(handoff.allFinalForces,{'numeric'},{'size',[n0,3],'real','finite'});
loadOnlyBoundaryNodeRows=setdiff((1:n0)',unique(bars(:)));
if any(~all(logical(handoff.translationalRestraints(loadOnlyBoundaryNodeRows,:)),2))
    error('frame3d:UnrestrainedBoundaryNode', ...
        'An original node without an incident frame member must have all translations restrained. Virtual load boundaries provide no stiffness.');
end
A=memberVector(handoff.A,m0,'A');E=memberVector(handoff.E_effective,m0,'E_effective');
G=memberVector(G,m0,'G');section=frame3d_section(sqrt(A),sqrt(A));
nodeIds=ids(handoff,'nodeIds',n0);memberIds=ids(handoff,'memberIds',m0);
originalNodes=sourceNodes;
if options.MirrorZ,originalNodes(:,3)=-originalNodes(:,3);end
lengths=sqrt(sum((originalNodes(bars(:,2),:)-originalNodes(bars(:,1),:)).^2,2));
if any(lengths<=100*eps(max(1,max(max(originalNodes)-min(originalNodes)))))
    error('frame3d:HandoffConnectivity','A member has zero or numerically negligible construction length.');
end
nn=n0+m0*(s-1);nm=m0*s;
nodes=zeros(nn,3);nodes(1:n0,:)=originalNodes;
restraints=false(nn,6);restraints(1:n0,1:3)=logical(handoff.translationalRestraints);
restraints(loadOnlyBoundaryNodeRows,4:6)=true;
inactiveRotationDofs=reshape((6*(loadOnlyBoundaryNodeRows(:)'-1))+(4:6)',[],1);
loads=zeros(nn,6);loads(1:n0,1:3)=double(handoff.allFinalForces);
newBars=zeros(nm,2);pointNames=cell(nn,1);frameNames=cell(nm,1);
frameMemberIds=zeros(nm,1);frameMemberRows=zeros(nm,1);frameSegments=zeros(nm,1);
for n=1:n0,pointNames{n}=sprintf('N%03d',nodeIds(n));end
nextNode=n0;
for m=1:m0
    start=bars(m,1);finish=bars(m,2);previous=start;
    for j=1:s
        if j==s
            following=finish;
        else
            nextNode=nextNode+1;following=nextNode;
            nodes(following,:)=originalNodes(start,:)+(j/s)*(originalNodes(finish,:)-originalNodes(start,:));
            pointNames{following}=sprintf('B%dP%d',memberIds(m),j);
        end
        f=(m-1)*s+j;newBars(f,:)=[previous,following];
        frameNames{f}=sprintf('M%03dS%d',memberIds(m),j);
        frameMemberIds(f)=memberIds(m);frameMemberRows(f)=m;frameSegments(f)=j;
        previous=following;
    end
end
model=struct('nodes',nodes,'bars',newBars,'E',E(frameMemberRows), ...
    'G',G(frameMemberRows),'restraints',restraints,'loads',loads);
properties={'A','I2','I3','J','As2','As3'};
for k=1:numel(properties)
    name=properties{k};model.(name)=section.(name)(frameMemberRows);
end
model.pointNames=pointNames;model.frameNames=frameNames;
model.frameMemberIds=frameMemberIds;model.frameMemberRows=frameMemberRows;
model.frameSegments=frameSegments;model.nodeIds=nodeIds;model.memberIds=memberIds;
model.originalNodeRows=(1:n0)';model.originalNodes=originalNodes;model.originalBars=bars;
model.sourceNodes=sourceNodes;model.originalMemberLengths=lengths;
model.originalA=A;model.originalE=E;model.originalG=G;model.options=options;
model.sourcePath=sourcePath;model.initialStressTransferred=false;
model.constructionReference='Copied chosen geometry, optionally reflected Z, initially unstressed';
model.loadConvention='Final global nodal forces copied unchanged; subdivision nodes have zero applied loads';
model.loadOnlyBoundaryNodeRows=loadOnlyBoundaryNodeRows;
model.loadOnlyBoundaryNodeIds=nodeIds(loadOnlyBoundaryNodeRows);
model.inactiveRotationDofs=inactiveRotationDofs;
model.supportConvention=['Source translations preserved; all physical frame-node rotations and all subdivision DOFs free. ' ...
    'Only empty rotational DOFs at fully translation-restrained nodes with no incident member are constrained; ' ...
    'these load-only boundary points retain their loads and reactions and add no member or structural stiffness.'];
model.sectionConvention=section.torsionModel;
model.scope='Linear frame input only; this helper does not perform section sizing or global stability verification';
end

function v=memberVector(v,n,name)
validateattributes(v,{'numeric'},{'vector','real','finite','positive','nonempty'},mfilename,name);
v=double(v(:));if isscalar(v),v=repmat(v,n,1);end
if numel(v)~=n,error('frame3d:HandoffProperty','%s needs one value per original member or a scalar.',name);end
end

function v=ids(handoff,name,n)
if isfield(handoff,name),v=handoff.(name);else,v=(1:n)';end
validateattributes(v,{'numeric'},{'vector','real','finite','positive','integer','numel',n},mfilename,name);
v=double(v(:));if numel(unique(v))~=n,error('frame3d:HandoffIds','%s must be unique.',name);end
end

function requireTrue(v,name)
if ~isscalar(v)||~(islogical(v)||isnumeric(v))||~isequal(double(v),1)
    error('frame3d:HandoffConvergence','Handoff %s must report successful convergence.',name);
end
end
