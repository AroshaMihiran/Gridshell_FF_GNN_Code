function model = gs_read_model(inputDirectory, options)
%GS_READ_MODEL Read the GUI CSV schema with explicit ID-to-row mapping.
% Global DOF numbers refer to nodal CSV row order: x1,y1,z1,x2,... .
% Panel identity is read once and retained during deformation.
n = requiredTable(inputDirectory,'nodal_coordinates.csv',{'Node_ID','X','Y','Z','Fixed_Free'});
assertIds(n.Node_ID,'node');
model.nodeIds = n.Node_ID(:);
model.nodes = [n.X,n.Y,n.Z];
validateattributes(model.nodes,{'numeric'},{'2d','ncols',3,'real','finite','nonempty'});
validateattributes(n.Fixed_Free,{'numeric','logical'},{'binary'});
nn = height(n);
e = requiredTable(inputDirectory,'edge_connectivity.csv',{'Edge_ID','Node1','Node2'});
assertIds(e.Edge_ID,'member');
model.memberIds = e.Edge_ID(:);
[ok1,b1] = ismember(e.Node1,model.nodeIds);
[ok2,b2] = ismember(e.Node2,model.nodeIds);
if any(~ok1 | ~ok2 | b1==b2)
    error('GridShell:Connectivity','Members must connect two different existing node IDs.');
end
model.bars = [b1,b2];
nm = height(e);
if nm == 0 || size(unique(sort(model.bars,2),'rows'),1) ~= nm
    error('GridShell:Connectivity','Members must be nonempty and have unique endpoint pairs.');
end
lengths = sqrt(sum((model.nodes(b2,:)-model.nodes(b1,:)).^2,2));
if any(lengths <= 1e-12), error('GridShell:Connectivity','Zero-length member.'); end
freeFile = fullfile(inputDirectory,'free_dofs.csv');
fixedFile = fullfile(inputDirectory,'fixed_dofs.csv');
if xor(isfile(freeFile),isfile(fixedFile))
    error('GridShell:DOF','Supply both free_dofs.csv and fixed_dofs.csv, or neither.');
elseif isfile(freeFile)
    f = requiredTable(inputDirectory,'free_dofs.csv',{'DOF'});
    c = requiredTable(inputDirectory,'fixed_dofs.csv',{'DOF'});
    model.freeDofs = f.DOF(:);
    fixedDofs = c.DOF(:);
    dofs = [model.freeDofs;fixedDofs];
    if numel(dofs)~=3*nn || ~isequal(sort(dofs),(1:3*nn)')
        error('GridShell:DOF','Free/fixed DOFs must partition all 3N row-based DOFs exactly once.');
    end
else
    freeNodeRows = find(n.Fixed_Free==0);
    model.freeDofs = reshape((3*freeNodeRows(:)'-2)+(0:2)',[],1);
    fixedDofs = setdiff((1:3*nn)',model.freeDofs);
end
isFixed = reshape(ismember((1:3*nn)',fixedDofs),3,nn)';
fullyFixed = all(isFixed,2);
if any(fullyFixed ~= logical(n.Fixed_Free))
    error('GridShell:DOF','Fixed_Free must be 1 exactly for fully restrained nodes.');
end
model.fixedDofs = fixedDofs;
model.fixedMembers = fullyFixed(b1) & fullyFixed(b2);
model.fixedNodes = fullyFixed;
unconnected=setdiff((1:nn)',unique(model.bars(:)));
if any(~fullyFixed(unconnected))
    error('GridShell:Connectivity','A node without any structural member must have all translations restrained. Virtual boundaries provide no stiffness.');
end
m = requiredTable(inputDirectory,'member_materials.csv',{'Member_ID','E','fcok','ftok','kmod'});
assertIds(m.Member_ID,'material member');
[found,order] = ismember(model.memberIds,m.Member_ID);
if ~all(found) || height(m)~=nm
    error('GridShell:Materials','Exactly one material row per member ID is required.');
end
for name = {'E','fcok','ftok','kmod'}
    v = m.(name{1})(order);
    validateattributes(v,{'numeric'},{'column','real','finite','positive'},mfilename,name{1});
    model.(name{1}) = v;
end
if ismember('E05',m.Properties.VariableNames)
    model.E05 = m.E05(order);
    validateattributes(model.E05,{'numeric'},{'column','real','finite','positive'});
end
if ismember('Area',m.Properties.VariableNames)
    model.A = m.Area(order);
else
    model.A = options.InitialArea*ones(nm,1);
end
validateattributes(model.A,{'numeric'},{'column','real','finite','positive'});
if any(model.A < options.MinArea | model.A > options.MaxArea)
    error('GridShell:Areas','Input areas lie outside the configured bounds.');
end
% Missing perimeter members can be closed ONLY by an explicit user outline.
% The outline is a load boundary, never an added physical structural member.
boundaryLoops={};boundaryIds=[];
if isfile(fullfile(inputDirectory,'boundary_nodes.csv'))
    bt=requiredTable(inputDirectory,'boundary_nodes.csv',{'Boundary_ID','Order','Node_ID'});
    validateattributes(bt.Boundary_ID,{'numeric'},{'column','integer','positive','finite'});
    validateattributes(bt.Order,{'numeric'},{'column','integer','positive','finite'});
    boundaryIds=unique(bt.Boundary_ID);
    for k=1:numel(boundaryIds)
        rows=find(bt.Boundary_ID==boundaryIds(k));[ord,ix]=sort(bt.Order(rows));rows=rows(ix);
        if numel(unique(ord))~=numel(ord),error('GridShell:BoundaryOrder','Boundary order must be unique within each loop.');end
        [found,indices]=ismember(bt.Node_ID(rows),model.nodeIds);
        if ~all(found),error('GridShell:BoundaryNodes','Boundary refers to an unknown node ID.');end
        boundaryLoops{k,1}=indices(:)'; %#ok<AGROW>
    end
end
[identified,topology]=gs_identify_panels(model.nodes,model.bars,boundaryLoops);
topology.boundaryIds=boundaryIds;model.panelTopology=topology;
model.boundaryLoops=topology.boundaryLoops;model.virtualBoundaryEdges=topology.virtualEdges;
if isfile(fullfile(inputDirectory,'panel_connectivity.csv'))
    pc=requiredTable(inputDirectory,'panel_connectivity.csv',{'Panel_ID','Nodes'});
    assertIds(pc.Panel_ID,'panel');model.panelIds=pc.Panel_ID(:);model.panels=cell(height(pc),1);
    matched=zeros(height(pc),1);
    for i=1:height(pc)
        txt=char(string(pc.Nodes(i)));
        if isempty(regexp(txt,'^\s*\[?\s*\d+(?:[\s,;]+\d+)*\s*\]?\s*$','once'))
            error('GridShell:Panels','Panel %g has an invalid node list.',pc.Panel_ID(i));
        end
        ids=str2double(regexp(txt,'\d+','match'));
        if numel(ids)>1&&ids(end)==ids(1),ids(end)=[];end
        [found,rows]=ismember(ids,model.nodeIds);
        if numel(rows)<3||~all(found)||numel(unique(rows))~=numel(rows)
            error('GridShell:Panels','Panel %g must have at least three distinct existing nodes.',pc.Panel_ID(i));
        end
        % Compare cyclic edge sets, not merely sorted vertex sets.
        edges=sort([rows(:),circshift(rows(:),-1)],2);edges=sortrows(edges);
        for j=1:numel(identified)
            face=identified{j}(:);fe=sortrows(sort([face,circshift(face,-1)],2));
            if isequal(edges,fe),matched(i)=j;break;end
        end
        if matched(i)==0
            error('GridShell:Panels','Panel %g is not a bounded face of physical members plus explicitly declared boundary edges.',pc.Panel_ID(i));
        end
        model.panels{i}=rows(:)';
    end
    if numel(unique(matched))~=numel(matched),error('GridShell:Panels','Duplicate panel faces supplied.');end
    model.panelTopology.inputPanelMode='explicit panel_connectivity.csv';
    model.panelTopology.selectedDiscoveredFaces=matched;
else
    model.panels=identified;model.panelIds=(1:numel(identified))';
    model.panelTopology.inputPanelMode='automatically identified from members and explicit outline';
    model.panelTopology.selectedDiscoveredFaces=(1:numel(identified))';
end
lp = requiredTable(inputDirectory,'load_patterns.csv',{'Load_ID','q1','q2'});
assertIds(lp.Load_ID,'load pattern');
if isfile(fullfile(inputDirectory,'panel_loads.csv'))
    pl=requiredTable(inputDirectory,'panel_loads.csv',{'Panel_ID','Load_Pattern'});
else
    if height(lp)~=1,error('GridShell:Loads','Discovered panels require panel_loads.csv when more than one load pattern exists.');end
    pl=table(model.panelIds,repmat(lp.Load_ID,numel(model.panels),1),'VariableNames',{'Panel_ID','Load_Pattern'});
end
assertIds(pl.Panel_ID,'loaded panel');
[found,porder] = ismember(model.panelIds,pl.Panel_ID);
if ~all(found) || height(pl)~=numel(model.panels)
    error('GridShell:Loads','Exactly one load assignment per panel ID is required.');
end
[found,lorder] = ismember(pl.Load_Pattern(porder),lp.Load_ID);
if ~all(found), error('GridShell:Loads','Panel refers to an unknown load pattern.'); end
model.q1 = lp.q1(lorder); model.q2 = lp.q2(lorder);
validateattributes(model.q1,{'numeric'},{'column','real','finite','nonnegative'});
validateattributes(model.q2,{'numeric'},{'column','real','finite','nonnegative'});
model.inputDirectory = char(inputDirectory);
end

function t = requiredTable(folder,file,columns)
path = fullfile(folder,file);
if ~isfile(path), error('GridShell:Input','Required input file is missing: %s',path); end
t = readtable(path,'Delimiter',',','TextType','string','VariableNamingRule','preserve');
if ~all(ismember(columns,t.Properties.VariableNames))
    error('GridShell:Input','%s needs columns: %s',file,strjoin(columns,', '));
end
end

function assertIds(ids,label)
validateattributes(ids,{'numeric'},{'column','integer','positive','finite'});
if numel(unique(ids))~=numel(ids)
    error('GridShell:IDs','Duplicate %s ID.',label);
end
end
