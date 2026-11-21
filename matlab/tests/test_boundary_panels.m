function report=test_boundary_panels()
% Geometric/topological checks independent of load integration and stiffness.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'code'));
tests={@open_grid_closure,@long_boundary_split,@bar_order_and_orientation, ...
    @concave_outline,@explicit_hole,@hole_not_filled,@crossing_members, ...
    @missing_outline,@nonplanar_reference,@self_crossing_outline, ...
    @member_outside,@unknown_boundary_node,@physical_through_node, ...
    @reader_nonconsecutive_ids,@reader_explicit_virtual_panel,@reader_disallowed_closure, ...
    @reader_single_pattern_default,@reader_missing_load_assignment,@reader_unconnected_free_node, ...
    @reader_explicit_panel_ids,@reader_negative_wind,@reader_fixed_boundary_only_nodes};
passed={};failures={};
for k=1:numel(tests)
    name=func2str(tests{k});
    try,tests{k}();passed{end+1}=name;fprintf('PASS boundary %s\n',name); %#ok<AGROW>
    catch ex,failures{end+1}=struct('test',name,'identifier',ex.identifier,'message',ex.message);fprintf('FAIL boundary %s: %s\n',name,ex.message);end %#ok<AGROW>
end
report=struct('passed',numel(passed),'failed',numel(failures),'tests',{passed},'failures',{failures});
gs_write_json(fullfile(root,'outputs','tests','boundary_panels_test_report.json'),report);
assert(isempty(failures),'GridShell:BoundaryTestFailure','Boundary tests failed.');
options=gs_options('',struct('InitialArea',.005,'MinArea',1e-6,'MaxArea',Inf));
model=gs_read_model(fullfile(root,'inputs','open_boundary_demo'),options);
audit=model.panelTopology;audit.nodeIds=model.nodeIds;audit.panels=model.panels;
audit.actualBars=model.bars;audit.panelIds=model.panelIds;
audit.virtualEdgeNodeIds=model.nodeIds(model.virtualBoundaryEdges);
gs_write_json(fullfile(root,'outputs','tests','boundary_demo_audit.json'),audit);
ids=model.panelIds;txt=strings(numel(ids),1);
for k=1:numel(ids),txt(k)="["+strjoin(string(model.nodeIds(model.panels{k}))'," ")+"]";end
writetable(table(ids,txt,'VariableNames',{'Panel_ID','Nodes'}),fullfile(root,'outputs','tests','boundary_demo_panel_connectivity.csv'));
end
function [x,b,outline]=grid(n)
[xx,yy]=meshgrid(0:n-1);x=[reshape(xx',[],1),reshape(yy',[],1),zeros(n*n,1)];b=[];
for row=1:n
    for col=1:n-1,b(end+1,:)=[(row-1)*n+col,(row-1)*n+col+1];end %#ok<AGROW>
end
for row=1:n-1
    for col=1:n,b(end+1,:)=[(row-1)*n+col,row*n+col];end %#ok<AGROW>
end
outline=[1,n,n*n,n*(n-1)+1];
end
function near(a,b),assert(max(abs(a(:)-b(:)))<1e-8);end
function mustFail(fun)
failed=false;try,fun();catch ex,failed=startsWith(ex.identifier,'GridShell:');end
assert(failed,'Expected explicit geometry/topology/input rejection.');
end
function [x,b,o]=openGrid()
[x,b,o]=grid(3);b(ismember(sort(b,2),[1,2;8,9],'rows'),:)=[];
end
function open_grid_closure()
[x,b,o]=openGrid();before=b;[p,i]=gs_identify_panels(x,b,{o});
assert(numel(p)==4&&i.virtualEdgeCount==2&&isequal(b,before));near(sum(i.panelAreas),4);
assert(isequal(sortrows(i.virtualEdges),[1,2;8,9]));
end
function long_boundary_split()
[x,b,o]=openGrid();[~,i]=gs_identify_panels(x,b,{o});assert(numel(i.boundaryLoops{1})==8);
assert(all(ismember([2,4,6,8],i.boundaryLoops{1})));
end
function bar_order_and_orientation()
[x,b,o]=openGrid();p=gs_identify_panels(x,b,{o});q=gs_identify_panels(x,fliplr(flipud(b)),{fliplr(o)});assert(isequal(p,q));
end
function concave_outline()
x=[0,0,0;1,0,0;2,0,0;0,1,0;1,1,0;2,1,0;0,2,0;1,2,0];
b=[1,2;2,3;4,5;5,6;7,8;1,4;4,7;2,5;5,8;3,6];
b(1,:)=[];[p,i]=gs_identify_panels(x,b,{[1,3,6,5,8,7]});assert(numel(p)==3);near(i.totalPanelArea,3);
end
function explicit_hole()
[x,b,o]=grid(4);[p,i]=gs_identify_panels(x,b,{[6,7,11,10],o});
assert(numel(p)==8&&i.outerBoundaryIndex==2);near(i.totalPanelArea,8);
assert(~any(cellfun(@(v)isequal(sort(v),[6,7,10,11]),p)));
end
function hole_not_filled()
x=[0,0,0;4,0,0;4,4,0;0,4,0;1,1,0;3,1,0;3,3,0;1,3,0];
b=[1,2;2,3;3,4;4,1;5,6;6,7;7,8;8,5];
mustFail(@()gs_identify_panels(x,b,{[1,2,3,4],[5,6,7,8]}));
end
function crossing_members()
x=[0,0,0;1,0,0;1,1,0;0,1,0];b=[1,2;2,3;3,4;4,1;1,3;2,4];mustFail(@()gs_identify_panels(x,b,{1:4}));
end
function missing_outline()
[x,~,~]=grid(3);mustFail(@()gs_identify_panels(x,[2,5;4,5;5,6;5,8],{}));
end
function nonplanar_reference()
[x,b,o]=openGrid();x(5,3)=.1;mustFail(@()gs_identify_panels(x,b,{o}));
end
function self_crossing_outline()
x=[0,0,0;1,0,0;1,1,0;0,1,0];mustFail(@()gs_identify_panels(x,[1,2;3,4],{[1,3,2,4]}));
end
function member_outside()
[x,b,o]=openGrid();x(end+1,:)=[3,1,0];b(end+1,:)=[6,10];mustFail(@()gs_identify_panels(x,b,{o}));
end
function unknown_boundary_node()
[x,b,o]=openGrid();mustFail(@()gs_identify_panels(x,b,{[o,99]}));
end
function physical_through_node()
[x,b,o]=grid(3);b=b(~ismember(sort(b,2),[1,2;2,3],'rows'),:);b(end+1,:)=[1,3];mustFail(@()gs_identify_panels(x,b,{o}));
end
function [folder,options]=fixture()
root=fileparts(fileparts(mfilename('fullpath')));
folder=tempname(fullfile(root,'outputs','tests'));mkdir(folder);
copyfile(fullfile(root,'inputs','open_boundary_demo','*'),folder);
options=gs_options('',struct('InitialArea',.005,'MinArea',1e-6,'MaxArea',Inf));
end
function reader_nonconsecutive_ids()
[f,o]=fixture();m=gs_read_model(f,o);
assert(numel(m.panels)==4&&size(m.bars,1)==10&&size(m.virtualBoundaryEdges,1)==2);
assert(isequal(m.nodeIds,[101;205;309;413;517;621;725;829;933]));
assert(isequal(m.panels{1},[1,2,5,4]));
end
function reader_explicit_virtual_panel()
[f,o]=fixture();m=gs_read_model(f,o);
ids=(1:4)';txt=strings(4,1);
for k=1:4,txt(k)="["+strjoin(string(m.nodeIds(m.panels{k}))'," ")+"]";end
writetable(table(ids,txt,'VariableNames',{'Panel_ID','Nodes'}),fullfile(f,'panel_connectivity.csv'));
n=gs_read_model(f,o);assert(isequal(n.panels,m.panels)&&size(n.bars,1)==10);
end
function reader_disallowed_closure()
[f,o]=fixture();writetable(table(1,"[101 309 933 725]",'VariableNames',{'Panel_ID','Nodes'}),fullfile(f,'panel_connectivity.csv'));
mustFail(@()gs_read_model(f,o));
end
function reader_single_pattern_default()
[f,o]=fixture();m=gs_read_model(f,o);near(m.q1,ones(4,1));near(m.q2,.5*ones(4,1));
end
function reader_missing_load_assignment()
[f,o]=fixture();writetable(table([1;2],[1;2],[.5;.5],'VariableNames',{'Load_ID','q1','q2'}),fullfile(f,'load_patterns.csv'));
mustFail(@()gs_read_model(f,o));
end
function reader_unconnected_free_node()
[f,o]=fixture();t=readtable(fullfile(f,'edge_connectivity.csv'));t(ismember(t.Node1,101)|ismember(t.Node2,101),:)=[];writetable(t,fullfile(f,'edge_connectivity.csv'));
n=readtable(fullfile(f,'nodal_coordinates.csv'));n.Fixed_Free(1)=0;writetable(n,fullfile(f,'nodal_coordinates.csv'));mustFail(@()gs_read_model(f,o));
end
function reader_explicit_panel_ids()
[f,o]=fixture();m=gs_read_model(f,o);ids=[90;12;56;7];txt=strings(4,1);
for k=1:4,txt(k)="["+strjoin(string(m.nodeIds(m.panels{k}))'," ")+"]";end
writetable(table(ids,txt,'VariableNames',{'Panel_ID','Nodes'}),fullfile(f,'panel_connectivity.csv'));
writetable(table([1;2],[1;2],[.5;.5],'VariableNames',{'Load_ID','q1','q2'}),fullfile(f,'load_patterns.csv'));
writetable(table(flipud(ids),[1;2;1;2],'VariableNames',{'Panel_ID','Load_Pattern'}),fullfile(f,'panel_loads.csv'));
n=gs_read_model(f,o);assert(isequal(n.panelIds,ids)&&isequal(n.panels,m.panels));near(n.q1,[2;1;2;1]);
end
function reader_negative_wind()
[f,o]=fixture();writetable(table(1,1,-.5,'VariableNames',{'Load_ID','q1','q2'}),fullfile(f,'load_patterns.csv'));
rejected=false;try,gs_read_model(f,o);catch,rejected=true;end
assert(rejected,'Wind magnitude must not be negative: select upward/downward in options.');
end
function reader_fixed_boundary_only_nodes()
[f,o]=fixture();ids=[101;205;309;413;517;621;725;829;933];bars=[2,5;4,5;5,6;5,8];
writetable(table((1:4)',ids(bars(:,1)),ids(bars(:,2)),'VariableNames',{'Edge_ID','Node1','Node2'}),fullfile(f,'edge_connectivity.csv'));
t=readtable(fullfile(f,'member_materials.csv'));t=t(1:4,:);writetable(t,fullfile(f,'member_materials.csv'));
m=gs_read_model(f,o);assert(numel(m.panels)==4&&size(m.bars,1)==4&&size(m.virtualBoundaryEdges,1)==8);
end
