function report=test_boundary_frame_handoff()
% Empty rotations at load-only support points must not spoil frame assembly.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'code'));
tests={@physical_rotations_stay_free,@isolated_boundary_reaction, ...
    @no_fake_beams_or_stiffness,@partly_free_boundary_rejected, ...
    @multiple_boundary_nodes,@input_unchanged};
passed={};failures={};
for k=1:numel(tests)
    name=func2str(tests{k});
    try,tests{k}();passed{end+1}=name;fprintf('PASS boundary frame %s\n',name); %#ok<AGROW>
    catch ex,failures{end+1}=struct('test',name,'identifier',ex.identifier,'message',ex.message);fprintf('FAIL boundary frame %s: %s\n',name,ex.message);end %#ok<AGROW>
end
report=struct('passed',numel(passed),'failed',numel(failures),'tests',{passed},'failures',{failures});
gs_write_json(fullfile(root,'outputs','tests','boundary_frame_handoff_test_report.json'),report);
assert(isempty(failures),'frame3d:BoundaryHandoffTestFailure','Boundary frame handoff tests failed.');
end
function h=fixture()
h=struct('nodes',[0,0,0;3,0,0;3,2,-.5;0,2,0], ...
    'bars',[1,3;2,3;4,3],'A',[.01;.02;.03], ...
    'E_effective',[1e7;1.2e7;1.4e7], ...
    'translationalRestraints',logical([1,1,1;1,1,1;0,0,0;1,1,1]), ...
    'allFinalForces',[0,0,-2;0,0,-3;2,-1,-8;0,0,-4], ...
    'nodeIds',[11;22;33;44],'memberIds',[15;27;38]);
end
function h=withBoundary()
h=fixture();h.nodes(end+1,:)=[1,-2,0];h.nodeIds(end+1)=99;
h.translationalRestraints(end+1,:)=[true,true,true];h.allFinalForces(end+1,:)=[4,-7,-11];
end
function near(a,b)
assert(isequal(size(a),size(b)),'Comparison dimensions differ.');
assert(all(abs(a(:)-b(:))<=1e-9*max(1,max(abs(b(:))))),'Independent comparison failed.');
end
function physical_rotations_stay_free()
m=frame3d_from_handoff(fixture(),6e5);assert(isempty(m.loadOnlyBoundaryNodeRows)&&isempty(m.inactiveRotationDofs));
assert(~any(m.restraints(:,4:6),'all'));
h=withBoundary();m=frame3d_from_handoff(h,6e5);assert(isequal(m.loadOnlyBoundaryNodeRows,5));
assert(isequal(m.loadOnlyBoundaryNodeIds,99)&&isequal(m.inactiveRotationDofs,(28:30)'));
connectedRows=setdiff((1:size(m.nodes,1))',5);
assert(~any(m.restraints(connectedRows,4:6),'all'));
assert(isequal(m.restraints(1:5,1:3),h.translationalRestraints));
end
function isolated_boundary_reaction()
base=frame3d_from_handoff(fixture(),6e5);with=frame3d_from_handoff(withBoundary(),6e5);
a=frame3d_linear(base);b=frame3d_linear(with);assert(a.success&&b.success);
near(a.displacements(1:4,:),b.displacements(1:4,:));near(a.reactions(1:4,:),b.reactions(1:4,:));
near(a.sectionForcesI,b.sectionForcesI);near(a.sectionForcesJ,b.sectionForcesJ);
near(b.reactions(5,1:3),[-4,7,11]);near(b.reactions(5,4:6),zeros(1,3));
near(b.displacements(5,:),zeros(1,6));near(b.globalForceClosure,zeros(1,3));near(b.globalMomentClosure,zeros(1,3));
end
function no_fake_beams_or_stiffness()
h=withBoundary();m=frame3d_from_handoff(h,6e5,struct('Segments',1));s=frame3d_linear(m);assert(s.success);
assert(isequal(m.bars,h.bars)&&size(m.nodes,1)==5&&size(m.bars,1)==3);
assert(~any(m.bars(:)==5));near(full(s.K(25:30,:)),zeros(6,30));near(m.loads(5,1:3),h.allFinalForces(5,:));
end
function partly_free_boundary_rejected()
h=withBoundary();h.translationalRestraints(5,3)=false;rejected=false;
try,frame3d_from_handoff(h,6e5);catch ex,rejected=strcmp(ex.identifier,'frame3d:UnrestrainedBoundaryNode');end
assert(rejected);
end
function multiple_boundary_nodes()
h=withBoundary();h.nodes(end+1,:)=[2,-3,0];h.nodeIds(end+1)=151;
h.translationalRestraints(end+1,:)=[true,true,true];h.allFinalForces(end+1,:)=[-2,3,5];
m=frame3d_from_handoff(h,6e5,struct('Segments',2));s=frame3d_linear(m);assert(s.success);
assert(isequal(m.loadOnlyBoundaryNodeRows,[5;6])&&isequal(m.inactiveRotationDofs,[28;29;30;34;35;36]));
near(s.reactions(5:6,1:3),-h.allFinalForces(5:6,:));assert(size(m.bars,1)==6);
end
function input_unchanged()
h=withBoundary();saved=h;frame3d_from_handoff(h,6e5);assert(isequaln(h,saved));
end
