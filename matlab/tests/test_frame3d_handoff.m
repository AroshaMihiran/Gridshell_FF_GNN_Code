function report=test_frame3d_handoff()
%TEST_FRAME3D_HANDOFF Verify reference, loads, properties and subdivision rules.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'code'));
tests={@default_geometry_and_topology,@load_and_moment_conservation, ...
    @supports_and_rotations,@member_properties,@original_input_unchanged, ...
    @prestress_and_reference_ignored,@unmirrored_single_segment, ...
    @json_file_input,@missing_G_rejected,@unsuccessful_source_rejected, ...
    @failed_residual_rejected,@numeric_residual_rejected};
passed={};failures={};
for k=1:numel(tests)
    name=func2str(tests{k});
    try,tests{k}();passed{end+1}=name;fprintf('PASS %s\n',name); %#ok<AGROW>
    catch ex,failures{end+1}=struct('test',name,'message',ex.message);fprintf('FAIL %s: %s\n',name,ex.message);end %#ok<AGROW>
end
report=struct('passed',numel(passed),'failed',numel(failures),'tests',{passed},'failures',{failures});
fid=fopen(fullfile(root,'outputs','tests','frame3d_handoff_test_report.json'),'w');
if fid>=0,fprintf(fid,'%s',jsonencode(report,'PrettyPrint',true));fclose(fid);end
fprintf('FRAME3D HANDOFF TESTS: %d passed; %d failed.\n',report.passed,report.failed);
assert(isempty(failures),'frame3d:HandoffTestFailure','Some handoff tests failed.');
end

function h=fixture()
h=struct('nodes',[0,0,0;3,0,0;3,2,-.5;0,2,0], ...
    'bars',[1,3;2,3;4,3],'A',[.01;.02;.03], ...
    'E_effective',[1e7;1.2e7;1.4e7], ...
    'translationalRestraints',logical([1,1,1;1,1,1;0,0,0;1,1,1]), ...
    'allFinalForces',[0,0,-2;0,0,-3;2,-1,-8;0,0,-4], ...
    'nodeIds',[11;22;33;44],'memberIds',[15;27;38], ...
    'success',true,'residualCheck',struct('passes',true,'stateConverged',true, ...
    'freeForceResidual_kN',1e-8,'forceTolerance_kN',1e-5), ...
    'Lref',[4;2;3],'signedAxialForce',[10;20;30],'signedAxialStress',[1e3;1e3;1e3]);
end

function close(a,b,tol)
if nargin<3,tol=1e-12;end
assert(isequal(size(a),size(b)),'frame3d:TestShape','Comparison shape mismatch.');
assert(all(abs(a(:)-b(:))<=tol*max(1,max(abs(b(:))))),'frame3d:TestValue','Value mismatch.');
end

function default_geometry_and_topology()
h=fixture();m=frame3d_from_handoff(h,6e5);assert(size(m.nodes,1)==13&&size(m.bars,1)==12);
expected=h.nodes;expected(:,3)=-expected(:,3);close(m.originalNodes,expected);close(m.nodes(1:4,:),expected);
assert(isequal(m.originalBars,h.bars));assert(all(m.frameSegments==repmat((1:4)',3,1)));
for i=1:3
    rows=find(m.frameMemberRows==i);assert(numel(rows)==4);
    assert(m.bars(rows(1),1)==h.bars(i,1)&&m.bars(rows(end),2)==h.bars(i,2));
    lengths=sqrt(sum((m.nodes(m.bars(rows,2),:)-m.nodes(m.bars(rows,1),:)).^2,2));
    close(sum(lengths),m.originalMemberLengths(i));
end
assert(strcmp(m.pointNames{1},'N011')&&strcmp(m.frameNames{1},'M015S1'));
end

function load_and_moment_conservation()
h=fixture();m=frame3d_from_handoff(h,6e5);
close(m.loads(1:4,1:3),h.allFinalForces);assert(all(m.loads(5:end,:)==0,'all'));
assert(all(m.loads(:,4:6)==0,'all'));close(sum(m.loads(:,1:3),1),sum(h.allFinalForces,1));
close(sum(cross(m.nodes,m.loads(:,1:3),2),1),sum(cross(m.originalNodes,h.allFinalForces,2),1));
end

function supports_and_rotations()
h=fixture();m=frame3d_from_handoff(h,6e5);
assert(isequal(m.restraints(1:4,1:3),h.translationalRestraints));
assert(~any(m.restraints(:,4:6),'all'));assert(~any(m.restraints(5:end,:),'all'));
assert(sum(any(m.restraints,2))==3);
end

function member_properties()
h=fixture();G=[5e5;6e5;7e5];m=frame3d_from_handoff(h,G);
s=frame3d_section(sqrt(h.A),sqrt(h.A));
close(m.A,h.A(m.frameMemberRows));close(m.E,h.E_effective(m.frameMemberRows));close(m.G,G(m.frameMemberRows));
for name={'I2','I3','J','As2','As3'}
    k=name{1};close(m.(k),s.(k)(m.frameMemberRows));
end
assert(all(m.G~=m.E/2.6));
end

function original_input_unchanged()
h=fixture();saved=h;frame3d_from_handoff(h,6e5);assert(isequaln(h,saved));
end

function prestress_and_reference_ignored()
h=fixture();a=frame3d_from_handoff(h,6e5);h.Lref=[1e-4;80;10];
h.signedAxialForce=-1e9*h.signedAxialForce;h.signedAxialStress=zeros(3,1);
h.strain=[100;200;300];h.invertedNodes=ones(4,3)*999;
b=frame3d_from_handoff(h,6e5);
for name={'nodes','bars','loads','restraints','A','E','G','I2','I3','J','As2','As3'}
    k=name{1};assert(isequaln(a.(k),b.(k)));
end
assert(~a.initialStressTransferred&&~b.initialStressTransferred);
end

function unmirrored_single_segment()
h=fixture();m=frame3d_from_handoff(h,6e5,struct('MirrorZ',false,'Segments',1));
close(m.nodes,h.nodes);assert(isequal(m.bars,h.bars));assert(size(m.loads,1)==4);
end

function json_file_input()
h=fixture();file=[tempname '.json'];cleanup=onCleanup(@()delete(file)); %#ok<NASGU>
fid=fopen(file,'w');fprintf(fid,'%s',jsonencode(h));fclose(fid);
a=frame3d_from_handoff(h,6e5);b=frame3d_from_handoff(file,6e5);
close(a.nodes,b.nodes);close(a.loads,b.loads);close(a.E,b.E);close(a.G,b.G);
assert(strcmp(b.sourcePath,file));
end

function missing_G_rejected()
failed=false;try,frame3d_from_handoff(fixture());catch ex,failed=strcmp(ex.identifier,'frame3d:GRequired');end
assert(failed);
end

function unsuccessful_source_rejected()
h=fixture();h.success=false;failed=false;
try,frame3d_from_handoff(h,6e5);catch ex,failed=strcmp(ex.identifier,'frame3d:HandoffConvergence');end
assert(failed);
end

function failed_residual_rejected()
h=fixture();h.residualCheck.passes=false;failed=false;
try,frame3d_from_handoff(h,6e5);catch ex,failed=strcmp(ex.identifier,'frame3d:HandoffConvergence');end
assert(failed);
end

function numeric_residual_rejected()
h=fixture();h.residualCheck.freeForceResidual_kN=1;failed=false;
try,frame3d_from_handoff(h,6e5);catch ex,failed=strcmp(ex.identifier,'frame3d:HandoffConvergence');end
assert(failed);
end
