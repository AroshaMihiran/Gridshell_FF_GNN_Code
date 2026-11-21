function report=test_frame3d()
%TEST_FRAME3D Analytical regression checks independent of ETABS output.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'code'));
tests={@section_properties,@cantilever_axial,@cantilever_torsion, ...
    @cantilever_bending2_shear,@cantilever_bending3_shear,@pure_moment, ...
    @rotated_frame_invariance,@member_subdivision,@rigid_body_energy, ...
    @global_equilibrium,@mechanism_rejected,@all_restrained, ...
    @invalid_axes_rejected,@invalid_connectivity_rejected};
passed={};failures={};
for k=1:numel(tests)
    name=func2str(tests{k});
    try,tests{k}();passed{end+1}=name;fprintf('PASS %s\n',name); %#ok<AGROW>
    catch ex,failures{end+1}=struct('test',name,'message',ex.message);fprintf('FAIL %s: %s\n',name,ex.message);end %#ok<AGROW>
end
report=struct('passed',numel(passed),'failed',numel(failures),'tests',{passed},'failures',{failures});
fid=fopen(fullfile(root,'outputs','tests','frame3d_test_report.json'),'w');
if fid>=0,fprintf(fid,'%s',jsonencode(report,'PrettyPrint',true));fclose(fid);end
fprintf('FRAME3D TESTS: %d passed; %d failed.\n',report.passed,report.failed);
assert(isempty(failures),'frame3d:TestFailure','Some analytical tests failed.');
end

function m=base_model()
m=struct('nodes',[0,0,0;3,0,0],'bars',[1,2],'E',1.1e7,'G',6.9e5, ...
    'restraints',[true(1,6);false(1,6)],'loads',zeros(2,6),'local2',[0,1,0]);
s=frame3d_section(.12,.30);names={'A','I2','I3','J','As2','As3'};
for k=1:numel(names),m.(names{k})=s.(names{k});end
end

function close(actual,expected,tol)
if nargin<3,tol=1e-10;end
assert(all(abs(actual(:)-expected(:))<=tol*max(1,max(abs(expected(:))))), ...
    'frame3d:Comparison','Maximum error %.12g.',max(abs(actual(:)-expected(:))));
end

function section_properties()
s=frame3d_section([.1;.2],[.3;.2]);
close(s.A,[.03;.04]);close(s.I2,[.3*.1^3/12;.2^4/12]);
close(s.I3,[.1*.3^3/12;.2^4/12]);close(s.W3,s.I3./([.3;.2]/2));
close(s.As2,5*s.A/6);close(s.J(2)/.2^4,.140577015,1e-8);
assert(all(s.J<s.I2+s.I3));
end

function cantilever_axial()
m=base_model();m.loads(2,1)=17;r=frame3d_linear(m);assert(r.success);
close(r.displacements(2,1),17*3/(m.E*m.A));
close(r.sectionForcesI(1,1),17);close(r.sectionForcesJ(1,1),17);
close(r.reactions(1,1),-17);close(r.displacements(2,2:6),zeros(1,5));
end

function cantilever_torsion()
m=base_model();m.loads(2,4)=9;r=frame3d_linear(m);assert(r.success);
close(r.displacements(2,4),9*3/(m.G*m.J));close(r.sectionForcesI(1,4),9);
close(r.sectionForcesJ(1,4),9);close(r.reactions(1,4),-9);
end

function cantilever_bending2_shear()
m=base_model();P=11;L=3;m.loads(2,2)=P;r=frame3d_linear(m);assert(r.success);
close(r.displacements(2,2),P*L^3/(3*m.E*m.I3)+P*L/(m.G*m.As2));
close(r.displacements(2,6),P*L^2/(2*m.E*m.I3));
close(r.sectionForcesI(1,[2,6]),[P,P*L]);close(r.sectionForcesJ(1,[2,6]),[P,0]);
end

function cantilever_bending3_shear()
m=base_model();P=8;L=3;m.loads(2,3)=P;r=frame3d_linear(m);assert(r.success);
close(r.displacements(2,3),P*L^3/(3*m.E*m.I2)+P*L/(m.G*m.As3));
close(r.displacements(2,5),-P*L^2/(2*m.E*m.I2));
close(r.sectionForcesI(1,[3,5]),[P,-P*L]);close(r.sectionForcesJ(1,[3,5]),[P,0]);
end

function pure_moment()
m=base_model();M=7;L=3;m.loads(2,6)=M;r=frame3d_linear(m);assert(r.success);
close(r.displacements(2,2),M*L^2/(2*m.E*m.I3));
close(r.displacements(2,6),M*L/(m.E*m.I3));
close(r.sectionForcesI(1,[2,6]),[0,M]);close(r.sectionForcesJ(1,[2,6]),[0,M]);
end

function rotated_frame_invariance()
m=base_model();m.loads(2,:)=[3,5,7,2,4,6];r=frame3d_linear(m);assert(r.success);
axis=[1,2,3];axis=axis/norm(axis);a=.713;S=[0,-axis(3),axis(2);axis(3),0,-axis(1);-axis(2),axis(1),0];
Q=eye(3)+sin(a)*S+(1-cos(a))*S*S;
n=m;n.nodes=m.nodes*Q'+[10,-7,4];n.local2=m.local2*Q';
n.loads(:,1:3)=m.loads(:,1:3)*Q';n.loads(:,4:6)=m.loads(:,4:6)*Q';
t=frame3d_linear(n);assert(t.success);
close(t.displacements(:,1:3),r.displacements(:,1:3)*Q');
close(t.displacements(:,4:6),r.displacements(:,4:6)*Q');close(t.endForces,r.endForces);
close(t.strainEnergy,r.strainEnergy);close(t.globalForceClosure,[0,0,0]);close(t.globalMomentClosure,[0,0,0]);
end

function member_subdivision()
m=base_model();m.loads(2,:)=[3,5,7,2,4,6];r=frame3d_linear(m);
n=m;n.nodes=[(0:4)'*3/4,zeros(5,2)];n.bars=[(1:4)',(2:5)'];n.local2=repmat(m.local2,4,1);
n.restraints=false(5,6);n.restraints(1,:)=true;n.loads=zeros(5,6);n.loads(5,:)=m.loads(2,:);
t=frame3d_linear(n);assert(t.success);close(t.displacements(end,:),r.displacements(2,:));
close(t.reactions(1,:),r.reactions(1,:));close(t.sectionForcesI(1,:),r.sectionForcesI);
close(t.sectionForcesJ(end,:),r.sectionForcesJ);
end

function rigid_body_energy()
m=base_model();r=frame3d_linear(m);omega=[.02,-.04,.03];shift=[.1,-.2,.3];
u=[repmat(shift,2,1)+cross(repmat(omega,2,1),m.nodes,2),repmat(omega,2,1)];
v=reshape(u',[],1);close(r.K*v,zeros(12,1),1e-8);close(v'*r.K*v,0,1e-9);
end

function global_equilibrium()
m=base_model();m.nodes=[4,-2,8;6,1,9];m.local2=[0,0,1];m.loads(2,:)=[3,-5,7,2,4,-6];
r=frame3d_linear(m);assert(r.success);close(r.globalForceClosure,[0,0,0]);close(r.globalMomentClosure,[0,0,0]);
close(r.strainEnergy,.5*sum(sum(m.loads.*r.displacements)));
end

function mechanism_rejected()
m=base_model();m.restraints(:)=false;m.loads(2,2)=1;r=frame3d_linear(m);
assert(~r.success);assert(contains(r.reason,'stiffness'));assert(any(isnan(r.displacements),'all'));
end

function all_restrained()
m=base_model();m.restraints(:)=true;m.loads(2,:)=[3,-5,7,2,4,-6];r=frame3d_linear(m);
assert(r.success);close(r.displacements,zeros(2,6));close(r.reactions,-m.loads);close(r.endForces,zeros(1,12));
end

function invalid_axes_rejected()
m=base_model();m.local2=[1,0,0];thrown=false;
try,frame3d_linear(m);catch ex,thrown=strcmp(ex.identifier,'frame3d:LocalAxes');end
assert(thrown);
end

function invalid_connectivity_rejected()
m=base_model();m.bars=[1,3];thrown=false;
try,frame3d_linear(m);catch ex,thrown=strcmp(ex.identifier,'frame3d:Connectivity');end
assert(thrown);
end
