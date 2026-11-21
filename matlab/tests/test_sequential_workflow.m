function report=test_sequential_workflow()
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'code'));
tests={@c24_properties,@uniform_cantilever,@fixed_beam_body_moments,@case_superposition, ...
    @body_load_global_closure,@member_reversal,@symmetry_groups,@invalid_symmetry_rejected, ...
    @wind_and_combination_signs,@design_factors_not_in_fft_defaults,@tension_branch_no_column_reduction,@invalid_sizing_settings_rejected};
passed={};failures={};
for i=1:numel(tests)
    name=func2str(tests{i});try,tests{i}();passed{end+1}=name;fprintf('PASS %s\n',name); %#ok<AGROW>
    catch ex,failures{end+1}=struct('name',name,'message',ex.message);fprintf('FAIL %s: %s\n',name,ex.message);end %#ok<AGROW>
end
report=struct('passed',numel(passed),'failed',numel(failures),'tests',{passed},'failures',{failures});
gs_write_json(fullfile(root,'outputs','tests','sequential_test_report.json'),report);
assert(isempty(failures),'sequential:Tests','Sequential tests failed.');
end
function model=beam()
s=frame3d_section(.2,.2);model=struct('nodes',[0,0,0;3,0,0],'bars',[1,2], ...
    'E',11e6,'G',.69e6,'restraints',[true(1,6);false(1,6)],'loads',zeros(2,6),'local2',[0,1,0]);
for f={'A','I2','I3','J','As2','As3'},model.(f{1})=s.(f{1});end
end
function near(a,b,tol)
if nargin<3,tol=2e-10;end
assert(max(abs(a(:)-b(:)))<=tol*max(1,max(abs(b(:)))),'sequential:Comparison','Mismatch with independent expectation.');
end
function c24_properties()
[m,o]=st_material();near(m.fc0k,21000);near(m.ft0k,14500);near(m.fmk,24000);near(m.unitWeight,4.1202);
near(m.gammaM,1.3);near(m.kmodPermanent,.6);near(m.kmodWind,.9);near(o.km,.7);near(o.betaC,.2);
end
function uniform_cantilever()
m=beam();q=-2;out=st_frame_batch(m,zeros(2,6),[0,q,0],{'UDL'});s=out.cases{1};L=3;
expected=q*(L^4/(8*m.E*m.I3)+L^2/(2*m.G*m.As2));near(s.displacements(2,2),expected);
near(s.displacements(2,6),q*L^3/(6*m.E*m.I3));near(s.reactions(1,2),-q*L);near(s.reactions(1,6),-q*L^2/2);
near(s.sectionForcesJ,zeros(1,6));near(max(s.envelopeActions(:,6)),abs(q)*L^2/2);
end
function fixed_beam_body_moments()
m=beam();m.restraints=true(2,6);q=-2;out=st_frame_batch(m,zeros(2,6),[0,q,0],{'fixed'});s=out.cases{1};L=3;
near(s.displacements,zeros(2,6));near(s.reactions(:,2),repmat(-q*L/2,2,1));
near(s.reactions(:,6),[-q*L^2/12;q*L^2/12]);near(max(s.envelopeActions(:,6)),abs(q)*L^2/12);
end
function case_superposition()
m=beam();f=zeros(2,6,3);f(2,3,1)=3;f(2,3,3)=3;q=zeros(1,3,3);q(1,2,2)=-2;q(1,2,3)=-2;
out=st_frame_batch(m,f,q,{'P','q','Pplusq'});near(out.cases{3}.displacements,out.cases{1}.displacements+out.cases{2}.displacements);
near(out.cases{3}.sectionForcesI,out.cases{1}.sectionForcesI+out.cases{2}.sectionForcesI);
end
function body_load_global_closure()
m=beam();m.nodes=[2,-3,4;4,1,7];m=rmfield(m,'local2');q=[.4,-1.2,-2.5];out=st_frame_batch(m,zeros(2,6),q,{'skew'});s=out.cases{1};
near(s.globalForceClosure,zeros(1,3));near(s.globalMomentClosure,zeros(1,3));
L=norm(m.nodes(2,:)-m.nodes(1,:));near(sum(s.appliedNodalForces(:,1:3),1),q*L);
near(sum(s.appliedNodalForces(:,4:6)+cross(m.nodes,s.appliedNodalForces(:,1:3),2),1),cross(mean(m.nodes),q*L));
end
function member_reversal()
m=beam();a=st_frame_batch(m,zeros(2,6),[0,-2,0],{'forward'});m.bars=[2,1];b=st_frame_batch(m,zeros(2,6),[0,-2,0],{'reverse'});
near(a.cases{1}.displacements,b.cases{1}.displacements);near(a.cases{1}.reactions,b.cases{1}.reactions);
end
function symmetry_groups()
x=[-1,-1,0;1,-1,0;1,1,0;-1,1,0];b=[1,2;2,3;3,4;4,1];[g,info]=st_symmetry_groups(x,b,true(4,3),'xy');
assert(info.numberOfGroups==2&&g(1)==g(3)&&g(2)==g(4)&&g(1)~=g(2));
end
function invalid_symmetry_rejected()
x=[-1,-1,0;1,-1,0;1,1,0;-1,1,0];x(1,1)=-1.1;failed=false;
try,st_symmetry_groups(x,[1,2;2,3;3,4;4,1],true(4,3),'xy');catch,failed=true;end;assert(failed);
end
function wind_and_combination_signs()
x=[0,0,0;2,0,0;2,2,0;0,2,0];[c,i]=st_load_cases(x,x,{[1,2,3,4]},0,3);
near(i.unfactoredPanelDeadNodal,zeros(4,3));near(sum(i.unfactoredPanelWindDownNodal(:,3)),-12);
assert(c(2).windSign==1&&c(4).windSign==-1&&c(1).gammaW==0);
near(c(1).kmod,.6);near(c(2).kmod,.9);near(c(2).gammaG,1.35);near(c(3).gammaG,1);near(c(2).gammaW,1.5);
end
function design_factors_not_in_fft_defaults()
o=gs_options('',struct());near(o.DeadLoadFactor,1);near(o.WindLoadFactor,1);near(o.FFTolerance,1e-4);
near(o.ForceTolerance,1e-5);near(o.AbsoluteForceTolerance,1e-6);
end
function tension_branch_no_column_reduction()
[m,o]=st_material();m.kmod=.9;s=frame3d_section(.1,.1);a=[.5*s.A*m.ft0k*m.kmod/m.gammaM,0,0,0,0,0];
c=timber_frame_checks(a,s,20,m,o);near(c.ratio,.5);assert(c.isTension&&~c.columnBucklingActive&&~c.fullDesignVerified);
end
function invalid_sizing_settings_rejected()
b=struct('originalA',.01);[m,o]=st_material();settings={struct('TargetRatio',1.1),struct('MaximumSide',.01),struct('MaximumIterations',0)};
for j=1:numel(settings)
    failed=false;try,st_iterative_size(b,m,o,[],1,settings{j});catch,failed=true;end;assert(failed);
end
failed=false;try,st_iterative_size(b,m,o,[],1.5,struct());catch,failed=true;end;assert(failed);
end
