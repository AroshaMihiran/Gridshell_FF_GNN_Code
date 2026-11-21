function report=test_timber_frame_checks()
% Independent arithmetic/mode tests for the normal-stress design checker.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'code'));
tests={@zero_load_scope,@stocky_compression,@pure_tension,@pure_biaxial_bending, ...
    @stocky_compression_bending,@slender_column_known_value,@slender_biaxial_interaction, ...
    @one_slender_axis,@tension_not_column_buckling,@moment_sign_invariance, ...
    @axis_exchange,@unit_scaling,@heterogeneous_properties,@physical_length_effect, ...
    @strict_capacity_boundary,@missing_strength_rejected,@invalid_inputs_rejected};
passed={};failures={};
for k=1:numel(tests)
    name=func2str(tests{k});
    try,tests{k}();passed{end+1}=name;fprintf('PASS %s\n',name); %#ok<AGROW>
    catch ex,failures{end+1}=struct('test',name,'message',ex.message);fprintf('FAIL %s: %s\n',name,ex.message);end %#ok<AGROW>
end
report=struct('passed',numel(passed),'failed',numel(failures),'tests',{passed},'failures',{failures});
fid=fopen(fullfile(root,'outputs','tests','timber_frame_checks_test_report.json'),'w');
if fid>=0,fprintf(fid,'%s',jsonencode(report,'PrettyPrint',true));fclose(fid);end
fprintf('TIMBER FRAME CHECK TESTS: %d passed; %d failed.\n',report.passed,report.failed);
assert(isempty(failures),'timberFrame:TestFailure','Some timber check tests failed.');
end

function [s,m]=base()
s=frame3d_section(.2,.2);
m=struct('fc0k',20000,'ft0k',14000,'fmk',24000,'E05',7.4e6,'kmod',.8,'gammaM',1.3);
end
function near(a,b,tol)
if nargin<3,tol=2e-12;end
assert(isequal(size(a),size(b))||isscalar(a)||isscalar(b));
assert(all(abs(a(:)-b(:))<=tol*max(1,max(abs(b(:))))),'timberFrame:Comparison','Numerical value does not match independent expectation.');
end
function a=demand(s,m,c,t,b2,b3)
a=[s.A*(t*m.ft0k-c*m.fc0k)*m.kmod/m.gammaM,0,0,0, ...
    b2*s.W2*m.fmk*m.kmod/m.gammaM,b3*s.W3*m.fmk*m.kmod/m.gammaM];
end
function zero_load_scope()
[s,m]=base();c=timber_frame_checks(zeros(1,6),s,3,m);near(c.ratio,0);assert(c.passes);
assert(~c.fullDesignVerified&&~c.globalStabilityVerified&&~c.lateralTorsionalBucklingVerified);
a=[0,5,0,2,0,0];c=timber_frame_checks(a,s,3,m);
assert(c.hasShearDemand&&c.hasTorsionDemand&&~c.shearResistanceVerified&&~c.torsionResistanceVerified);
end
function stocky_compression()
[s,m]=base();c=timber_frame_checks(demand(s,m,.6,0,0,0),s,.1,m);
near(c.kc2,1);near(c.utilizationCrossSection2,.36);near(c.ratio,.6);assert(~c.columnBucklingActive);
end
function pure_tension()
[s,m]=base();c=timber_frame_checks(demand(s,m,0,.75,0,0),s,8,m);
near(c.ratio,.75);assert(~c.columnBucklingActive);near(c.utilizationBuckling2,0);
end
function pure_biaxial_bending()
[s,m]=base();c=timber_frame_checks(demand(s,m,0,0,.4,.3),s,3,m);
near(c.utilizationCrossSection2,.61);near(c.utilizationCrossSection3,.58);near(c.ratio,.61);
end
function stocky_compression_bending()
[s,m]=base();c=timber_frame_checks(demand(s,m,.4,0,.5,.2),s,.1,m);
near(c.utilizationCrossSection2,.80);near(c.utilizationCrossSection3,.71);near(c.ratio,.80);
end
function slender_column_known_value()
[s,m]=base();L=pi*sqrt(s.I2/s.A)*sqrt(m.E05/m.fc0k); % lambda=1 exactly
c=timber_frame_checks(demand(s,m,.4,0,0,0),s,L,m);
near(c.lambdaRel2,1);expected=1/(1.07+sqrt(1.07^2-1));
near(c.kc2,expected);near(c.ratio,.4/expected);assert(c.columnBucklingActive);
end
function slender_biaxial_interaction()
[s,m]=base();L=pi*sqrt(s.I2/s.A)*sqrt(m.E05/m.fc0k);
c=timber_frame_checks(demand(s,m,.4,0,.2,.3),s,L,m);
kc=1/(1.07+sqrt(1.07^2-1));
near(c.utilizationBuckling2,.4/kc+.2+.7*.3);
near(c.utilizationBuckling3,.4/kc+.7*.2+.3);near(c.ratio,.4/kc+.7*.2+.3);
end
function one_slender_axis()
[~,m]=base();s=frame3d_section(.1,.8);
L=.8*pi*sqrt(s.I2/s.A)*sqrt(m.E05/m.fc0k);
c=timber_frame_checks(demand(s,m,.3,0,.1,.7),s,L,m);
near(c.lambdaRel2,.8);near(c.lambdaRel3,.1);near(c.kc3,1);
% A stocky second axis STILL uses the linear column equation when either is slender.
near(c.utilizationBuckling3,.3+.7*.1+.7);
assert(c.columnBucklingActive&&~c.passes);
end
function tension_not_column_buckling()
[s,m]=base();c=timber_frame_checks(demand(s,m,0,.3,.2,.4),s,100,m);
near(c.ratio,.84);near(c.utilizationBuckling2,0);assert(~c.columnBucklingActive);
end
function moment_sign_invariance()
[s,m]=base();a=demand(s,m,.25,0,.3,.15);b=a;b(5:6)=-b(5:6);
c=timber_frame_checks(a,s,3,m);d=timber_frame_checks(b,s,3,m);near(c.ratio,d.ratio);
end
function axis_exchange()
[~,m]=base();s=frame3d_section(.12,.28);a=[-180,0,0,0,5,11];
o=struct('EffectiveLengthFactor2',.8,'EffectiveLengthFactor3',1.2);
c=timber_frame_checks(a,s,2.3,m,o);s2=frame3d_section(.28,.12);a([5,6])=a([6,5]);
o2=struct('EffectiveLengthFactor2',1.2,'EffectiveLengthFactor3',.8);
d=timber_frame_checks(a,s2,2.3,m,o2);near(c.ratio,d.ratio);near(c.kc2,d.kc3);
end
function unit_scaling()
[s,m]=base();a=[-230,0,0,0,12,-7];c=timber_frame_checks(a,s,3,m);
% Equivalent N-mm-MPa system: force x1000, length x1000.
sm=frame3d_section(200,200);mm=m;
for name={'fc0k','ft0k','fmk','E05'},mm.(name{1})=m.(name{1})/1000;end
am=a;am(1:3)=am(1:3)*1000;am(4:6)=am(4:6)*1e6;
d=timber_frame_checks(am,sm,3000,mm);near(c.ratio,d.ratio);near(c.lambdaRel2,d.lambdaRel2);
end
function heterogeneous_properties()
[s,m]=base();a=repmat(demand(s,m,0,.5,0,0),2,1);m.ft0k=[14000;7000];
c=timber_frame_checks(a,s,[1;2],m);near(c.ratio,[.5;1]);
end
function physical_length_effect()
[s,m]=base();a=demand(s,m,.5,0,.1,0);c=timber_frame_checks(a,s,4,m);
d=timber_frame_checks(a,s,1,m);assert(c.ratio>d.ratio&&c.kc2<d.kc2);
o=struct('EffectiveLengthFactor2',4,'EffectiveLengthFactor3',4);
e=timber_frame_checks(a,s,1,m,o);near(c.ratio,e.ratio);
end
function strict_capacity_boundary()
[s,m]=base();c=timber_frame_checks(demand(s,m,0,1.001,0,0),s,1,m);assert(~c.passes);
c=timber_frame_checks(demand(s,m,0,.999,0,0),s,1,m);assert(c.passes);
end
function missing_strength_rejected()
[s,m]=base();expectError(@()timber_frame_checks(zeros(1,6),s,1,rmfield(m,'fmk')));
expectError(@()timber_frame_checks(zeros(1,6),s,1,rmfield(m,'E05')));
end
function invalid_inputs_rejected()
[s,m]=base();expectError(@()timber_frame_checks([NaN,zeros(1,5)],s,1,m));
expectError(@()timber_frame_checks(zeros(1,6),s,0,m));
expectError(@()timber_frame_checks(zeros(1,6),s,1,m,struct('km',1.1)));
expectError(@()timber_frame_checks(zeros(1,6),s,1,m,struct('EffectiveLengthFactor2',0)));
expectError(@()timber_frame_checks(zeros(1,6),s,1,m,struct('unrecognised',1)));
m.E05=[1;2];expectError(@()timber_frame_checks(zeros(1,6),s,1,m));
end
function expectError(f)
failed=false;try,f();catch,failed=true;end;assert(failed,'Expected invalid input to be rejected.');
end
