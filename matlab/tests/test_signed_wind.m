function report=test_signed_wind
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'code'),'-begin');
X=[0 0 0;1 0 0;0 1 0];P={1:3};o=struct('WindDirection','signed');
[F,~]=gs_panel_loads(X,P,4,-1,o);assert(abs(sum(F(3:3:end))+1.5)<1e-14);
[F2,~]=gs_panel_loads(X,P,6,1,o);assert(abs(sum(F2(3:3:end))+3.5)<1e-14);
for geometry=1:2
 if geometry==2,X(3,3)=.4;end
 for magnitude=[0 .5 1 4]
  [Sd,~]=gs_panel_loads(X,P,4,magnitude,o);
  [Od,~]=gs_panel_loads(X,P,4,magnitude,struct('WindDirection','downward'));
  assert(isequal(Sd,Od));
  [Su,~]=gs_panel_loads(X,P,4,-magnitude,o);
  [Ou,~]=gs_panel_loads(X,P,4,magnitude,struct('WindDirection','upward'));
  assert(isequal(Su,Ou));
 end
end
rejected=false;try,gs_panel_loads(X,P,4,-1,struct('WindDirection','downward'));catch e,rejected=strcmp(e.identifier,'gs:LoadMagnitude');end
assert(rejected);
bad=gs_options(fullfile(root,'inputs','nonuniform'),struct());rejected=false;
try,gs_read_model(fullfile(root,'inputs','nonuniform'),bad);catch e,rejected=strcmp(e.identifier,'gs:LoadMagnitude');end
assert(rejected);
good=gs_options(fullfile(root,'inputs','nonuniform'),struct('WindDirection','signed'));m=gs_read_model(fullfile(root,'inputs','nonuniform'),good);
assert(all(m.q2(m.q1==4)==-1)&&all(m.q2(m.q1==6)==-1));
report=struct('passes',true,'flat_signed_cases',2,'old_mode_exact_equivalence_cases',16, ...
 'negative_wind_rejected_without_explicit_signed_mode',true,'read_model_negative_requires_signed_mode',true, ...
 'signed_wind_convention','Positive qW downward; negative qW upward per projected area');
gs_write_json(fullfile(root,'outputs','tests','signed_wind_tests.json'),report);disp(report);
end
