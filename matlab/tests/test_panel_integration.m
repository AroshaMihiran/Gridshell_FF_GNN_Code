function report=test_panel_integration()
% Independent geometry, static-equivalence and sign tests for new load surfaces.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'code'));
tests={@uniform_triangle,@square_symmetry,@irregular_quad_exact,@regular_hexagon, ...
    @irregular_polygon_centroid,@collinear_polygon_nodes,@wind_direction, ...
    @load_superposition,@geometry_area_update,@projection_invariance, ...
    @warped_quad_quadrature,@warped_polygon_quadrature,@node_order_invariance, ...
    @rigid_rotation_surface,@translation_moment,@shared_node_assembly, ...
    @cache_identity,@cache_reject_mismatch,@bad_magnitudes,@bad_geometry, ...
    @polygon_reference_required,@shape_function_completeness,@polygon_derivatives};
passed={};failures={};
for k=1:numel(tests)
    name=func2str(tests{k});
    try,tests{k}();passed{end+1}=name;fprintf('PASS %s\n',name); %#ok<AGROW>
    catch ex,failures{end+1}=struct('test',name,'identifier',ex.identifier,'message',ex.message);fprintf('FAIL %s: %s\n',name,ex.message);end %#ok<AGROW>
end
report=struct('passed',numel(passed),'failed',numel(failures),'tests',{passed},'failures',{failures});
gs_write_json(fullfile(root,'outputs','tests','panel_integration_test_report.json'),report);
fprintf('PANEL INTEGRATION TESTS: %d passed; %d failed.\n',report.passed,report.failed);
assert(isempty(failures),'gs:PanelTestFailure','Panel integration tests failed.');
end
function v=square(),v=[0 0 0;2 0 0;2 2 0;0 2 0];end
function v=hexagon()
a=(0:5)'*pi/3;v=[cos(a),sin(a),zeros(6,1)];
end
function near(a,b,tol)
if nargin<3,tol=2e-11;end
assert(all(abs(a(:)-b(:))<=tol*max(1,max(abs(b(:))))),'gs:Comparison','Value differs from independent expectation.');
end
function expectError(f,identifier)
try,f();catch ex,if nargin<2||strcmp(ex.identifier,identifier),return;end;rethrow(ex);end
error('gs:MissingError','Invalid input was accepted.');
end
function [f,i]=loads(v,q1,q2,o)
if nargin<4,o=struct();end
[F,i]=gs_panel_loads(v,{1:size(v,1)},q1,q2,o);f=reshape(F,3,[])';
near(i.forceResidual,[0 0 0]);near(i.momentResidual,[0 0 0]);
end
function uniform_triangle()
v=[0 0 0;4 0 0;1 3 0];[f,i]=loads(v,2,0);
near(f(:,3),[-4;-4;-4]);near(i.panelAreas,6);assert(strcmp(i.methods{1},'T3 barycentric'));
end
function square_symmetry()
[f,i]=loads(square(),3,0);near(f(:,3),-3*ones(4,1));near(i.panelAreas,4);
assert(strcmp(i.methods{1},'Q4 bilinear'));
end
function irregular_quad_exact()
v=[0 0 0;4 0 0;2 2 0;0 2 0];[f,i]=loads(v,1,0);
near(f(:,3),-[5;5;4;4]/3);near(i.panelAreas,6);
near(i.totalMoment,[-16/3,28/3,0]);
end
function regular_hexagon()
[f,i]=loads(hexagon(),2,0);A=3*sqrt(3)/2;
near(i.panelAreas,A);near(f(:,3),-2*A/6*ones(6,1));
assert(strcmp(i.methods{1},'Polygon mean-value'));
end
function irregular_polygon_centroid()
v=[0 0 0;3 0 0;4 1 0;2 3 0;0 2 0];[A,c]=polygonAreaCentroid(v(:,1:2));
[f,i]=loads(v,2.7,.6);near(sum(f(:,3)),-3.3*A);
near(i.totalMoment,cross([c 0],[0 0 -3.3*A]));near(i.panelAreas,A);
end
function collinear_polygon_nodes()
v=[0 0 0;1 0 0;2 0 0;2 2 0;0 2 0];[f,i]=loads(v,1,0);
near(i.panelAreas,4);near(i.totalMoment,[-4 4 0]);
assert(all(f(:,3)<0),'A collinear physical node must retain its load contribution.');
assert(size(i.PanelIntegration{1}.N,2)==5);
end
function wind_direction()
v=square();[down,~]=loads(v,0,2);[up,~]=loads(v,0,2,struct('WindDirection','upward'));
near(up,-down);near(up(:,3),2*ones(4,1));
end
function load_superposition()
v=square();a=loads(v,1,0);b=loads(v,0,.6);c=loads(v,1,.6);near(a+b,c);
d=loads(v,1,.6,struct('WindDirection','upward'));near(d,a-b);
e=loads(v,1,.6,struct('DeadLoadFactor',1.35,'WindLoadFactor',1.5));near(e,1.35*a+1.5*b);
end
function geometry_area_update()
v=square();o=struct('ReferenceNodes',v);v(3:4,3)=1.5;
[f,i]=loads(v,1,.5,o);near(i.panelAreas,5);near(i.panelProjectedAreas,4);near(f(:,3),-1.75*ones(4,1));
end
function projection_invariance()
v=square();flat=loads(v,0,1);o=struct('ReferenceNodes',v);v(:,3)=[0;.2;1.1;-.4];
warped=loads(v,0,1,o);near(warped,flat);
end
function warped_quad_quadrature()
ref=square();v=ref;v(:,3)=[.1;.8;-.2;.4];
o=struct('ReferenceNodes',ref,'IntegrationOrder',8);[f8,i8]=loads(v,1,0,o);
o.IntegrationOrder=16;[f16,i16]=loads(v,1,0,o);near(f8,f16,1e-8);near(i8.panelAreas,i16.panelAreas,1e-8);
o.IntegrationOrder=32;f32=loads(v,1,0,o);near(f16,f32,1e-11);
end
function warped_polygon_quadrature()
ref=hexagon();v=ref;v(:,3)=[.2;-.1;.3;-.15;.1;-.05];
o=struct('ReferenceNodes',ref,'IntegrationOrder',8);f8=loads(v,1,0,o);
o.IntegrationOrder=16;f16=loads(v,1,0,o);
o.IntegrationOrder=32;f32=loads(v,1,0,o);
near(f8,f32,5e-4);near(f16,f32,3e-5);
assert(norm(f16-f32,'fro')<norm(f8-f32,'fro'),'Quadrature refinement must improve this diagnostic.');
end
function node_order_invariance()
for v={square(),hexagon(),[0 0 0;3 0 0;4 1 0;2 3 0;0 2 0]}
    x=v{1};n=size(x,1);[f,~]=loads(x,1,.5);
    for ix={[2:n,1],n:-1:1}
        order=ix{1};g=loads(x(order,:),1,.5);back=zeros(size(g));back(order,:)=g;near(back,f,1e-9);
    end
end
end
function rigid_rotation_surface()
v=hexagon();f=loads(v,1,0);a=.65;R=[1 0 0;0 cos(a) -sin(a);0 sin(a) cos(a)];
g=loads(v*R',1,0);near(g,f,1e-10);
end
function translation_moment()
v=hexagon();[f,i]=loads(v,1,.2);d=[100 -30 20];[g,j]=loads(v+d,1,.2);
near(g,f,2e-10);near(j.totalMoment,i.totalMoment+cross(d,i.totalForce),2e-10);
end
function shared_node_assembly()
v=[0 0 0;1 0 0;2 0 0;0 1 0;1 1 0;2 1 0];
[F,i]=gs_panel_loads(v,{[1 2 5 4],[2 3 6 5]},[1;3],0,struct());
f=reshape(F,3,[])';near(f(:,3),-[.25;1;.75;.25;1;.75]);near(i.totalForce,[0 0 -4]);
end
function cache_identity()
ref=hexagon();v=ref;v(:,3)=[.1;.2;-.1;0;.15;-.2];o=struct('ReferenceNodes',ref);
[f,i]=loads(v,1,0,o);o.PanelIntegration=i.PanelIntegration;g=loads(v,1,0,o);near(g,f);
end
function cache_reject_mismatch()
ref=hexagon();[~,i]=loads(ref,1,0);o=struct('ReferenceNodes',ref,'PanelIntegration',{i.PanelIntegration},'IntegrationOrder',16);
expectError(@()loads(ref,1,0,o),'gs:IntegrationCache');
end
function bad_magnitudes()
v=square();expectError(@()loads(v,1,-1),'gs:LoadMagnitude');
expectError(@()loads(v,-1,1),'gs:LoadMagnitude');
expectError(@()loads(v,1,1,struct('WindDirection','auto')),'gs:WindDirection');
expectError(@()loads(v,1,1,struct('WindMode','normal')),'gs:WindMode');
    expectError(@()loads(v,realmax,1),'gs:LoadOverflow');
end
function bad_geometry()
v=square();expectError(@()loads(v([1 3 2 4],:),1,0));
expectError(@()loads([0 0 0;1 0 0;2 0 0],1,0));
expectError(@()loads([0 0 0;2 0 0;.5 .5 0;0 2 0],1,0),'gs:ConvexPanel');
v=hexagon();expectError(@()loads(v([1 3 5 2 4 6],:),1,0));
end
function polygon_reference_required()
v=hexagon();v(:,3)=[0;.3;0;.2;0;-.1];
expectError(@()loads(v,1,0),'gs:ReferencePlanarity');
end
function shape_function_completeness()
v=[0 0 0;3 0 0;4 1 0;2 3 0;0 2 0];[~,i]=loads(v,1,0);
r=i.PanelIntegration{1};near(sum(r.N,2),ones(size(r.N,1),1));near(sum(r.dN1,2),zeros(size(r.N,1),1),1e-9);
near(sum(r.dN2,2),zeros(size(r.N,1),1),1e-9);
% The mapped planar Jacobian must be one throughout this reference domain.
j=cross(r.dN1*v,r.dN2*v,2);near(sqrt(sum(j.^2,2)),ones(size(j,1),1),1e-9);
end
function [A,c]=polygonAreaCentroid(v)
w=v([2:end,1],:);a=v(:,1).*w(:,2)-w(:,1).*v(:,2);A=sum(a)/2;
c=sum((v+w).*a,1)/(6*A);
end
function polygon_derivatives()
v=[0 0;3 0;4 1;2 3;0 2];x=[.7 .8;2 .2;2.4 1.5];h=1e-6;
[N,dx,dy]=gs_load_polygon_coordinates(v,x);
nx=(gs_load_polygon_coordinates(v,x+[h 0])-gs_load_polygon_coordinates(v,x-[h 0]))/(2*h);
ny=(gs_load_polygon_coordinates(v,x+[0 h])-gs_load_polygon_coordinates(v,x-[0 h]))/(2*h);
near(dx,nx,2e-8);near(dy,ny,2e-8);near(N*v,x,1e-11);
% Verify a warped mapped surface tangent independently by coordinate differences.
xyz=[v,[0;.2;-.1;.3;-.2]];
near(dx*xyz,(gs_load_polygon_coordinates(v,x+[h 0])*xyz-gs_load_polygon_coordinates(v,x-[h 0])*xyz)/(2*h),2e-8);
end
