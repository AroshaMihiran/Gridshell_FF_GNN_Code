function [N,dx,dy]=gs_load_polygon_coordinates(v,x)
%GS_LOAD_POLYGON_COORDINATES Mean-value coordinates and analytic derivatives.
% V: CCW convex planar polygon, including collinear boundary vertices.
% X: strictly interior query points. This low-level helper is evaluated at
% quadrature points by gs_panel_loads, which validates the polygon geometry.
% No corner/edge limits are needed for those strictly interior quadratures.
% Uses two equivalent tan-half-angle forms to avoid edge cancellation.
% Source: Hormann & Floater (2006), Mean Value Coordinates for Arbitrary
% Planar Polygons, ACM TOG 25(4), 1424-1441, doi:10.1145/1183287.1183295.
validateattributes(v,{'numeric'},{'2d','ncols',2,'real','finite','nonempty'});
validateattributes(x,{'numeric'},{'2d','ncols',2,'real','finite','nonempty'});
n=size(v,1);nq=size(x,1);next=[2:n,1];prev=[n,1:n-1];
ax=v(:,1)'-x(:,1);ay=v(:,2)'-x(:,2);rad=hypot(ax,ay);
if any(rad(:)<=0),error('gs:PolygonQuadrature','Quadrature point coincides with a polygon corner.');end
rx=-ax./rad;ry=-ay./rad;
cr=ax.*ay(:,next)-ay.*ax(:,next);
dotp=ax.*ax(:,next)+ay.*ay(:,next);
den=rad.*rad(:,next)+dotp;
edge=v(next,:)-v;
cx=repmat(-edge(:,2)',nq,1);cy=repmat(edge(:,1)',nq,1);
denx=rx.*rad(:,next)+rad.*rx(:,next)-ax-ax(:,next);
deny=ry.*rad(:,next)+rad.*ry(:,next)-ay-ay(:,next);
tt=zeros(size(cr));tx=tt;ty=tt;
nearEdge=dotp<0;ordinary=~nearEdge;
if any(den(ordinary)<=0)||any(cr(nearEdge)<=0),error('gs:PolygonQuadrature','Invalid half-angle geometry.');end
tt(ordinary)=cr(ordinary)./den(ordinary);
tx(ordinary)=(cx(ordinary)-tt(ordinary).*denx(ordinary))./den(ordinary);
ty(ordinary)=(cy(ordinary)-tt(ordinary).*deny(ordinary))./den(ordinary);
num=rad.*rad(:,next)-dotp;
numx=rx.*rad(:,next)+rad.*rx(:,next)+ax+ax(:,next);
numy=ry.*rad(:,next)+rad.*ry(:,next)+ay+ay(:,next);
tt(nearEdge)=num(nearEdge)./cr(nearEdge);
tx(nearEdge)=(numx(nearEdge)-tt(nearEdge).*cx(nearEdge))./cr(nearEdge);
ty(nearEdge)=(numy(nearEdge)-tt(nearEdge).*cy(nearEdge))./cr(nearEdge);
ss=tt(:,prev)+tt;raw=ss./rad;
rawx=(tx(:,prev)+tx)./rad-raw.*rx./rad;
rawy=(ty(:,prev)+ty)./rad-raw.*ry./rad;
total=sum(raw,2);N=raw./total;
dx=(rawx-N.*sum(rawx,2))./total;dy=(rawy-N.*sum(rawy,2))./total;
end
