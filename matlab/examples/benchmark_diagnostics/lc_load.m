function [F,info]=lc_load(method,nodes,m,o)
% Controlled load alternatives. Original current and legacy sources unchanged.
% All alternatives use the same current geometry and fixed panel IDs.
[Fq,base]=gs_panel_loads(nodes,m.panels,m.q1,m.q2,o);
np=numel(m.panels);nn=size(nodes,1);nodal=zeros(nn,3);
windSign=-1;if strcmp(o.WindDirection,'upward'),windSign=1;end
ws=cell(np,1);wh=cell(np,1);panelF=cell(np,1);
area=base.panelAreas;projected=base.panelProjectedAreas;
legacyAudit=struct('called',false);
switch method
    case 'integrated_q4'
        F=Fq;info=compact(base);info.method=method;return
    case {'equal_integrated_area','angle_integrated_area','jiang_weights_integrated_area'}
        for p=1:np
            v=nodes(m.panels{p},:);n=size(v,1);
            if strcmp(method,'equal_integrated_area')
                a=ones(n,1)/n;b=a;
            elseif strcmp(method,'angle_integrated_area')
                a=legacy_angles(v);vp=v;vp(:,3)=0;b=legacy_angles(vp);
            else
                a=fan_shares(v);a=a/sum(a);vp=v;vp(:,3)=0;b=fan_shares(vp);b=b/sum(b);
            end
            ws{p}=area(p)*a;wh{p}=projected(p)*b;
        end
    case 'equal_newell_fixed_panels'
        for p=1:np
            v=nodes(m.panels{p},:);vp=v;vp(:,3)=0;n=size(v,1);
            area(p)=legacy_newell(v);projected(p)=legacy_newell(vp);
            ws{p}=repmat(area(p)/n,n,1);wh{p}=repmat(projected(p)/n,n,1);
        end
    case 'jiang_centroid_fan'
        for p=1:np
            v=nodes(m.panels{p},:);vp=v;vp(:,3)=0;
            ws{p}=fan_shares(v);wh{p}=fan_shares(vp);
            area(p)=sum(ws{p});projected(p)=sum(wh{p});
        end
        % Directly verify against the unchanged bundled Jiang function.
        plane=nodes;plane(:,3)=0;
        td=GetTributaryArea(nodes,m.panels);tw=GetTributaryArea(plane,m.panels);
        assert(all(m.q1==m.q1(1))&&all(m.q2==m.q2(1)));
        actualLegacyF=zeros(3*nn,1);actualLegacyF(3:3:end)=-m.q1(1)*td+windSign*m.q2(1)*tw;
        legacyAudit=struct('called',true,'function','GetTributaryArea','fixedPanelCount',np);
    otherwise,error('experiment:Method','Unknown method %s',method)
end
for p=1:np
    ids=m.panels{p};f=zeros(numel(ids),3);
    f(:,3)=-m.q1(p)*ws{p}+windSign*m.q2(p)*wh{p};
    nodal(ids,:)=nodal(ids,:)+f;panelF{p}=f;
end
F=reshape(nodal',[],1);
if strcmp(method,'jiang_centroid_fan')
    legacyAudit.maximumForceDifferenceFromFixedPanelReplica=max(abs(F-actualLegacyF));
    assert(legacyAudit.maximumForceDifferenceFromFixedPanelReplica<1e-9,'experiment:LegacyForce','Exact legacy load differs');
    F=actualLegacyF;nodal=reshape(F,3,[])';
end

info=struct('method',method,'panelAreas',area,'panelProjectedAreas',projected, ...
    'surfaceNodalAreas',{ws},'projectedNodalAreas',{wh},'panelNodalForces',{panelF}, ...
    'nodalForces',nodal,'totalForce',sum(nodal,1),'totalMoment',sum(cross(nodes,nodal,2),1), ...
    'legacyAudit',legacyAudit,'integratedReferencePanelResultants',base.panelResultants, ...
    'integratedReferencePanelMoments',base.panelMoments, ...
    'integratedReferenceAreas',base.panelAreas,'integratedReferenceProjectedAreas',base.panelProjectedAreas);
end

function result=compact(b)
names={'panelAreas','panelProjectedAreas','surfaceNodalAreas','projectedNodalAreas', ...
    'panelNodalForces','nodalForces','totalForce','totalMoment','panelResultants','panelMoments'};
for i=1:numel(names),result.(names{i})=b.(names{i});end
result.integratedReferencePanelResultants=b.panelResultants;
result.integratedReferencePanelMoments=b.panelMoments;
result.integratedReferenceAreas=b.panelAreas;
result.integratedReferenceProjectedAreas=b.panelProjectedAreas;
end

function A=legacy_newell(V)
% Same operation order as polygon_area3D in the unchanged original source.
K=size(V,1);Nx=0;Ny=0;Nz=0;
for i=1:K
    j=i+1;if j>K,j=1;end
    Xi=V(i,1);Yi=V(i,2);Zi=V(i,3);Xj=V(j,1);Yj=V(j,2);Zj=V(j,3);
    Nx=Nx+(Yi-Yj)*(Zi+Zj);Ny=Ny+(Zi-Zj)*(Xi+Xj);Nz=Nz+(Xi-Xj)*(Yi+Yj);
end
A=.5*norm([Nx Ny Nz]);
end

function w=legacy_angles(V)
% Exact optional legacy 'voronoi' branch: normalized corner angles, not Voronoi cells.
K=size(V,1);ang=zeros(K,1);
for i=1:K
    im=i-1;if im<1,im=K;end;ip=i+1;if ip>K,ip=1;end
    a=V(im,:)-V(i,:);b=V(ip,:)-V(i,:);
    ang(i)=real(acos(max(-1,min(1,dot(a,b)/(norm(a)*norm(b)+eps)))));
end
w=ang/sum(ang);
end

function w=fan_shares(V)
% Exact endpoint half-shares of centroid-edge triangles used by Jiang.
K=size(V,1);center=mean(V,1);w=zeros(K,1);
for i=1:K
    j=mod(i,K)+1;half=norm(cross(V(i,:)-center,V(j,:)-center)/2)/2;
    w(i)=w(i)+half;w(j)=w(j)+half;
end
end
