function state = gs_internal_state(nodes,bars,E,A,Lref)
%GS_INTERNAL_STATE Axial truss state with a fixed, explicit stress-free length.
% Units: m, kN, kN/m^2. Positive force/stress denotes tension. Global vectors
% use x1,y1,z1,x2,... ordering. internalForce is the strain-energy gradient.
validateattributes(nodes,{'numeric'},{'2d','ncols',3,'real','finite','nonempty'});
validateattributes(bars,{'numeric'},{'2d','ncols',2,'integer','positive','finite','nonempty'});
if any(bars(:)>size(nodes,1)) || any(bars(:,1)==bars(:,2))
    error('GridShell:Connectivity','Invalid member endpoint rows.');
end
nm = size(bars,1);
E = memberVector(E,nm,'E');
A = memberVector(A,nm,'A');
Lref = memberVector(Lref,nm,'Lref');
delta = nodes(bars(:,2),:)-nodes(bars(:,1),:);
lengths = sqrt(sum(delta.^2,2));
if any(lengths <= 1e-12*max(1,max(Lref)))
    error('GridShell:CollapsedMember','A current member has collapsed to zero length.');
end
strain = (lengths-Lref)./Lref;
stress = E.*strain;
force = A.*stress;
endForce = delta.*(force./lengths);
nodalForce = zeros(size(nodes));
for d = 1:3
    nodalForce(:,d) = accumarray([bars(:,1);bars(:,2)], ...
        [-endForce(:,d);endForce(:,d)],[size(nodes,1),1]);
end
state = struct('nodes',nodes,'lengths',lengths,'strain',strain, ...
    'stress',stress,'force',force,'internalForce',reshape(nodalForce',[],1), ...
    'nodalInternalForce',nodalForce,'strainEnergy',sum(0.5*E.*A.*Lref.*strain.^2), ...
    'E',E,'A',A,'Lref',Lref);
end

function v = memberVector(v,n,name)
validateattributes(v,{'numeric'},{'vector','real','finite','positive'},mfilename,name);
if isscalar(v), v = repmat(v,n,1); else, v = v(:); end
if numel(v)~=n, error('GridShell:Material','%s needs one value per member.',name); end
end
