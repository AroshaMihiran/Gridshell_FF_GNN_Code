function section = frame3d_section(b,h)
%FRAME3D_SECTION Properties of a solid rectangular frame section in metres.
% b is width parallel to local 3; h is depth parallel to local 2.
% I2 is therefore h*b^3/12, I3=b*h^3/12; W2=I2/(b/2).
% Shear areas are 5A/6 (uniform rectangle, energy-equivalent shear factor).
% J uses the Saint-Venant odd-term series, not the polar second moment.
% ETABS automatic J can differ slightly; import its actual J when validating
% a saved ETABS model rather than silently changing either representation.
validateattributes(b,{'numeric'},{'real','finite','positive','vector','nonempty'});
validateattributes(h,{'numeric'},{'real','finite','positive','vector','nonempty'});
b=double(b(:));h=double(h(:));
n=max(numel(b),numel(h));
if isscalar(b),b=repmat(b,n,1);end
if isscalar(h),h=repmat(h,n,1);end
if numel(b)~=numel(h),error('frame3d:SectionSize','b and h must have compatible lengths.');end
A=b.*h;I2=h.*b.^3/12;I3=b.*h.^3/12;
a=max(b,h);t=min(b,h);odd=1:2:199;
series=sum(tanh((pi/2)*(a./t)*odd)./(odd.^5),2);
J=a.*t.^3/3.*(1-(192/pi^5)*(t./a).*series);
section=struct('b',b,'h',h,'A',A,'I2',I2,'I3',I3,'J',J, ...
    'W2',2*I2./b,'W3',2*I3./h,'As2',5*A/6,'As3',5*A/6, ...
    'shape','solid_rectangle','torsionModel','Saint-Venant series, first 100 positive odd terms', ...
    'axes','h along local 2; b along local 3','shearAreaModel','5A/6 each direction');
end
