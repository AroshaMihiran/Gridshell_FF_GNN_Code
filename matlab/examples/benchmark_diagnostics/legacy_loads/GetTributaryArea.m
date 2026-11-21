function [TribArea]=GetTributaryArea(NODE,ELEM)
%   [TribArea]=GetTributaryArea(NODE,ELEM) where NODE is an Nn x 3 vector
%   with the Nn nodes and ELEM is a cell variable where each row contains the numbers of the nodes surrounding a shell panel. 
%   Get the tributary area associated with each node

Nn=size(NODE,1);
Ne =size(ELEM,1);
TribArea=zeros(Nn,1);
for i=1:Ne
     Panel=ELEM{i,1};
    % Compute element area and center of mass 
    for j=1:size(Panel,2)
     N(j,:) = NODE(Panel(1,j),:);
    end
     CG = mean(N);
     Aelem=0; CMA=0;
     for j=1:size(Panel,2)
         if j==size(Panel,2)
            A(j,:) = cross(N(j,:)-CG, N(1,:)-CG)/2;
            CM(j,:) = mean([CG; N(j,:); N(1,:)]);
         else
            A(j,:)= cross(N(j,:)-CG, N(j+1,:)-CG)/2;
            CM(j,:) = mean([CG; N(j,:); N(j+1,:)]);
         end
         if A(j,3)<0
             Area(j) = -norm(A(j,:));
         else
             Area(j)= norm(A(j,:));
         end
         
         CMA= CM(j)*Area(j);
         Aelem = Aelem+Area(j);
     end
     CM = CMA/Aelem;
    % Compute nodal tributary area
     for j=1:size(Panel,2)
         if j==size(Panel,2)
            TribArea(Panel(1,j)) = TribArea(Panel(1,j))+(norm(cross(N(j,:)-CG, N(1,:)-CG)/2)/2);
            TribArea(Panel(1,1)) = TribArea(Panel(1,1))+(norm(cross(N(j,:)-CG, N(1,:)-CG)/2)/2);
         else
             TribArea(Panel(1,j)) = TribArea(Panel(1,j))+(norm(cross(N(j,:)-CG, N(j+1,:)-CG)/2)/2);
             TribArea(Panel(1,j+1)) = TribArea(Panel(1,j+1))+(norm(cross(N(j,:)-CG, N(j+1,:)-CG)/2)/2);
         end
     end
     
 clear N A Area CM
end
% 