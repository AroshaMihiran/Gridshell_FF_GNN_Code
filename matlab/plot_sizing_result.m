function plot_sizing_result(h,design,filename)
%PLOT_SIZING_RESULT Final physical-member dimensions and checked utilization.
x=h.nodes;x(:,3)=-x(:,3);b=h.bars;
f=figure('Visible','off','Color','w','Position',[100 100 1150 460]);cleanup=onCleanup(@()close(f));
tiledlayout(f,1,2,'TileSpacing','compact');values={design.final.sides*1000,design.final.memberRatios};
labels={'Square-section side (mm)','Governing checked utilization'};
for k=1:2
    ax=nexttile;hold(ax,'on');set(ax,'Color','w','XColor','k','YColor','k','ZColor','k','GridColor',[.7 .7 .7]);v=values{k};cmap=parula(256);lo=min(v);hi=max(v);range=max(hi-lo,eps);
    for e=1:size(b,1)
        a=x(b(e,:),:);index=max(1,min(256,1+round(255*(v(e)-lo)/range)));
        plot3(ax,a(:,1),a(:,2),a(:,3),'-','Color',cmap(index,:),'LineWidth',1.5);
    end
    colormap(ax,cmap);clim(ax,[lo max(hi,lo+eps(max(1,abs(lo))))]);cb=colorbar(ax);cb.Color='k';
    axis(ax,'equal');grid(ax,'on');view(ax,35,24);title(ax,labels{k},'Color','k');xlabel(ax,'x (m)','Color','k');ylabel(ax,'y (m)','Color','k');zlabel(ax,'z (m)','Color','k');
end
exportgraphics(f,filename,'Resolution',180,'BackgroundColor','white');
end
