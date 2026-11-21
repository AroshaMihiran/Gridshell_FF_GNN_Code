function plot_fft_result(result,filename)
%PLOT_FFT_RESULT Show physical members only, without virtual boundary bars.
f=figure('Visible','off','Color','w','Position',[100 100 1350 460]);cleanup=onCleanup(@()close(f));
layout=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
x={result.model.nodes,result.state.nodes,result.state.nodes};x{3}(:,3)=-x{3}(:,3);
labels={'Initial layout','Hanging equilibrium','Mirrored geometry'};
bars=result.model.bars;fixed=result.model.fixedNodes;
for j=1:3
    ax=nexttile(layout);hold(ax,'on');set(ax,'Color','w','XColor','k','YColor','k','ZColor','k','GridColor',[.7 .7 .7]);
    for e=1:size(bars,1),a=x{j}(bars(e,:),:);plot3(ax,a(:,1),a(:,2),a(:,3),'-','Color',[.15 .25 .35],'LineWidth',.8);end
    scatter3(ax,x{j}(fixed,1),x{j}(fixed,2),x{j}(fixed,3),18,[.65 .15 .1],'filled');
    axis(ax,'equal');grid(ax,'on');view(ax,35,24);xlabel(ax,'x (m)','Color','k');ylabel(ax,'y (m)','Color','k');zlabel(ax,'z (m)','Color','k');title(ax,labels{j},'Color','k');
end
exportgraphics(f,filename,'Resolution',180,'BackgroundColor','white');
end
