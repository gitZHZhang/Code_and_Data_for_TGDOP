function plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)
contour_values = 10;
hs1 = slice(plot_tensor,[],[],slice_idx) ;
shading interp
set(hs1,'FaceAlpha',0.8);
colormap('jet');
% colormap parula(8)
% colormap hsv

% colormap(sky)
cb = colorbar;
cb.Label.String = 'Positioning Error (dB)';
hold on;
[c,handle]=contour(1:size(plot_tensor,1),1:size(plot_tensor,2),plot_tensor(:,:,1),contour_values);
title(title_str,'interpreter','latex')

xmin = floor(x_span(1)/100)*100;
xmax = ceil(x_span(end)/100)*100;
x_tick_label = linspace(xmin,xmax,5);
func_x = @(x) ((x-2*x_span(1)+x_span(2))/(x_span(2)-x_span(1))); 
x_tick = func_x(x_tick_label);
ymin = floor(y_span(1)/100)*100;
ymax = ceil(y_span(end)/100)*100;
y_tick_label = linspace(ymin,ymax,5);
func_y = @(x) ((x-2*y_span(1)+y_span(2))/(y_span(2)-y_span(1))); 
y_tick = func_y(y_tick_label);
set(gca,'XTick',x_tick);%设置要显示坐标刻度
set(gca,'XTickLabel',x_tick_label);%给坐标加标签
set(gca,'YTick',y_tick);%设置要显示坐标刻度
set(gca,'YTickLabel',y_tick_label);%给坐标加标签
set(gca,'ZTick',slice_idx);%设置要显示坐标刻度
set(gca,'ZTickLabel',z_span(slice_idx));%给坐标加标签
xlabel('x (m)');
ylabel('y (m)');
zlabel('z (m)');
xlim([x_tick(1),x_tick(end)])
ylim([y_tick(1),y_tick(end)])

% x_y_z1 = 0:size(plot_tensor,1)/4:size(plot_tensor,1);
% x_y_z1(1)=1;
% x_y_z2 = -400:20*(x_span(2)-x_span(1)):400;
% 
% set(gca,'XTick',x_y_z1);%设置要显示坐标刻度
% set(gca,'XTickLabel',x_y_z2);%给坐标加标签
% set(gca,'YTick',x_y_z1);%设置要显示坐标刻度
% set(gca,'YTickLabel',x_y_z2);%给坐标加标签
% set(gca,'ZTick',slice_idx);%设置要显示坐标刻度
% set(gca,'ZTickLabel',z_span(slice_idx));%给坐标加标签
% xlabel('x(km)');
% ylabel('y(km)');
% zlabel('z(km)');