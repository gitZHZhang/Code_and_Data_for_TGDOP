function plot_heat_map(target_Matrix,fs)
% 画出matrix对应的热图
[x,y] = meshgrid(-399:fs:400);
z = target_Matrix;
figure;
surf(x,y,z);
colorbar
figure;
pcolor(x,y,z);
colorbar
% color=[131 0 162;160 0 198;109 0 219;31 60 249;0  160 230;0 198 198;0 209 139;0 219 0;160 230  51;230 219 51;230 175 45;239 130 41;239 0 0;219  0 98;255 1 118]/255;
% tick=linspace(0,300,size(color,1)+1);
% mode='v';
% 
% %----------------------------------------------------------
% %lon lat data_9是绘图数据</font>
% [ax1,c,h]=contf_line(x,y,z,tick,color,mode);
% set(gca,'fontname','Times New  Roman','fontsize',16,'fontangle','italic');
% set(c,'fontname','Times New Roman','fontsize',14)

end