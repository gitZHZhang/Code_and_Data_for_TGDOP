
function [ax1,c,h]=contf_line(lon,lat,X,tick,color,mode)
% ---------------------------------
%contourf()绘图，带尖端的colorbar
%[ax1,c,h]=CONTF_LINE(lon,lat,X,tick,color,mode)
% 
%tick的个数比color的行数多一个
%ax1控制绘图窗，c控制colorbar,h控制contourf函数
%--------
%修改colorbar间隔：
%lab=get(c,'x/yticklabel');
%lab(2:2:end)={' '};
%set(c,'x/yticklabel',lab);
%--------
%copyright 傅辉煌
% ----------------------------------------


[m,~]=size(color);
len_tick=length(tick);
if len_tick-m ~= 1
    return
end
[ax1,c]=colorbarn(tick,color,mode);
[~,h]=contourf(lon,lat,X);
set(h,'levellist',tick,'linecolor','none');
colormap(color);
caxis([tick(1) tick(end)])
end


