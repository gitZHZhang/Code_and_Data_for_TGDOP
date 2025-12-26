function [targetGrid] = searchGrid(target_cor,GridsX,GridsY)
%% 找出某坐标在给定网格内的位置
gridsX = unique(GridsX);
gridsY = unique(GridsY);
targetX = target_cor(1);
targetY = target_cor(2);
indicesX = find(gridsX <= targetX);
indicesY = find(gridsY <= targetY);
% 如果坐标不在网格边界上，就找一个最近的网格点，在边界上就取最近的边界点
resX = 0; resY = 0;
if length(indicesX)==length(gridsX)
    grid_i = indicesX(end);
    grid_j = 1;
elseif isempty(indicesX)
    grid_i = 1;
    grid_j = length(gridsX);
else
    grid_i = indicesX(end);
    grid_j = indicesX(end)+1;
end
if abs(gridsX(grid_i)-targetX)>abs(gridsX(grid_j)-targetX)
    resX = gridsX(grid_j);
else
    resX = gridsX(grid_i);
end

if length(indicesY)==length(gridsY)
    grid_i = indicesY(end);
    grid_j = 1;
elseif isempty(indicesY)
    grid_i = 1;
    grid_j = length(gridsY);
else
    grid_i = indicesY(end);
    grid_j = indicesY(end)+1;
end
if abs(gridsY(grid_i)-targetY)>abs(gridsY(grid_j)-targetY)
    resY = gridsY(grid_j);
else
    resY = gridsY(grid_i);
end
targetGrid = [resX,resY]; %所属网格的中心点