load fluidtemp x y z temp  
Tensor = log(load ('test12_data/tdoa1/G2.mat').G2+1);

% 降采样因子
r = 2;


% 初始化降采样后的张量
downsampled_tensor = zeros(floor(size(Tensor, 1)/r), floor(size(Tensor, 2)/r), size(Tensor, 3));

% 对每个切片进行降采样
for i = 1:size(Tensor, 3)
    downsampled_tensor(:, :, i) = imresize(Tensor(:, :, i), 1/r, 'nearest');
end



xslice = [1 7 size(downsampled_tensor,1)];                               % define the cross sections to view
yslice = [1 ];
zslice = [1 8];
% slice(x, y, z, temp, xslice, yslice, zslice)    % display the slices
slice(downsampled_tensor, xslice, yslice, zslice)    % display the slices
% ylim([-3 3])
% view(-34,24)
colormap('jet');
cb = colorbar('Ticks',[]);                                  % create and label the colorbar
cb.Label.String = 'Positioning Error';
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
set(gca,'ztick',[],'zticklabel',[])