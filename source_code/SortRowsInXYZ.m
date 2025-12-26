function [out] = SortRowsInXYZ(target_data,dim)
% target_data数据格式：m*4,其中target_data(:,1:3)表示(x,y,z)，target_data(4)表示在(x,y,z)处的值
% 循环先后顺序是x-y-z，现将其转换为z-y-x
sort_column = 3;
target_data_1 = sortrows(target_data, sort_column);%z维度的处理
target_data_2 = [];
for i=1:1:dim(3)
    target_data_1_slic = target_data_1(dim(1)*dim(2)*(i-1)+1:dim(1)*dim(2)*i,:);
    sort_column = 2;
    target_data_2 = [target_data_2;sortrows(target_data_1_slic, sort_column)];%y维度的处理
end
out = target_data_2;