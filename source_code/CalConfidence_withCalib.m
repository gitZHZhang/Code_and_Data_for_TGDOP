function [confidence] = CalConfidence_withCalib(emitter,emitter_samples,sensor,gridsize,plot_trigger)
[gdop,~,~,P] = cal_gdop(emitter,sensor);%计算真实位置的协方差
n=size(emitter_samples,1);%定位次数
% 确定网格范围
gridspan = ceil(max(max(abs(emitter_samples-emitter))));

% 1-生成 x 和 y 的网格
x = linspace(-gridspan, gridspan, gridspan*2/gridsize)+emitter(1);
y = linspace(-gridspan, gridspan, gridspan*2/gridsize)+emitter(2);
[X, Y] = meshgrid(x, y);

% 2-给出定位结果所在网格以及这些网格的边界
[targetGrid] = searchGrid(emitter,X, Y); %真实位置所在的网格
posGrid_list = []; %定位结果们所在的网格列表
for i =1:1:size(emitter_samples,1)
    posGrid_list = [posGrid_list;searchGrid(emitter_samples(i,:),X, Y)];
end
posGrid_list_xmin = posGrid_list(:,1)-(X(1,2)-X(1,1))/2;
posGrid_list_xmax = posGrid_list(:,1)+(X(1,2)-X(1,1))/2;
posGrid_list_ymin = posGrid_list(:,2)-(Y(2,1)-Y(1,1))/2;
posGrid_list_ymax = posGrid_list(:,2)+(Y(2,1)-Y(1,1))/2;
targetGrid_xmin = targetGrid(1)-(X(1,2)-X(1,1))/2;
targetGrid_xmax = targetGrid(1)+(X(1,2)-X(1,1))/2;
targetGrid_ymin = targetGrid(2)-(Y(2,1)-Y(1,1))/2;
targetGrid_ymax = targetGrid(2)+(Y(2,1)-Y(1,1))/2;
% 2-给出真实位置的概率密度函数（注意不是真实位置对应的网格）
[f1,~,~,~]=GetGaussFunction(P,emitter);
for row_i = 1:1:size(X,1)
    for col_i = 1:1:size(X,2)
        Z(row_i,col_i)=f1(X(row_i,col_i),Y(row_i,col_i));
    end
end
if (plot_trigger)
    figure
    surf(X,Y,Z)
    shading interp
    colorbar;
end

% I = integral2(f1, 99, 101, 99, 101); % f, xmin, xmax, ymin, ymax
% 3-根据概率密度函数给出概率
% 真实值所在网格的概率
xmin = targetGrid_xmin;xmax = targetGrid_xmax;
ymin = targetGrid_ymin;ymax = targetGrid_ymax;
I_target = integral2(f1, xmin, xmax, ymin, ymax); %真实值所在网格为真实定位结果的概率P=0.0767，因为网格很小，而且网格中心也不是真实位置（有偏）
I_target2 = integral2(f1, emitter(1)-0.8, emitter(1)+0.8, emitter(2)-0.8, emitter(2)+0.8); %以真值为中心，扩大网格后，P=0.8195
% 各个定位结果所在网格的概率
for i=1:1:size(emitter_samples,1)
    xmin = posGrid_list_xmin(i);xmax = posGrid_list_xmax(i);
    ymin = posGrid_list_ymin(i);ymax = posGrid_list_ymax(i);
    I_i = integral2(f1, xmin, xmax, ymin, ymax);
    confidence(i) = I_i/I_target;
end