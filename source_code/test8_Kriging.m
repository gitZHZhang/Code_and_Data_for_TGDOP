close all
clear all



%% 加载原始数据（和test4的数据相同）
load G.mat 
load sigma2_x.mat
load sigma2_y.mat 
load sigma2_z.mat

fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;z_span = linspace(10,20,11);
for i=1:1:size(G,1)/fs
    for j=1:1:size(G,2)/fs
        for k=1:1:size(G,3)
            G2(j,i,k) = G(fs*i,fs*j,k);
            Sigma2_x(j,i,k) = sigma2_x(fs*i,fs*j,k);
            Sigma2_y(j,i,k) = sigma2_y(fs*i,fs*j,k);
            Sigma2_z(j,i,k) = sigma2_z(fs*i,fs*j,k);
        end
    end
end
% figure
% mlrankest(Sigma2_y)
% size_core = mlrankest(Sigma2_x);%计算sizecore
% size_core = [3,3,2];
% [UXhat,SXhat,output]=lmlra(Sigma2_x,size_core); 
% [UYhat,SYhat,output]=lmlra(Sigma2_y,size_core); 
% size_core = [4,4,2];
% [UZhat,SZhat,output]=lmlra(Sigma2_z,size_core); 


%% 传感器位置
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
x3 = 0;y3 = -L;z3 = 0.3;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];
%% 按照比例抽取部分G作为观测量
incomplete_T = G2;
ALL_ELE = 1:numel(incomplete_T);
NAN_ratio = 0.99;
% CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
%画出观测数据在全部区域的分布
a = zeros(1,numel(incomplete_T));
a(unselected_elements)=1;
a_3d = reshape(a,size(G2,1),size(G2,2),size(G2,3));
plot_title = 'The 3-D distribution of the selected grid.';
[~] = plot_3D_tensor(a_3d,x_span,y_span,z_span,plot_title);
incomplete_T(CHOSEN_IDX) = NaN; %将G2中96%的值设置为未知


%% 开始和别的算法作比较:Proposed/AI/kriging/SVT
%% VS kriging（成功）
[row_indices, col_indices, z_indices] = ind2sub(size(incomplete_T), unselected_elements);%注意这里的row指引的是y轴，col是x轴
meas_matrix = [x_span(col_indices)'+0.01, y_span(row_indices)'+0.01, z_span(z_indices)'+0.001,incomplete_T(unselected_elements)'];%[x,y,z,G]
z_indices_unique = unique(z_indices);%按照不同的高度（z）分别处理，每一个z类似于一个矩阵，分别应用矩阵处理算法
indices_cell = cell(1,length(z_indices_unique));
frob_list = [];
recon_tensor = zeros(size(G2));
for i =1:1:length(z_indices_unique)
    indices_cell{i} = z_indices==z_indices_unique(i);
    meas_matrix_i = meas_matrix(indices_cell{i},:);
    %S存储了点位坐标值，Y为观测值
    S = meas_matrix_i(:,1:2);
    Y = meas_matrix_i(:,4);
    theta = [10 10]; lob = [1e-1 1e-1]; upb = [20 20];
    %变异函数模型为高斯模型
    [dmodel, ~] = dacefit(S, Y, @regpoly2, @corrgauss, theta, lob, upb);

    X = gridsamp([-400 -400;400 400], size(G2,1));%创建一个szie*size的正方形格网，标注范围为-400,400
    % X=[83.731	32.36];     %单点预测的实现
    %格网点的预测值返回在矩阵YX中，预测点的均方根误差返回在矩阵MSE中
    [YX,MSE] = predictor(X, dmodel);
    X1 = reshape(X(:,1),size(G2,1),size(G2,1)); X2 = reshape(X(:,2),size(G2,1),size(G2,1));
    YX = reshape(YX, size(X1));         %size(X1)=400*400
    figure(1), mesh(X1, X2, YX)         %绘制预测表面
    hold on,
    plot3(S(:,1),S(:,2),Y,'.k', 'MarkerSize',10)    %绘制原始散点数据
    hold off
    figure(2),mesh(X1, X2, reshape(MSE,size(X1)));  %绘制每个点的插值误差大小
    figure();
    [c,handle]=contour(X1(1,:),X2(:,1),YX,20);
    clabel(c,handle);
    title('The reconstructed distribution of $G$ using Kriging','interpreter','latex');
    xlabel('x(km)');
    ylabel('y(km)');
    % disp(frob(G2(:,:,i)-YX))
    recon_tensor(:,:,i) = YX;
    frob_list = [frob_list,frob(YX-G2(:,:,i))];
end
 