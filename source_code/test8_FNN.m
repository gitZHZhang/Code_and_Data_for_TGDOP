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

%% VS神经网络（成功）
% 1-生成神经网络训练数据集
%把张量中已知测量值的向量索引对应到张量行列高索引上
[row_indices, col_indices, z_indices] = ind2sub(size(incomplete_T), unselected_elements);%注意这里的row指引的是y轴，col是x轴
meas_matrix = [x_span(col_indices)'+0.01, y_span(row_indices)'+0.01, z_span(z_indices)'+0.001,incomplete_T(unselected_elements)'];%[x,y,z,G]
meas_matrix = [repmat(sensor(:)',  length(unselected_elements),1),meas_matrix];
%%%%%%%%%%%%%%%%%%%%%%%% 用来检验是否能够准确对应，可以注释掉
% err = 0;
% for i=1:1:size(meas_matrix,1)
%     m=meas_matrix(i,1);
%     n=meas_matrix(i,2);
%     p=meas_matrix(i,3);
%     emitter = [m,n,p];
%     [gdop_ji,gdop2_TDOA_ji,gdop2_S_ji,sigma2_x_ji,sigma2_y_ji,sigma2_z_ji,g_ji] = cal_gdop2(emitter,sensor);
%     err = err + abs(g_ji-meas_matrix(i,4));
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2-加载神经网络训练好的模型对全局的预测
load test8_data/FNN/res_tensor.mat
res_tensor = res_tensor(2:end,:);
target_data = res_tensor(:,end-3:end);
%由于循环次数是x-y-z，现将其转换为z-y-x为了更方便的转成张量表示%%%%%%%%%%%%%
dim = size(G2);
[target_data_2] = SortRowsInXYZ(target_data,dim);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
target_G = target_data_2(:,end);
res_tensor_3d = reshape(target_G,dim);
res_tensor_3d_1 = res_tensor_3d(:,:,1);
figure();
[c,handle]=contour(x_span,y_span,res_tensor_3d_1',20);
clabel(c,handle);
title('The reconstructed distribution of $G$ using FNN','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
disp(frob(G2(:,:,1:11)-res_tensor_3d))
% plot_heat_map(res_tensor_3d_1,fs)
% plot_heat_map(G2(:,:,1),fs)
frob_list = [];
    for i = 1:1:dim(3)
        frob_list = [frob_list,frob(res_tensor_3d(:,:,i)-G2(:,:,i))];
    end

