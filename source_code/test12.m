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
            G2(j,i,k) = G(fs*(i-1)+1,fs*(j-1)+1,k);
            Sigma2_x(j,i,k) = sigma2_x(fs*(i-1)+1,fs*(j-1)+1,k);
            Sigma2_y(j,i,k) = sigma2_y(fs*(i-1)+1,fs*(j-1)+1,k);
            Sigma2_z(j,i,k) = sigma2_z(fs*(i-1)+1,fs*(j-1)+1,k);

        end
    end
end
% figure
% mlrankest(Sigma2_x)
% mlrankest(Sigma2_y)
% mlrankest(Sigma2_z)
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

CHOSEN_IDX_LIST = [];
Mont_times = 50;
NAN_ratio = 0.99; %80 85 90 93 95 97 99
for i=1:1:Mont_times
    CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round(NAN_ratio*numel(G2)))];
end

%% 和别的算法作比较:Proposed/FNN/kriging/SVT/NNM-T
Algorithm_list = {'Proposed'};
ContrastAlgorithm=Algorithm_list{1};
frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差  
save_path = ['test8_data/',ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/'];
min_frob_list = 1e8;%当froblist均值最小时存储相应的重构张量
%% 按照比例抽取部分G作为观测量
incomplete_T = G2+0.01*randn(size(G2));%0.01km标准差的噪声
incomplete_Tx = Sigma2_x+0.01*randn(size(Sigma2_x));%0.01km标准差的噪声
incomplete_Ty = Sigma2_y+0.01*randn(size(Sigma2_y));%0.01km标准差的噪声
incomplete_Tz = Sigma2_z+0.01*randn(size(Sigma2_z));%0.01km标准差的噪声
ALL_ELE = 1:numel(incomplete_T);
CHOSEN_IDX = CHOSEN_IDX_LIST(1,:);
unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
incomplete_T(CHOSEN_IDX) = NaN; %将G2中的值设置为未知
incomplete_Tx(CHOSEN_IDX) = NaN;
incomplete_Ty(CHOSEN_IDX) = NaN;
incomplete_Tz(CHOSEN_IDX) = NaN;
incomplete_X = outprod(incomplete_Tx,[1 0 0])+outprod(incomplete_Ty,[0 1 0])+outprod(incomplete_Tz,[0 0 1]);


%画出观测数据在全部区域的分布
a = zeros(1,numel(incomplete_T));
a(unselected_elements)=1;
a_3d = reshape(a,size(G2,1),size(G2,2),size(G2,3));
% plot_title = 'The 3-D distribution of the selected grid.';
% [~] = plot_3D_tensor(a_3d,x_span,y_span,z_span,plot_title);

% BTD
incomplete_T2x = fmt(incomplete_Tx);
incomplete_T2y = fmt(incomplete_Ty);
incomplete_T2z = fmt(incomplete_Tz);
size_tens = incomplete_T2x.size;
% size_tens = size(incomplete_T);
L1 = [10 10 3];
L2 = [10 10 3];
L3 = [10 10 3];
L1 = [5 5 3];
L2 = [6 6 3];
L3 = [7 7 3];
model= struct;

%% 列为m-1阶多项式初值的因子矩阵
m=3;
model.variables.A1=randn(L1(1),m);
model.variables.B1=randn(L1(2),m);
model.variables.C1=randn(L1(3),m);
model.variables.S1=randn(L1(1),L1(2),L1(3));
model.variables.A2=randn(L2(1),m);
model.variables.B2=randn(L2(2),m);
model.variables.C2=randn(L2(3),m);
model.variables.S2=randn(L2(1),L2(2),L2(3));
model.variables.A3=randn(L3(1),m);
model.variables.B3=randn(L3(2),m);
model.variables.C3=randn(L3(3),m);
model.variables.S3=randn(L3(1),L3(2),L3(3));

t1=1:1:size_tens(1);
t2=1:1:size_tens(3);
model.factors={ {'A1',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
    {'B1',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
    {'C1',  @(z,task) struct_poly(z,task,t2),@struct_nonneg},...
    {'S1',@struct_nonneg},...
    {'A2',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
    {'B2',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
    {'C2',  @(z,task) struct_poly(z,task,t2),@struct_nonneg},...
    {'S2',@struct_nonneg},...
    {'A3',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
    {'B3',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
    {'C3',  @(z,task) struct_poly(z,task,t2),@struct_nonneg},...
    {'S3',@struct_nonneg} };

model.factorizations.mybtd1.data=incomplete_T2x;
model.factorizations.mybtd1.btd={{1,2,3,4}};
model.factorizations.mybtd2.data=incomplete_T2y;
model.factorizations.mybtd2.btd={{5,6,7,8}};
model.factorizations.mybtd3.data=incomplete_T2z;
model.factorizations.mybtd3.btd={{9,10,11,12}};

sdf_check(model,'print');
[sol,output] = sdf_nls(model);
[A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res,A3_res,B3_res,C3_res,S3_res] = deal(sol.factors{:});
Sigma2_x_hat = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
Sigma2_y_hat = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
Sigma2_z_hat = tmprod(S3_res,{A3_res,B3_res,C3_res},1:3);
recon_tensor1 = Sigma2_x_hat + Sigma2_y_hat + Sigma2_z_hat;
frob_list1 = [];
for i = 1:1:size(recon_tensor1,3)
    frob_list1 = [frob_list1,frob(recon_tensor1(:,:,i)-G2(:,:,i))];
end


figure();
subplot(2,1,1)
[c,handle]=contour(x_span,y_span,recon_tensor1(:,:,1),20);
clabel(c,handle);
subplot(2,1,2)
[c,handle]=contour(x_span,y_span,G2(:,:,1),20);
clabel(c,handle);
figure();
subplot(2,1,1)
[c,handle]=contour(x_span,y_span,Sigma2_x_hat(:,:,1),20);
clabel(c,handle);
subplot(2,1,2)
[c,handle]=contour(x_span,y_span,Sigma2_x(:,:,1),20);
clabel(c,handle);
figure();
subplot(2,1,1)
[c,handle]=contour(x_span,y_span,Sigma2_y_hat(:,:,1),20);
clabel(c,handle);
subplot(2,1,2)
[c,handle]=contour(x_span,y_span,Sigma2_y(:,:,1),20);
clabel(c,handle);
figure();
subplot(2,1,1)
[c,handle]=contour(x_span,y_span,Sigma2_z_hat(:,:,1),20);
clabel(c,handle);
subplot(2,1,2)
[c,handle]=contour(x_span,y_span,Sigma2_z(:,:,1),20);
clabel(c,handle);
% title('Real distribution of $G$','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');
% % leg = title('$\sigma_{y}^2的真实分布（降采样后）$','interpreter','latex');
% % set(leg,'Interpreter','latex')
% subplot(2,1,2)
% [c,handle]=contour(x_span,y_span,recon_tensor(:,:,1),20);
% clabel(c,handle);
% title('Reconstructed distribution of $G$','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');
% figure();
% subplot(3,2,1)
% [c,handle]=contour(x_span,y_span,Sigma2_x(:,:,11),20);
% % clabel(c,handle);
% title('Real distribution of $G_x$.','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');
% subplot(3,2,2)
% [c,handle]=contour(x_span,y_span,Sigma2_x_hat(:,:,11),20);
% % clabel(c,handle);
% title('Reconstructed distribution of $G_x$.','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');
% subplot(3,2,3)
% [c,handle]=contour(x_span,y_span,Sigma2_y(:,:,11),20);
% % clabel(c,handle);
% title('Real distribution of $G_y$.','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');
% subplot(3,2,4)
% [c,handle]=contour(x_span,y_span,Sigma2_y_hat(:,:,11),20);
% % clabel(c,handle);
% title('Reconstructed distribution of $G_y$.','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');
% subplot(3,2,5)
% [c,handle]=contour(x_span,y_span,Sigma2_z(:,:,11),20);
% % clabel(c,handle);
% title('Real distribution of $G_z$.','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');
% subplot(3,2,6)
% [c,handle]=contour(x_span,y_span,Sigma2_z_hat(:,:,11),20);
% % clabel(c,handle);
% title('Reconstructed distribution of $G_z$.','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');