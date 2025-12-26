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
%% SVT
% incomplete_T = G2;
% ALL_ELE = 1:numel(incomplete_T);
% NAN_ratio = 0.9;
% CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
% load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
% unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
frob_list = [];
recon_tensor = zeros(size(G2));
for i = 1:1:size(G2,3)
    M = G2(:,:,i);
    [n1,n2] = size(M);
    p = 1-NAN_ratio;
    Omega = unselected_elements((i-1)*n1*n2+1<unselected_elements & ...
        unselected_elements<i*n1*n2);
    Omega = Omega - (i-1)*n1*n2;
    data = M(Omega);
    tau = 5*sqrt(n1*n2);
    delta = 0.1/p;
    maxiter = 5000;
    tol = 1e-5;
    [U,S,V,numiter] = SVT([n1 n2],Omega,data,tau,delta,maxiter,tol);
    toc
    X = U*S*V';
    figure();
    [c,handle]=contour(x_span,y_span,X,20);
    clabel(c,handle);
    title('The reconstructed distribution of $G$ using SVT','interpreter','latex');
    xlabel('x(km)');
    ylabel('y(km)');
    figure();
    [c,handle]=contour(x_span,y_span,G2(:,:,i),20);
    clabel(c,handle);
    title('The real distribution of $G$','interpreter','latex');
    xlabel('x(km)');
    ylabel('y(km)');
    % plot_heat_map(X,fs)
    % plot_heat_map(G2(:,:,i),fs)
    frob_list = [frob_list,frob(X-G2(:,:,i))];
    recon_tensor(:,:,i) = X;
end
