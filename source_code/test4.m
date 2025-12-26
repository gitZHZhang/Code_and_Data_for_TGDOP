close all
clear all

% size_tens =[17 19 21];
% size_core =[3 5 7];
% [U,S0]= lmlra_rnd(size_tens,size_core); % (S 3*5*7) (U1 17*3) (U2 19*5) (U3 21*7)
% T=lmlragen(U,S0);% 等价于T=tmprod(S,U,1:length(U)); 其中tmprod模n积
% Tn = noisy(T,20); %加噪，信噪比20dB
% mlrankest(Tn)
% rankest(Tn)
files = dir('*.mat');
for i=1:1:length(files)
    str = [files(i).name];
    load(str)
end
fs = 10; %降采样倍数
%% 对原始数据降采样加转置

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

%% 传感器位置
%三星双时差2维定位
L=10;
%s0中心站
xt = floor(0+size(G2,1)/2);  yt = floor(0+size(G2,2)/2);
% s1
x1 = floor(L*cos(30*pi/180)+size(G2,1)/2);
y1 = floor(L*sin(30*pi/180)+size(G2,2)/2);
% s2
x2 = floor(L*cos(150*pi/180)+size(G2,1)/2);
y2 = floor(L*sin(150*pi/180)+size(G2,2)/2);
% s3
x3 = floor(0+size(G2,1)/2);  y3 = floor(-L+size(G2,2)/2);
figure(1)
plot(x1,y1,'r.','MarkerSize',10);hold on
plot(x2,y2,'r.','MarkerSize',10);hold on
plot(x3,y3,'r.','MarkerSize',10);hold on
plot(xt,yt,'r.','MarkerSize',10);hold on

trigger = 1;
switch trigger
case 0
%% 0-秩分析
% figure()
% mlrankest(Sigma2_x)
% figure()
% rankest(Sigma2_x)
%% 1-若G仅有少量观测量，用BTD分解求解完整的G
case 1
full_T = G2;
%1-稀疏张量均匀采样
% incomplete_T = NaN(size(G2));
% for k=1:1:11
%     incomplete_T(1:400,400,k) = G2(1:400,400,k);
%     incomplete_T(400,1:400,k) = G2(400,1:400,k);
% end
% % xnum = 350; ynum=350; znum=11;
% x_interval = 1.2; y_interval = 1.2; z_interval = 1;
% xnum = ceil(size(G2,1)/x_interval); ynum=ceil(size(G2,2)/y_interval); znum=ceil(size(G2,3)/z_interval);
% % x_interval = floor(size(G2,1)/xnum); y_interval = floor(size(G2,2)/ynum); z_interval = floor(size(G2,3)/znum);
% for i=1:1:xnum
%     for j=1:1:ynum
%         for k=1:1:znum
%             x_idx = floor(x_interval*i);y_idx = floor(y_interval*j);z_idx = floor(z_interval*k);
%             if x_idx>400
%                 x_idx = 400;
%                 y_idx = 400;
%             end
%             if z_idx>11
%                 z_idx = 11;
%             end
%             incomplete_T(x_idx,y_idx,z_idx) = G2(x_idx,y_idx,z_idx);
%         end
%     end
% end
%2-稀疏张量按轨迹采样
% incomplete_T = NaN(size(G2));
% cir = 1; % 输入螺旋线的旋向（右旋为1，左旋为0）
% r0 = 10; % 螺旋线的起始半径
% cir_num = 10;%螺旋线的圈数
% L = size(G2,1)/2/cir_num; % 输入螺旋线的螺距;
% if cir == 0
%     angle = linspace(0,2*pi*cir_num,round(0.9*numel(incomplete_T)));
%     r = r0 + L*angle/(2*pi);
%     x = r.*cos(angle);                 %x-y平面图形基于阿基米德螺旋线
%     y = r.*sin(angle);
%     z = angle/pi;                      %z方向自定义
%     figure
%     plot3(x,y,z);
% elseif cir == 1
%     angle = linspace(0,-2*pi*cir_num,round(0.9*numel(incomplete_T)));
%     r = r0 + L*(-angle)/(2*pi);
%     x = r.*cos(angle);                 %x-y平面图形基于阿基米德螺旋线
%     y = r.*sin(angle);
%     z = -angle/pi;                      %z方向自定义
%     figure
%     plot3(x,y,z);
% end
% x_chosen = floor(x+size(G2,1)/2);x_chosen(x_chosen<=0)=1;x_chosen(x_chosen>size(G2,1))=size(G2,1);
% y_chosen = floor(y+size(G2,2)/2);y_chosen(y_chosen<=0)=1;y_chosen(y_chosen>size(G2,2))=size(G2,2);
% z_chosen = floor(z/max(z)*size(G2,3));z_chosen(z_chosen<=0)=1;z_chosen(z_chosen>size(G2,3))=size(G2,3);
% cordinate = [x_chosen',y_chosen',z_chosen'];
% cordinate2 = unique(cordinate,'rows','stable');
% figure
% plot3(cordinate2(:,1),cordinate2(:,2),cordinate2(:,3));
% a_3d = zeros(size(G2));
% for i =1:1:size(cordinate2,1)
%     % incomplete_T(cordinate2(i,1),cordinate2(i,2),cordinate2(i,3))=G2(cordinate2(i,1),cordinate2(i,2),cordinate2(i,3));
%     % a_3d(cordinate2(i,1),cordinate2(i,2),cordinate2(i,3))=1;
%     incomplete_T(cordinate2(i,1),cordinate2(i,2),1:11)=G2(cordinate2(i,1),cordinate2(i,2),1:11);
%     a_3d(cordinate2(i,1),cordinate2(i,2),1:11)=1;
% end
% x_span = -399:fs:400;y_span = -399:fs:400;z_span = 10:1:20;
% plot_title = 'The 3-D distribution of the selected grid.';
% [~] = plot_3D_tensor(a_3d,x_span,y_span,z_span,plot_title);



%3-稀疏张量随机采样
incomplete_T = G2;
ALL_ELE = 1:numel(incomplete_T);
CHOSEN_IDX = randperm(numel(incomplete_T),round(0.96*numel(incomplete_T)));
unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
a = zeros(1,numel(incomplete_T));
a(unselected_elements)=1;
a_3d = reshape(a,400,400,11);
x_span = -399:fs:400;y_span = -399:fs:400;z_span = 10:1:20;
plot_title = 'The 3-D distribution of the selected grid.';
[~] = plot_3D_tensor(a_3d,x_span,y_span,z_span,plot_title);
incomplete_T(CHOSEN_IDX) = NaN; %将G2中90%的值设置为未知


incomplete_T = fmt(incomplete_T);
size_tens = incomplete_T.size;
L1 = [10 10 3];
L2 = [10 10 3];
L3 = [10 10 3];
model= struct;
model.variables.A1=randn(size_tens(1),L1(1));
model.variables.B1=randn(size_tens(2),L1(2));
model.variables.C1=randn(size_tens(3),L1(3));
model.variables.S1=randn(L1(1),L1(2),L1(3));
model.variables.A2=randn(size_tens(1),L2(1));
model.variables.B2=randn(size_tens(2),L2(2));
model.variables.C2=randn(size_tens(3),L2(3));
model.variables.S2=randn(L2(1),L2(2),L2(3));
model.variables.A3=randn(size_tens(1),L3(1));
model.variables.B3=randn(size_tens(2),L3(2));
model.variables.C3=randn(size_tens(3),L3(3));
model.variables.S3=randn(L3(1),L3(2),L3(3));
model.factors={ 'A1','B1','C1','S1', 'A2','B2','C2','S2', 'A3','B3','C3','S3' };
model.factorizations.mybtd.data=incomplete_T;
model.factorizations.mybtd.btd={{1,2,3,4},{5,6,7,8},{9,10,11,12}};
sdf_check(model,'print');
[sol,output] = sdf_nls(model);
[A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res,A3_res,B3_res,C3_res,S3_res] = deal(sol.factors{:});
Sigma2_x_hat = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
Sigma2_y_hat = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
Sigma2_z_hat = tmprod(S3_res,{A3_res,B3_res,C3_res},1:3);
sum_res = Sigma2_x_hat + Sigma2_y_hat + Sigma2_z_hat;
err_T = abs(full_T-sum_res);
err_T2 = log((err_T - min(min(min(err_T)))) / max(max(max(err_T))));
x_span = -399:fs:400;y_span = -399:fs:400;z_span = 10:1:20;
plot_title='The 3-D distribution of $log(| \hat{\textbf{e}}|)$';
[~] = plot_3D_tensor(err_T2,x_span,y_span,z_span,plot_title);


frob_list = [];
for i = 1:1:size(sum_res,3)
    frob_list = [frob_list,frob(sum_res(:,:,i)-full_T(:,:,i))];
end
figure
plot(10:1:20,frob_list,'.-')
xlabel('Height(km)')
ylabel('$\Vert e \Vert_F$','interpreter','latex')
title('Tensor reconstruction error varies with height');
frob(sum_res-full_T)
figure();
subplot(2,2,1)
[c,handle]=contour(1:1:size(full_T,1),1:1:size(full_T,2),full_T(:,:,1),20);
clabel(c,handle);
title('The real distribution of $G$','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
% leg = title('$\sigma_{y}^2的真实分布（降采样后）$','interpreter','latex');
% set(leg,'Interpreter','latex')
subplot(2,2,2)
[c,handle]=contour(1:1:size(full_T,1),1:1:size(full_T,2),sum_res(:,:,1),20);
clabel(c,handle);
title('The reconstructed distribution of $G$','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
subplot(2,2,3)
err_data = abs(full_T(:,:,1)-sum_res(:,:,1));
h=bar3(err_data);
colorbar
for n=1:numel(h)
    cdata=get(h(n),'zdata');
    set(h(n),'cdata',cdata,'facecolor','interp','edgecolor','none')
end
title('The distribution of reconstructed error','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
zlabel('reconstructed error')
subplot(2,2,4)
histogram(log(err_data(:)),'edgecolor','none')
title('The histogram of reconstructed log-error','interpreter','latex');
xlabel('log(error)');
ylabel('count');
% % 残差的正态性检验
% chi2gof(log(err_data(:)),'CDF',{@normcdf,mean(log(err_data(:))),std(log(err_data(:)))})
% normplot(log(err_data(:)))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%张量不同层数&不同缺失值对重构误差的影响
figure
load ./20240723稀疏重构GDOP/张量不同层数/80缺失值/frob_list.mat
plot(10:1:20,frob_list,'.-');hold on
load ./20240723稀疏重构GDOP/张量不同层数/85缺失值/frob_list.mat
plot(10:1:20,frob_list,'.-');hold on
load ./20240723稀疏重构GDOP/张量不同层数/90缺失值/frob_list.mat
plot(10:1:20,frob_list,'.-');hold on
load ./20240723稀疏重构GDOP/张量不同层数/93缺失值/frob_list.mat
plot(10:1:20,frob_list,'.-');hold on
load ./20240723稀疏重构GDOP/张量不同层数/95缺失值/frob_list.mat
plot(10:1:20,frob_list,'.-');hold on
xlabel('Height(km)')
ylabel('$\Vert e \Vert_F$','interpreter','latex')
title('Tensor reconstruction error varies with height');
legend('20% data valid','15% data valid','10% data valid','7% data valid','5% data valid')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
case 2
%% 2-Tucker分解sigma2_x
% [size1,size2,size3]=size(Sigma2_x);
% L = mlrankest(sigma2_x);% 6 6 3
L = [20 20 3];
[Ax_res,Bx_res,Cx_res,Sx_res,Tx_hat] = Tucker_decomp(Sigma2_x,Sigma2_x+0.1*randn(size(Sigma2_x)),L,1);
% % svd(Ax_res)
% figure
% [c,handle]=contour(1:1:size(Sx_res,2),1:1:size(Sx_res,1),Sx_res(:,:,1),20);
% clabel(c,handle);
% title('S')
% temp = tmprod(Sx_res,{Ax_res},1);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*A')
% temp = tmprod(Sx_res,{Bx_res},2);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*B')
% temp = tmprod(Sx_res,{Cx_res},3);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*C')
% temp = tmprod(Sx_res,{Ax_res,Bx_res},1:2);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*A*B')
% temp = tmprod(Sx_res,{Ax_res,Cx_res},[1,3]);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*A*C')
% temp = tmprod(Sx_res,{Bx_res,Cx_res},[2,3]);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*B*C')
% temp = tmprod(Sx_res,{Ax_res,Bx_res,Cx_res},1:3);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*A*B*C')

case 3
%% 3-Tucker分解sigma2_y
L = [20 20 3];
[Ay_res,By_res,Cy_res,Sy_res,Ty_hat] = Tucker_decomp(Sigma2_y,Sigma2_y+0.1*randn(size(Sigma2_y)),L,1);
% figure
% [c,handle]=contour(1:1:size(Sy_res,2),1:1:size(Sy_res,1),Sy_res(:,:,1),20);
% clabel(c,handle);
% title('S')
% temp = tmprod(Sy_res,{Ay_res},1);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*A')
% temp = tmprod(Sy_res,{By_res},2);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*B')
% temp = tmprod(Sy_res,{Cy_res},3);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*C')
% temp = tmprod(Sy_res,{Ay_res,By_res},1:2);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*A*B')
% temp = tmprod(Sy_res,{Ay_res,Cy_res},[1,3]);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*A*C')
% temp = tmprod(Sy_res,{By_res,Cy_res},[2,3]);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*B*C')
% temp = tmprod(Sy_res,{Ay_res,By_res,Cy_res},1:3);
% figure
% [c,handle]=contour(1:1:size(temp,2),1:1:size(temp,1),temp(:,:,1),20);
% clabel(c,handle);
% title('S*A*B*C')

case 4
%% 4-Tucker分解sigma2_z
% L = mlrankest(sigma2_z);% 10 10 3
L = [10 10 3];
[Az_res,Bz_res,Cz_res,Sz_res,Tz_hat] = Tucker_decomp(Sigma2_z,Sigma2_z+0.1*randn(size(Sigma2_z)),L,1);

case 5
%% 5-分解sigma2_x+sigma2_y
target_T = Tx_hat + Ty_hat;
size_tens = size(target_T);
L1 = [10 3 3];
L2 = [3 10 3];
model= struct;
model.variables.A1=randn(size_tens(1),L1(1));
model.variables.B1=randn(size_tens(2),L1(2));
model.variables.C1=randn(size_tens(3),L1(3));
model.variables.S1=randn(L1(1),L1(2),L1(3));
model.variables.A2=randn(size_tens(1),L2(1));
model.variables.B2=randn(size_tens(2),L2(2));
model.variables.C2=randn(size_tens(3),L2(3));
model.variables.S2=randn(L2(1),L2(2),L2(3));
model.factors={ 'A1','B1','C1','S1', 'A2','B2','C2','S2' };
model.factorizations.mybtd.data=target_T;
model.factorizations.mybtd.btd={{1,2,3,4},{5,6,7,8}};
sdf_check(model,'print');
[sol,output] = sdf_nls(model);
[A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res] = deal(sol.factors{:});
Sigma2_x_hat = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
Sigma2_y_hat = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
sum_res = Sigma2_x_hat + Sigma2_y_hat;
frob(sum_res-target_T)
figure
[c,handle]=contour(1:1:size(G2,1),1:1:size(G2,2),Sigma2_x_hat(:,:,1),20);
clabel(c,handle);
title('The reconstructed distribution of $\sigma_x^2$','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
figure
[c,handle]=contour(1:1:size(G2,1),1:1:size(G2,2),Sigma2_y_hat(:,:,1),20);
clabel(c,handle);
title('The reconstructed distribution of $\sigma_x^2$','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
Sigma2_x_hat(Sigma2_x_hat(:,:,:)<1)=0;

case 6
%% 6-G重构时因子矩阵的不唯一性验证
%观测结果
Sigma2_x_N = Sigma2_x+0.1*randn(size(Sigma2_x));
Sigma2_y_N = Sigma2_y+0.1*randn(size(Sigma2_y));
Sigma2_z_N = Sigma2_z+0.1*randn(size(Sigma2_z));
T_N = Sigma2_x_N + Sigma2_y_N + Sigma2_z_N;
L = [10 10 3];
%构造一组可行解
[Ax_res,Bx_res,Cx_res,Sx_res,Tx_hat] = Tucker_decomp(Sigma2_x,Sigma2_x_N,L,1);
[Ay_res,By_res,Cy_res,Sy_res,Ty_hat] = Tucker_decomp(Sigma2_y,Sigma2_y_N,L,1);
[Az_res,Bz_res,Cz_res,Sz_res,Tz_hat] = Tucker_decomp(Sigma2_z,Sigma2_z_N,L,1);
Sigma2_x_hat1 = tmprod(Sx_res,{Ax_res,Bx_res,Cx_res},1:3);
Sigma2_y_hat1 = tmprod(Sy_res,{Ay_res,By_res,Cy_res},1:3);
Sigma2_z_hat1 = tmprod(Sz_res,{Az_res,Bz_res,Cz_res},1:3);
ReconstructedT1 = Tx_hat + Ty_hat + Tz_hat;
%不唯一解
size_tens = size(T_N);
model= struct;
model.variables.A1=randn(size_tens(1),L(1));
model.variables.B1=randn(size_tens(2),L(2));
model.variables.C1=randn(size_tens(3),L(3));
model.variables.S1=randn(L(1),L(2),L(3));
model.variables.A2=randn(size_tens(1),L(1));
model.variables.B2=randn(size_tens(2),L(2));
model.variables.C2=randn(size_tens(3),L(3));
model.variables.S2=randn(L(1),L(2),L(3));
model.variables.A3=randn(size_tens(1),L(1));
model.variables.B3=randn(size_tens(2),L(2));
model.variables.C3=randn(size_tens(3),L(3));
model.variables.S3=randn(L(1),L(2),L(3));
model.factors={ 'A1','B1','C1','S1', 'A2','B2','C2','S2', 'A3','B3','C3','S3' };
model.factorizations.mybtd.data=T_N;
model.factorizations.mybtd.btd={{1,2,3,4},{5,6,7,8},{9,10,11,12}};
sdf_check(model,'print');
[sol,output] = sdf_nls(model);
[A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res,A3_res,B3_res,C3_res,S3_res] = deal(sol.factors{:});
Sigma2_x_hat2 = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
Sigma2_y_hat2 = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
Sigma2_z_hat2 = tmprod(S3_res,{A3_res,B3_res,C3_res},1:3);
ReconstructedT2 = Sigma2_x_hat2 + Sigma2_y_hat2 + Sigma2_z_hat2;

figure();
[c,handle]=contour(1:1:size(ReconstructedT1,1),1:1:size(ReconstructedT1,2),ReconstructedT1(:,:,1),20);
clabel(c,handle);
% title('A set of global optimal solutions for $\underline{G}(:,:,1)$','interpreter','latex');
figure();
[c,handle]=contour(1:1:size(Sigma2_x_hat1,1),1:1:size(Sigma2_x_hat1,2),Sigma2_x_hat1(:,:,1),20);
% clabel(c,handle);
% title('A set of global optimal solutions for $\underline{G}_{x}(:,:,1)$','interpreter','latex');
figure();
[c,handle]=contour(1:1:size(Sigma2_y_hat1,1),1:1:size(Sigma2_y_hat1,2),Sigma2_y_hat1(:,:,1),20);
% clabel(c,handle);
% title('A set of global optimal solutions for $\underline{G}_{y}(:,:,1)$','interpreter','latex');
figure();
[c,handle]=contour(1:1:size(Sigma2_z_hat1,1),1:1:size(Sigma2_z_hat1,2),Sigma2_z_hat1(:,:,1),20);
% clabel(c,handle);
% title('A set of global optimal solutions for $\underline{G}_{z}(:,:,1)$','interpreter','latex');
figure();
[c,handle]=contour(1:1:size(ReconstructedT2,1),1:1:size(ReconstructedT2,2),ReconstructedT2(:,:,1),20);
clabel(c,handle);
% title('A set of local optimal solutions for $\underline{G}(:,:,1)$','interpreter','latex');
figure();
[c,handle]=contour(1:1:size(Sigma2_x_hat2,1),1:1:size(Sigma2_x_hat2,2),Sigma2_x_hat2(:,:,1),20);
% clabel(c,handle);
% title('A set of local optimal solutions for $\underline{G}_{x}(:,:,1)$','interpreter','latex');
figure();
[c,handle]=contour(1:1:size(Sigma2_y_hat2,1),1:1:size(Sigma2_y_hat2,2),Sigma2_y_hat2(:,:,1),20);
% clabel(c,handle);
% title('A set of local optimal solutions for $\underline{G}_{y}(:,:,1)$','interpreter','latex');
figure();
[c,handle]=contour(1:1:size(Sigma2_z_hat2,1),1:1:size(Sigma2_z_hat2,2),Sigma2_z_hat2(:,:,1),20);
% clabel(c,handle);
% title('A set of local optimal solutions for $\underline{G}_{z}(:,:,1)$','interpreter','latex');

end




















