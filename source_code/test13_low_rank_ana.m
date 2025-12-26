clear
close all

%% DOA
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.20;
x3 = 0;y3 = -L;z3 = 0.30;
% z1=zt;z2=zt;z3=zt;%不考虑高程差的仿真结果
% p=10;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];

fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;
% z_span = linspace(10,20,11);
z_span = 10;
A = rand(2*size(sensor,1)); term = tril(A, -1) + triu(A', 0);
for k =1:1:length(z_span)
    for i = 1:1:length(x_span)
        for j = 1:1:length(y_span)
            m=x_span(i)+0.01;
            n=y_span(j)+0.01;
            p=z_span(k)+0.001;%height=10m
            emitter = [m,n,p];
            value.sigma_theta1 = ones(1,4).*0.1/180*pi;
            value.sigma_theta2 = ones(1,4).*0.08/180*pi;
            value.sigma_s = 1e-3;
            [Gx,Gy,Gz,G] = cal_gdop_doa_for_test13(emitter,sensor,value,term);
            recon_tensor(j,i,k) = G;
        end
    end
end
rank_ana = [];
s = svd(recon_tensor);
sum_s = sum(s);
for i = 1:1:30
    rank_ana = [rank_ana,sum(s(1:i))/sum_s];
end
figure
% subplot(2,2,2)
plot(1:1:30,rank_ana,'ro-');
text(5,rank_ana(5),['(5,',num2str(rank_ana(5),'%.3f'),')'],'Color','r')
xlabel('$i$','interpreter','latex')
ylabel('$\zeta_{i}$','interpreter','latex')
title('Ratio of eigenvalues.'); hold on;
%% TDOA
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
x3 = 0;y3 = -L;z3 = 0.3;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];

fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;
% z_span = linspace(10,20,11);
z_span = 10;
% vars_real = [30e-3,30e-3,30e-3,5e-3,0.3,0.3,0.3];
vars_real = randn(1,7);
for k =1:1:length(z_span)
    for i = 1:1:length(x_span)
        for j = 1:1:length(y_span)
            m=x_span(i)+0.01;
            n=y_span(j)+0.01;
            p=z_span(k)+0.001;%height=10m
            emitter = [m,n,p];
            [Gx_eq,Gy_eq,Gz_eq,G_eq] = cal_diff(emitter,sensor,vars_real);%时间单位us，距离单位km
            G2(j,i,k) = G_eq;
            Sigma2_x(j,i,k) = Gx_eq;
            Sigma2_y(j,i,k) = Gy_eq;
            Sigma2_z(j,i,k) = Gz_eq;
        end
    end
end
rank_ana = [];
s = svd(G2);
sum_s = sum(s);
for i = 1:1:30
    rank_ana = [rank_ana,sum(s(1:i))/sum_s];
end
% figure
% subplot(2,2,4)
plot(1:1:30,rank_ana,'b>-');
% text(x_span(0+center)+text_bias,y_span(bias+center)-text_bias,num2str(tensor_slice(0+center,bias+center),'%.3f'))
text(5,rank_ana(5),['(5,',num2str(rank_ana(5),'%.3f'),')'],'Color','b')
% xlabel('$i$','interpreter','latex')
% ylabel('$\zeta_{i}$','interpreter','latex')
% title('Ratio of eigenvalues in TDOA.')
ylim([0.92,1.01])
legend('DOA','TDOA')
figure
% subplot(2,2,1)
[c,handle]=contour(x_span,y_span,recon_tensor);
% clabel(c,handle);
xlabel('x(km)');
ylabel('y(km)');
title('DOA')
figure
% subplot(2,2,3)
[c,handle]=contour(x_span,y_span,G2);
% clabel(c,handle);
xlabel('x(km)');
ylabel('y(km)');
title('TDOA')