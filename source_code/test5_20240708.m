close all
clear all
A = zeros(800,800);%整体范围



%% 传感器位置
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
x3 = 0;y3 = -L;z3 = 0.3;
% z1=zt;z2=zt;z3=zt;%不考虑高程差的仿真结果
% p=10;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];
%% 计算GDOP的三维空间分布&验证公式正确性
cal_z = 1;%是否计算三维的gdop
y=-400:1:400;x=-400:1:400;z=10:1:20;
% [gdop,G,sigma2_x,sigma2_y,sigma2_z,gdop2_TDOA,gdop2_S] = deal(zeros(length(x),length(y),length(z)));
% for i=1:length(x)
%     for j=1:length(y)
%         for k=1:1:length(z)
%             m=x(i)+0.01;
%             n=y(j)+0.01;
%             p=z(k)+0.001;%height=10m
%             emitter = [m,n,p];
% %             [sol1,sol2,sol3,sol4] = eq_verify(emitter,sensor);%用于验证论文中公式
% %             if sum([sol1,sol2,sol3,sol4])>1
% %                 disp('wrong')
% %             end
%             if(cal_z)
%                 [gdop_ji,gdop2_TDOA_ji,gdop2_S_ji,sigma2_x_ji,sigma2_y_ji,sigma2_z_ji,g_ji] = cal_gdop2(emitter,sensor);
%             else
%                 [gdop_ji,gdop2_TDOA_ji,gdop2_S_ji] = cal_gdop(emitter,sensor);
%             end
%             
%             %             gdop_ji = cal_gdop(emitter,sensor);
%             gdop(i,j,k) = gdop_ji;
%             G(i,j,k) = g_ji;
%             sigma2_x(i,j,k) = sigma2_x_ji;
%             sigma2_y(i,j,k) = sigma2_y_ji;
%             sigma2_z(i,j,k) = sigma2_z_ji;
%             gdop2_TDOA(i,j,k) = gdop2_TDOA_ji;
%             gdop2_S(i,j,k) = gdop2_S_ji;
%         end
%     end
% end
%% 加载上一步的三维空间的GDOP、sigmax2等参数（节省时间）
files = dir('*.mat');
for i=1:1:length(files)
    str = files(i).name;
    load(str)
end
%% 对GDOP等参数可视化
% disp(frob(G-sigma2_x-sigma2_y-sigma2_z))
% disp(frob(G-gdop2_TDOA-gdop2_S))
% disp(frob(G-gdop.^2))
% figure();
% %contour(Z,n),n指定了等高线的条数，用于绘制矩阵的等高线
% [c,handle]=contour(x,y,gdop(:,:,1),20);
% clabel(c,handle);
% xlabel('x方向(单位:km)');
% ylabel('y方向(单位:km)');
% title('GDOP');
% figure();
% %contour(Z,n),n指定了等高线的条数，用于绘制矩阵的等高线
% [c,handle]=contour(x,y,sigma2_x(:,:,1),20);
% clabel(c,handle);
% xlabel('x方向(单位:m)');
% ylabel('y方向(单位:m)');
% title('GDOPX');
%% 设计无人机飞行轨迹
% % 定义参数方程
% t = linspace(0, 1, 100); % 参数t从0到1
% x = sin(2*pi*t); % 示例方程 x = sin(2*pi*t)
% y = cos(2*pi*t); % 示例方程 y = cos(2*pi*t)
% 
% % 绘制曲线
% plot(x, y);
% xlabel('X');
% ylabel('Y');
% title('S曲线轨迹');
% grid on;
%% 最小二乘解
fs_x = 200;
fs_y = 200;
fs_z = 2;
syms sigmat1 sigmat2 sigmat3
syms sigmas
syms eta12 eta13 eta23
syms M
%% 最小二乘解析解情况1-eta不为0但未知，sigmat均相同
% x方向和y方向采样率1/200，z方向全部保留
var_value = [sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23];%变量列表，eta初始化为0，eta必须取常数而不是变量
real_value = [30e-3, 30e-3, 30e-3,5e-3, 0.3, 0.3, 0.3];%变量对应的真实值标签，可以看到eta并不是0
% 0- 根据真实误差项的值仿真TDOA，每个定位点处假设能有N条时差测量数据
for i=1:length(x)/fs_x
    disp(['processing calTDOA_',num2str(i)])
    for j=1:length(y)/fs_y
        for k=1:1:length(z)/fs_z
            pos{i,j,k}=[x(fs_x*i)+0.01,y(fs_y*j)+0.01,z(k)+0.001];%抽样选取的位置
            N=3;
            tdoa{i,j,k} = cal_tdoa(pos{i,j,k},sensor,real_value,N);%根据real_value里的值构造带相关噪声的sigmat1、2、3
        end
    end
end
% 1- 通过统计TDOA，给出初始的TDOA方差估计值以及相关系数eta
[sigma_t1_initial,sigma_t2_initial,sigma_t3_initial,eta12_initial,eta13_initial,eta23_initial,num] = deal(0);
for i=1:length(x)/fs_x
    disp(['processing calTDOA_',num2str(i)])
    for j=1:length(y)/fs_y
        for k=1:1:length(z)/fs_z
            sigma_t1_initial = sigma_t1_initial + var(tdoa{i,j,k}(:,1));
            sigma_t2_initial = sigma_t2_initial + var(tdoa{i,j,k}(:,2));
            sigma_t3_initial = sigma_t3_initial + var(tdoa{i,j,k}(:,3));
            eta12_initial = eta12_initial + corr(tdoa{i,j,k}(:,1),tdoa{i,j,k}(:,2));
            eta13_initial = eta13_initial + corr(tdoa{i,j,k}(:,1),tdoa{i,j,k}(:,3));
            eta23_initial = eta23_initial + corr(tdoa{i,j,k}(:,2),tdoa{i,j,k}(:,3));
            num = num + 1;
        end
    end
end
eta12_initial = eta12_initial / num; sigma_t1_initial = sqrt(sigma_t1_initial/num);
eta13_initial = eta13_initial / num; sigma_t2_initial = sqrt(sigma_t2_initial/num);
eta23_initial = eta23_initial / num; sigma_t3_initial = sqrt(sigma_t3_initial/num);
% 2- 根据估计的sigmat和相关系数eta，将sigmas作为未知变量，求解最小二乘解析结果
% var_value_cal1 = [sigmat1,sigmat1,sigmat1,sigmas,eta12_initial,eta13_initial,eta23_initial];%实际代入公式计算的变量列表
var_value_cal1 = [sigma_t1_initial,sigma_t2_initial,sigma_t3_initial,sigmas,eta12_initial,eta13_initial,eta23_initial];%实际代入公式计算的变量列表
coeffs = [];
b = [];
for i=1:length(x)/fs_x
    disp(['processing sigmas_',num2str(i)])
    for j=1:length(y)/fs_y
        for k=1:1:length(z)/fs_z            
            [~,~,~,G_eq] = cal_diff(pos{i,j,k},sensor,var_value); %计算G关于各变量的偏微分以及G_cal关于变量的表达式，是真实表达式
            G_real = double(subs(G_eq, var_value,real_value)); %在表达式中代入真实误差，得到真实的G
            G_real = awgn(G_real,20);%加噪
            G_Cal = subs(G_eq, var_value,var_value_cal1); %在表达式中代入变量和预设值，得到计算的G
            var_name = [sigmas^2];%G的表达式中有哪些多项式的项
            coeff(1) = double(subs(G_Cal, var_name,1));%求1次项式系数
            coeff(2) = double(subs(G_Cal, var_name,0));%求常数项式系数
            coeffs =  [coeffs;coeff];
%             var_name = [sigmat1^2,sigmas^2];%G的表达式中有哪些多项式的项
%             coeff = double(expr_coeff(G_Cal, var_name));
%             coeffs =  [coeffs;coeff];
            b = [b;G_real];
        end
    end
end
sigmas_initial = sqrt(pinv(coeffs(:,1))*(b-coeffs(:,2)));
% res1 = sqrt(pinv(coeffs)*b);%在eta=0的情况下得到的sigmat和sigmas的估计值resss=[sigmat1,sigmas];
% 3- 根据估计的sigmas和相关系数eta，将sigmat作为未知变量，求解最小二乘解析结果
% var_value_cal1 = [sigmat1,sigmat1,sigmat1,sigmas,eta12_initial,eta13_initial,eta23_initial];%实际代入公式计算的变量列表
var_value_cal1 = [sigmat1,sigmat1,sigmat1,sigmas_initial,eta12_initial,eta13_initial,eta23_initial];%实际代入公式计算的变量列表
coeffs = [];
for i=1:length(x)/fs_x
    disp(['processing sigmat_',num2str(i)])
    for j=1:length(y)/fs_y
        for k=1:1:length(z)/fs_z            
            [~,~,~,G_eq] = cal_diff(pos{i,j,k},sensor,var_value); %计算G关于各变量的偏微分以及G_cal关于变量的表达式，是真实表达式
            G_Cal = subs(G_eq, var_value,var_value_cal1); %在表达式中代入变量和预设值，得到计算的G
            var_name = [sigmat1^2];%G的表达式中有哪些多项式的项
            coeff(1) = double(subs(G_Cal, var_name,1));%求1次项式系数
            coeff(2) = double(subs(G_Cal, var_name,0));%求常数项式系数
            coeffs =  [coeffs;coeff];
        end
    end
end
sigmat_initial = sqrt(pinv(coeffs(:,1))*(b-coeffs(:,2)));
% res1 = sqrt(pinv(coeffs)*b);%在eta=0的情况下得到的sigmat和sigmas的估计值resss=[sigmat1,sigmas];
% 4- 根据估计的sigmas和sigmat，将eta作为未知变量，求解最小二乘解析结果
% var_value_cal1 = [sigmat1,sigmat1,sigmat1,sigmas,eta12_initial,eta13_initial,eta23_initial];%实际代入公式计算的变量列表
var_value_cal1 = [sigmat_initial,sigmat_initial,sigmat_initial,sigmas_initial,eta12,eta12,eta12];%实际代入公式计算的变量列表
coeffs = [];
for i=1:length(x)/fs_x
    disp(['processing eta_',num2str(i)])
    for j=1:length(y)/fs_y
        for k=1:1:length(z)/fs_z            
            [~,~,~,G_eq] = cal_diff(pos{i,j,k},sensor,var_value); %计算G关于各变量的偏微分以及G_cal关于变量的表达式，是真实表达式
            G_Cal = subs(G_eq, var_value,var_value_cal1); %在表达式中代入变量和预设值，得到计算的G
            var_name = [eta12];%G的表达式中有哪些多项式的项
            coeff(1) = double(subs(G_Cal, var_name,1));%求1次项式系数
            coeff(2) = double(subs(G_Cal, var_name,0));%求常数项式系数
            coeffs =  [coeffs;coeff];
        end
    end
end
eta_initial = pinv(coeffs(:,1))*(b-coeffs(:,2));
% res1 = sqrt(pinv(coeffs)*b);%在eta=0的情况下得到的sigmat和sigmas的估计值resss=[sigmat1,sigmas];





% 3- 根据sigmas和sigmat的估计值更新eta
[cov12_initial,cov13_initial,cov23_initial,num] = deal(0);
for i=1:length(x)/fs_x
    disp(['processing calEta_',num2str(i)])
    for j=1:length(y)/fs_y
        for k=1:1:length(z)/fs_z
            cov12_initial = cov12_initial + cov(tdoa{i,j,k}(:,1),tdoa{i,j,k}(:,2));
            cov13_initial = cov13_initial + cov(tdoa{i,j,k}(:,1),tdoa{i,j,k}(:,3));
            cov23_initial = cov23_initial + cov(tdoa{i,j,k}(:,2),tdoa{i,j,k}(:,3));
            num = num + 1;
        end
    end
end
eta12_processed = cov12_initial(1,2) /num/res1(1)/res1(1);
eta13_processed = cov13_initial(1,2) /num/res1(1)/res1(1);
eta23_processed = cov23_initial(1,2) /num/res1(1)/res1(1);

var_value_cal2 = [res1(1),res1(1),res1(1),res1(2),eta12,eta12,eta12];%变量列表，代入计算出的sigmat和sigmas，eta取变量求eta
coeffs2 = [];
for i=1:length(x)/fs_x
    disp(['processing eta_',num2str(i)])
    for j=1:length(y)/fs_y
        for k=1:1:length(z)/fs_z
            pos{i,j,k}=[x(fs_x*i)+0.01,y(fs_y*j)+0.01,z(k)+0.001];%抽样选取的位置
            [~,~,~,G_Cal] = cal_diff(pos{i,j,k},sensor,var_value_cal2); %计算G关于各变量的偏微分以及G_cal关于变量的表达式
            var_name = [eta12];%G的表达式中有哪些多项式的项
%             coeff = double(expr_coeff(G_Cal, var_name));
            coeff(1) = double(subs(G_Cal, var_name,1));%求1次项式系数
            coeff(2) = double(subs(G_Cal, var_name,0));%求常数项式系数
            coeffs2 =  [coeffs2;coeff];
        end
    end
end
res2 = sqrt(pinv(coeffs2)*b);%在估计出sigmat和sigmas的情况下得到的eta的估计值resss=[eta,1];

%% 最小二乘解析解情况2-eta均为0，sigmat各不同
% x方向和y方向采样率1/20，z方向全部保留
fs_x = 100;
fs_y = 100;
syms sigmat1 sigmat2 sigmat3
syms sigmas
syms eta12 eta13 eta23
syms M
% var_value = [sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23];
% var_value = [sigmat1,sigmat1,sigmat1,sigmas,eta12,eta12,eta12];
var_value = [sigmat1,sigmat2,sigmat3,sigmas,0,0,0];%变量列表，eta必须取常数而不是变量，因为eta所在的项的系数与其他几项有相关性，那么系数矩阵就不会列满秩
real_value = [30e-3, 25e-3, 15e-3,5e-3, 0, 0, 0];%变量对应的真实值标签
var_value = [sigmat1,sigmat1,sigmat1,sigmas,0,0,0];%变量列表，eta必须取常数而不是变量，因为eta所在的项的系数与其他几项有相关性，那么系数矩阵就不会列满秩
real_value = [30e-3, 30e-3, 30e-3,5e-3, 0.2, 0.2, 0.2];%变量对应的真实值标签
coeffs = [];
b = [];

for i=1:length(x)/fs_x
    for j=1:length(y)/fs_y
        for k=1:1:length(z)
            %                 G_hat(i,j,k) = G(fs_x*i,fs_y*j,k);%观测值
            pos{i,j,k}=[x(fs_x*i)+0.01,y(fs_y*j)+0.01,z(k)+0.001];%抽样选取的位置
            [~,~,~,G_Cal] = cal_diff(pos{i,j,k},sensor,var_value); %计算G关于各变量的偏微分以及G_cal关于变量的表达式
            G_real = double(subs(G_Cal, [sigmat1, sigmat2, sigmat3,sigmas, eta12, eta13, eta23],real_value)); %代入真实误差，得到真实的G
            G_real = awgn(G_real,20);%加噪
            
%             var_name = [sigmas^2,sigmat1^2,sigmat2^2,sigmat3^2,eta12*sigmat1*sigmat2,eta13*sigmat1*sigmat3,eta23*sigmat2*sigmat3];
%             var_name = [sigmas^2,sigmat1^2,eta12*sigmat1^2];
            var_name = [sigmat1^2,sigmat2^2,sigmat3^2,sigmas^2];%G的表达式中有哪些多项式的项
            var_name = [sigmat1^2,sigmas^2];%G的表达式中有哪些多项式的项
            coeff = double(expr_coeff(G_Cal, var_name));
%             coeff(1) = double(subs(G_Cal, var_name,[1 0 0 0]));%求多项式系数
%             coeff(2) = double(subs(G_Cal, var_name,[0 1 0 0]));
%             coeff(3) = double(subs(G_Cal, var_name,[0 0 1 0]));
%             coeff(4) = double(subs(G_Cal, var_name,[0 0 0 1]));
%             
%             %                 coeff(1) = double(subs(G_Cal, [sigmas^2, sigmat1,  eta12],[1 0 0]));
%             %                 coeff(2) = double(subs(G_Cal, [sigmas^2, sigmat1,  eta12],[0 1 0]));
%             %                 coeff(3) = double(subs(G_Cal, [sigmas^2, sigmat1,  eta12],[0 1 1]))-coeff(2);
%             % %                 coeff = double(expr_coeff(G_Cal, var_name)); %求G_cal中各项系数
%             
%             real_value_cell = num2cell(real_value);
%             [asigmat1, asigmat2, asigmat3,asigmas, aeta12, aeta13, aeta23]=deal(real_value_cell{:});
%             %                 a=coeff * [asigmas^2,asigmat1^2,asigmat2^2,asigmat3^2,aeta12*asigmat1*asigmat2,aeta13*asigmat1*asigmat3,aeta23*asigmat2*asigmat3]';%当代入真实误差时，系数乘以误差等于G_hat
%             %                 a=coeff * [asigmas^2,asigmat1^2,aeta12*asigmat1^2]';%当代入真实误差时，系数乘以误差等于G_hat
%             a=coeff * [asigmat1^2,asigmat2^2,asigmat3^2,asigmas^2]';%当代入真实误差时，系数乘以误差等于G_real
%             disp(G_real-a)
            coeffs =  [coeffs;coeff];
            b = [b;G_real];
            %                 e = G_hat(i,j,k)-G_Cal;
            %                 res = res_x+res_y+res_z;
            %                 delta_value = delta_value + 1/N* (-e)*res;
        end
    end
end

resss = sqrt(pinv(coeffs)*b);

%% 梯度下降解
% x方向和y方向采样率1/20，z方向全部保留
fs_x = 100;
fs_y = 100;
var_value = randn(1,7);%初始化
var_value = [30e-3,30e-3,30e-3,5e-3,0.3,0.3,0.3];
delta_value = zeros(1,7);
syms sigmat1 sigmat2 sigmat3
syms sigmas
syms eta12 eta13 eta23
syms M
var_value = [sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23];
var_value = [sigmat1,sigmat1,sigmat1,sigmas,eta12,eta12,eta12];
% var_value = [sigmat1,sigmat2,sigmat3,sigmas,0,0,0];
real_value = [30e-3, 25e-3, 15e-3,5e-3, 0, 0, 0];%eta必须取常数而不是变量，因为eta所在的项的系数与其他几项有相关性，那么系数矩阵就不会列满秩
coeffs = [];
b = [];
% for iter = 1:1:100
%     var_value = var_value - 0.00001*delta_value;
%     N = length(x)/fs_x*length(y)/fs_y*length(z);
    for i=1:length(x)/fs_x
        for j=1:length(y)/fs_y
            for k=1:1:length(z)
%                 G_hat(i,j,k) = G(fs_x*i,fs_y*j,k);%观测值
                pos{i,j,k}=[x(fs_x*i)+0.01,y(fs_y*j)+0.01,z(k)+0.001];%抽样选取的位置
                [res_x,res_y,res_z,G_Cal] = cal_diff(pos{i,j,k},sensor,var_value); %计算G关于各变量的偏微分以及G_cal关于变量的表达式
                G_real = double(subs(G_Cal, [sigmat1, sigmat2, sigmat3,sigmas, eta12, eta13, eta23],real_value)); %代入真实误差，得到真实的G
                G_real = awgn(G_real,20);%加噪
                
                var_name = [sigmas^2,sigmat1^2,sigmat2^2,sigmat3^2,eta12*sigmat1*sigmat2,eta13*sigmat1*sigmat3,eta23*sigmat2*sigmat3];
                var_name = [sigmas^2,sigmat1^2,eta12*sigmat1^2];
                var_name = [sigmat1^2,sigmat2^2,sigmat3^2,sigmas^2];%G的表达式中有哪些多项式的项
                coeff(1) = double(subs(G_Cal, var_name,[1 0 0 0]));%求多项式系数
                coeff(2) = double(subs(G_Cal, var_name,[0 1 0 0]));
                coeff(3) = double(subs(G_Cal, var_name,[0 0 1 0]));
                coeff(4) = double(subs(G_Cal, var_name,[0 0 0 1]));
                
%                 coeff(1) = double(subs(G_Cal, [sigmas^2, sigmat1,  eta12],[1 0 0]));
%                 coeff(2) = double(subs(G_Cal, [sigmas^2, sigmat1,  eta12],[0 1 0]));
%                 coeff(3) = double(subs(G_Cal, [sigmas^2, sigmat1,  eta12],[0 1 1]))-coeff(2);
% %                 coeff = double(expr_coeff(G_Cal, var_name)); %求G_cal中各项系数

                real_value_cell = num2cell(real_value);
                [asigmat1, asigmat2, asigmat3,asigmas, aeta12, aeta13, aeta23]=deal(real_value_cell{:});
%                 a=coeff * [asigmas^2,asigmat1^2,asigmat2^2,asigmat3^2,aeta12*asigmat1*asigmat2,aeta13*asigmat1*asigmat3,aeta23*asigmat2*asigmat3]';%当代入真实误差时，系数乘以误差等于G_hat
%                 a=coeff * [asigmas^2,asigmat1^2,aeta12*asigmat1^2]';%当代入真实误差时，系数乘以误差等于G_hat
                a=coeff * [asigmat1^2,asigmat2^2,asigmat3^2,asigmas^2]';%当代入真实误差时，系数乘以误差等于G_real
                disp(G_real-a)
                coeffs =  [coeffs;coeff];
                b = [b;G_real];
%                 e = G_hat(i,j,k)-G_Cal;
%                 res = res_x+res_y+res_z;
%                 delta_value = delta_value + 1/N* (-e)*res;
            end
        end
    end
% end
resss = sqrt(pinv(coeffs)*b);
