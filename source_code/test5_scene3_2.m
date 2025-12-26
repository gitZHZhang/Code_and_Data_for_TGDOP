close all
clear all
% scene3：任一sigmati有相对准确的估计值时，可以优化出其余eta、sigmat和sigmas等变量的参数

use_former_res = 0;

%% 传感器位置
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
x3 = 0;y3 = -L;z3 = 0.3;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];
y=-400:1:400;x=-400:1:400;z=10:1:20;%感兴趣范围
%% 最小二乘解
%采样间隔 % x方向和y方向采样率1/200，z方向1/2
fs_x = 200;
fs_y = 200;
fs_z = 2;
%未知变量
syms sigmat1 sigmat2 sigmat3
syms sigmas
syms eta12 eta13 eta23
syms M
%% 最小二乘解析解情况：sigmas有一个估计值，其余变量的值未知
%% 0-给出各变量的真值
var_value = [sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23];%变量列表，eta初始化为0，eta必须取常数而不是变量
real_value = [30e-3, 25e-3, 15e-3,5e-3, 0.2, 0.15, 0.1];%变量对应的真实值标签，可以看到eta并不是0
real_valuecell = num2cell(real_value);
[sigmat1_real,sigmat2_real,sigmat3_real,sigmas_real,eta12_real,eta13_real,eta23_real] = deal(real_valuecell{:});
iter_N = 50;

if ~(use_former_res) %不选用之前的实验结果，重新开始统计
%% 1-给出各变量的估计值（初值）
% 0- 根据真实误差项的值仿真TDOA，每个定位点处假设能有N条时差测量数据
for i=1:length(x)/fs_x
    disp(['simulating TDOA ',num2str(i)])
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
    disp(['calculating initial value of parameters ',num2str(i)])
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

%% 2-将sigmati估计值作为已知量，将其余变量作为未知量求解
% 2- 根据估计的相关系数eta，将sigmat和sigmas作为位置变量，求解最小二乘解析结果
rel_err_list = 8:1:10; %单位百分比,表示标准差是真值的百分之多少
for iter_G = 1:1:length(rel_err_list)
    rel_err = rel_err_list(iter_G);
    [sigmat1_list,sigmat2_list,sigmat3_list,sigmas_list,eta12_list,eta13_list,eta23_list] = deal([]);%用于存放不同iter的估计值
    for iter = 1:1:iter_N
        sigma = rel_err/100*sigmat1_real;
        sigmat1_initial = mean(sigmat1_real+sigma*randn(10,1));
        var_value_cal1 = [sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23];%实际代入公式计算的变量列表
        coeffs = [];
        constant = [];
        b = [];
        for i=1:length(x)/fs_x
            disp(['processing sigmat&eta_',num2str(i)])
            for j=1:length(y)/fs_y
                for k=1:1:length(z)/fs_z
                    [~,~,~,G_eq] = cal_diff(pos{i,j,k},sensor,var_value); %计算G关于各变量的偏微分以及G_cal关于变量的表达式，是真实表达式
                    G_real = double(subs(G_eq, var_value,real_value)); %在表达式中代入真实误差，得到真实的G
                    G_Cal = subs(G_eq, var_value,var_value_cal1); %在表达式中代入变量和预设值，得到计算的G
                    %                 G_Cal = subs(G_Cal, sigmat1,sigmat1_initial); %在表达式中代入变量和预设值，得到计算的G
                    G_Cal = subs(G_Cal, sigmat1,sigmat1_initial); %在表达式中代入变量和预设值，得到计算的G
                    
                    var_name = [sigmat2,sigmat3,sigmas,eta12,eta13,eta23];%G的表达式中有哪些多项式的项
                    %             G_Cal_real = double(subs(G_Cal, var_name,[sigmat1_real,sigmat2_real,sigmat3_real,eta12_real,eta13_real,eta23_real]))
                    constant = [constant;double(subs(G_Cal, var_name,zeros(1,length(var_name))))];%求常数项式系数
                    %             coeff(2) = double(subs(G_Cal, var_name,[1,0]))-coeff(1);%求sigmat2系数
                    %             coeff(3) = double(subs(G_Cal, var_name,[1,1]))-coeff(1)-coeff(2);%求eta*sigmat2项系数
                    %             coeffs =  [coeffs;coeff];
                    var_name = [sigmat2^2,sigmat3^2,sigmas^2,eta12*sigmat2,eta13*sigmat3,eta23*sigmat2*sigmat3];%G的表达式中有哪些多项式的项
                    coeff = double(expr_coeff(G_Cal, var_name));
                    %             coeff*[sigmat1_real^2,sigmat2_real^2,sigmat3_real^2,eta12_real*sigmat1_real*sigmat2_real,eta13_real*sigmat1_real*sigmat3_real,...
                    %                 eta23_real*sigmat2_real*sigmat3_real]'+double(subs(G_Cal, var_name,zeros(1,length(var_name))))
                    coeffs =  [coeffs;coeff];
                    b = [b;G_real];
                end
            end
        end
        
        res1 = pinv(coeffs)*(b-constant);
        sigmt2_res = sqrt(res1(1));sigmat2_list = [sigmat2_list,sigmt2_res];
        sigmt3_res = sqrt(res1(2));sigmat3_list = [sigmat3_list,sigmt3_res];
        sigmas_res = sqrt(res1(3));sigmas_list = [sigmas_list,sigmas_res];
        eta12_res = res1(4)/sigmt2_res; eta12_list = [eta12_list,eta12_res];
        eta13_res = res1(5)/sigmt3_res; eta13_list = [eta13_list,eta13_res];
        eta23_res = res1(6)/sigmt2_res/sigmt3_res; eta23_list = [eta23_list,eta23_res];
    end
    savepath = ['scene3/2初值影响分析/rel_err=',num2str(rel_err)];
%     savepath = ['scene3/噪声影响分析/G有噪/方案1/n_samples=',num2str(n_samples)];
    save([savepath,'/eta12_list_0.2.mat'],'eta12_list')
    save([savepath,'/eta13_list_0.15.mat'],'eta13_list')
    save([savepath,'/eta23_list_0.1.mat'],'eta23_list')
    save([savepath,'/sigmas_list_0.005.mat'],'sigmas_list')
    save([savepath,'/sigmat2_list_0.025.mat'],'sigmat2_list')
    save([savepath,'/sigmat3_list_0.015.mat'],'sigmat3_list')   
    figure
    plot(1:1:iter_N,sigmat2_real*ones(1,iter_N),'b-');hold on
    plot(1:1:iter_N,sigmat2_list,'b.-');hold on
    plot(1:1:iter_N,sigmat3_real*ones(1,iter_N),'g-');hold on
    plot(1:1:iter_N,sigmat3_list,'g.-');hold on
    plot(1:1:iter_N,sigmas_real*ones(1,iter_N),'m-');hold on
    plot(1:1:iter_N,sigmas_list,'m.-');hold on
    % legend('sigmat2-real','sigmat2-esti','sigmat3-real','sigmat3-esti','sigmas-real','sigmas-esti')
    leg = legend('$\sigma_{\Delta t_2}$','$\tilde{\sigma}_{\Delta t_2}$','$\sigma_{\Delta t_3}$','$\tilde{\sigma}_{\Delta t_3}$',...
        '$\sigma_s$','$\tilde{\sigma}_s$');
    set(leg,'Interpreter','latex')
    saveas(gcf,[savepath,'/1.png'])
    figure
    plot(1:1:iter_N,eta12_real*ones(1,iter_N),'r-');hold on
    plot(1:1:iter_N,eta12_list,'r.-');hold on
    plot(1:1:iter_N,eta13_real*ones(1,iter_N),'b-');hold on
    plot(1:1:iter_N,eta13_list,'b.-');hold on
    plot(1:1:iter_N,eta23_real*ones(1,iter_N),'g-');hold on
    plot(1:1:iter_N,eta23_list,'g.-');hold on
    % legend('eta12-real','eta12-esti','eta13-real','eta13-esti','eta23-real','eta23-esti')
    leg = legend('$\eta_{12}$','$\tilde{\eta}_{12}$','$\eta_{13}$','$\tilde{\eta}_{13}$','$\eta_{23}$','$\tilde{\eta}_{23}$');
    set(leg,'Interpreter','latex')
    saveas(gcf,[savepath,'/2.png'])
end

end

%% 加载上面的结果，并画图分析
if use_former_res%使用之前的实验结果做分析
files = dir('scene3/*.mat');
for i=1:1:length(files)
    str = ['scene3/',files(i).name];
    load(str)
end

figure

plot(1:1:iter_N,sigmat2_real*ones(1,iter_N),'b-');hold on
plot(1:1:iter_N,sigmat2_list,'b.-');hold on
plot(1:1:iter_N,sigmat3_real*ones(1,iter_N),'g-');hold on
plot(1:1:iter_N,sigmat3_list,'g.-');hold on
plot(1:1:iter_N,sigmas_real*ones(1,iter_N),'m-');hold on
plot(1:1:iter_N,sigmas_list,'m.-');hold on
% legend('sigmat2-real','sigmat2-esti','sigmat3-real','sigmat3-esti','sigmas-real','sigmas-esti')
leg = legend('$\sigma_{\Delta t_2}$','$\tilde{\sigma}_{\Delta t_2}$','$\sigma_{\Delta t_3}$','$\tilde{\sigma}_{\Delta t_3}$',...
    '$\sigma_s$','$\tilde{\sigma}_s$');
set(leg,'Interpreter','latex')
figure
plot(1:1:iter_N,eta12_real*ones(1,iter_N),'r-');hold on
plot(1:1:iter_N,eta12_list,'r.-');hold on
plot(1:1:iter_N,eta13_real*ones(1,iter_N),'b-');hold on
plot(1:1:iter_N,eta13_list,'b.-');hold on
plot(1:1:iter_N,eta23_real*ones(1,iter_N),'g-');hold on
plot(1:1:iter_N,eta23_list,'g.-');hold on
% legend('eta12-real','eta12-esti','eta13-real','eta13-esti','eta23-real','eta23-esti')
leg = legend('$\eta_{12}$','$\tilde{\eta}_{12}$','$\eta_{13}$','$\tilde{\eta}_{13}$','$\eta_{23}$','$\tilde{\eta}_{23}$');
set(leg,'Interpreter','latex')

end
