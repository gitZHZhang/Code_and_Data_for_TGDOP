close all
clear
% scene1：sigmas有相对准确的估计值时，可以优化出eta和sigmat等6个变量的参数

use_former_res = 0;

%% 计算数据维度大小
fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;z_span = linspace(10,20,11);
G2 = zeros(length(x_span),length(y_span),length(z_span));
%% 传感器位置
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
x3 = 0;y3 = -L;z3 = 0.3;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];

%未知变量
syms sigmat1 sigmat2 sigmat3
syms sigmas
syms eta12 eta13 eta23
syms M
%% 最小二乘解析解情况：将cov(sigmati,sigmatj)估计值作为已知量，将其余变量作为未知量求解
%% 0-给出各变量的真值
var_value = [sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23];%变量列表，eta初始化为0，eta必须取常数而不是变量
real_value = [30e-3, 30e-3, 30e-3,5e-3, 0.2, 0.15, 0.1];%变量对应的真实值标签，可以看到eta并不是0
real_valuecell = num2cell(real_value);
[sigmat1_real,sigmat2_real,sigmat3_real,sigmas_real,eta12_real,eta13_real,eta23_real] = deal(real_valuecell{:});

Mont_times = 50;
NAN_ratio = 0.99;
err_list = [0.01,0.5,1,1.5,2,2.5,3,5,8,12,16];
SNR_LIST = [20,24,28,32,36,40,1e3];
err_idx = 1; 
snr_idx = length(SNR_LIST);
for err_idx = 1:1:length(err_list)
% for snr_idx = 1:1:length(SNR_LIST)-1
    CHOSEN_IDX_LIST = [];
    err = err_list(err_idx);
    SNR_PARA = SNR_LIST(snr_idx);
    save_path = ['test5_data/scene4/err_',num2str(err),'_SNR_',num2str(SNR_PARA),'/'];
    if ~exist(save_path,'dir')
        mkdir(save_path)
    end
    for i=1:1:Mont_times
        CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round(NAN_ratio*numel(G2)))];
    end

    [sigmat1_list,sigmat2_list,sigmat3_list,sigmas_list,eta12_list,eta13_list,eta23_list] = deal([]);%用于存放不同iter的估计值
    for iter = 1:1:Mont_times
        %% 最小二乘解
        %% 按照比例抽取部分G作为观测量
        ALL_ELE = 1:numel(G2);
        CHOSEN_IDX = CHOSEN_IDX_LIST(iter,:);
        unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
        %% 1-给出测量值矩阵
        [row_indices, col_indices, z_indices] = ind2sub(size(G2), unselected_elements);%注意这里的row指引的是y轴，col是x轴
        meas_matrix = [x_span(col_indices)'+0.01, y_span(row_indices)'+0.01, z_span(z_indices)'+0.001];%[x,y,z,]

        %% 2-将cov(sigmati,sigmatj)估计值作为已知量，将其余变量作为未知量求解
        % 2- 根据估计的相关系数eta，将sigmat和sigmas作为位置变量，求解最小二乘解析结果
        
        disp_str = ['err:',num2str(err),'----','SNR_PARA:',num2str(SNR_PARA),'----','mont_time:',num2str(iter),'----'];
        disp(disp_str)
        eta12_initial = eta12_real+eta12_real*10^(-SNR_PARA/20)*randn;
        eta13_initial = eta13_real+eta13_real*10^(-SNR_PARA/20)*randn;
        eta23_initial = eta23_real+eta23_real*10^(-SNR_PARA/20)*randn;
        var_value_cal1 = [sigmat1,sigmat1,sigmat1,sigmas,eta12_initial,eta13_initial,eta23_initial];%实际代入公式计算的变量列表
        coeffs = [];
        constant = [];
        b = [];
        for i=1:1:size(meas_matrix,1)
            pos_i = meas_matrix(i,1:3);
            % 计算G关于各变量的偏微分以及G_cal关于变量的表达式，是真实表达式
            [~,~,~,G_eq] = cal_diff(pos_i,sensor,var_value);
            % G的真实值和测量值
            G_real = double(subs(G_eq, var_value,real_value)); %真实值
            G_mea = G_real+err*randn;%实际测量值
            % 在表达式中代入变量和预设的已知量，得到计算的G的表达式
            % G_Cal = subs(G_eq, var_value,var_value_cal1);
            G_Cal = subs(G_eq, var_value,var_value_cal1);

            var_name = [sigmat1,sigmas];%G的表达式中有哪些多项式的项
            constant = [constant;double(subs(G_Cal, var_name,zeros(1,length(var_name))))];%求常数项式系数
            var_name = [sigmat1^2,sigmas^2];%G的表达式中有哪些多项式的项
            coeff = double(expr_coeff(G_Cal, var_name));
            coeffs =  [coeffs;coeff];
            b = [b;G_mea];
        end

        res1 = pinv(coeffs)*(b-constant);
        sigmt1_res = sqrt(res1(1));sigmat1_list = [sigmat1_list,sigmt1_res];
        sigmas_res = sqrt(res1(2));sigmas_list = [sigmas_list,sigmas_res];
    end
    res_all = [sigmat1_list;sigmas_list];
    save([save_path,'res_all.mat'],'res_all')
end
%% 加载上面的结果，并画图分析
% if use_former_res%使用之前的实验结果做分析
% files = dir('scene1/G无噪/*.mat');
% for i=1:1:length(files)
%     str = ['scene1/',files(i).name];
%     load(str)
% end
%
% figure
% plot(1:1:iter_N,sigmat1_real*ones(1,iter_N),'r-');hold on
% plot(1:1:iter_N,sigmat1_list,'r.-');hold on
% plot(1:1:iter_N,sigmat2_real*ones(1,iter_N),'b-');hold on
% plot(1:1:iter_N,sigmat2_list,'b.-');hold on
% plot(1:1:iter_N,sigmat3_real*ones(1,iter_N),'g-');hold on
% plot(1:1:iter_N,sigmat3_list,'g.-');hold on
% % legend('sigmat1-real','sigmat1-esti','sigmat2-real','sigmat2-esti','sigmat3-real','sigmat3-esti')
% leg = legend('$\sigma_{\Delta t_1}$','$\tilde{\sigma}_{\Delta t_1}$','$\sigma_{\Delta t_2}$','$\tilde{\sigma}_{\Delta t_2}$',...
%     '$\sigma_{\Delta t_3}$','$\tilde{\sigma}_{\Delta t_3}$');
% set(leg,'Interpreter','latex')
% figure
% plot(1:1:iter_N,eta12_real*ones(1,iter_N),'r-');hold on
% plot(1:1:iter_N,eta12_list,'r.-');hold on
% plot(1:1:iter_N,eta13_real*ones(1,iter_N),'b-');hold on
% plot(1:1:iter_N,eta13_list,'b.-');hold on
% plot(1:1:iter_N,eta23_real*ones(1,iter_N),'g-');hold on
% plot(1:1:iter_N,eta23_list,'g.-');hold on
% % legend('eta12-real','eta12-esti','eta13-real','eta13-esti','eta23-real','eta23-esti')
% leg = legend('$\eta_{12}$','$\tilde{\eta}_{12}$','$\eta_{13}$','$\tilde{\eta}_{13}$','$\eta_{23}$','$\tilde{\eta}_{23}$');
% set(leg,'Interpreter','latex')
% end
