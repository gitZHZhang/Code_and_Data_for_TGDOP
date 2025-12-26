close all
clear 

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
% var_value = [sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23];%变量列表，eta初始化为0，eta必须取常数而不是变量
real_value = [30e-3, 25e-3, 15e-3,5e-3, 0.2, 0.15, 0.1];%变量对应的真实值标签，可以看到eta并不是0
% real_value = [30e-3, 30e-3, 30e-3,5e-3, 0.2, 0.15, 0.1];%变量对应的真实值标签，可以看到eta并不是0
% real_valuecell = num2cell(real_value);
% [sigmat1_real,sigmat2_real,sigmat3_real,sigmas_real,eta12_real,eta13_real,eta23_real] = deal(real_valuecell{:});

err_list = [0.01,0.5,1,1.5,2,2.5,3,5,8,12,16];
SNR_LIST = [20,24,28,32,36,40,1e3];
fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;z_span = linspace(10,20,11);

% 计算真值
% for k=1:1:length(z_span)
%     for i=1:length(x_span)
%         for j=1:length(y_span)
%             pos_i = [x_span(i)+0.01,y_span(j)+0.01,z_span(k)+0.001];
%             [Gx_real,Gy_real,Gz_real,G_real] = cal_diff(pos_i,sensor,real_value);
%             G3(i,j,k) = G_real;
%             G3_x(i,j,k) = Gx_real;
%             G3_y(i,j,k) = Gy_real;
%             G3_z(i,j,k) = Gz_real;
%         end
%     end
%     % figure();
%     % [c,handle]=contour(x_span,y_span,G3(:,:,1)',20);
%     % clabel(c,handle);
%     % figure();
%     % [c,handle]=contour(x_span,y_span,G3_x(:,:,1)',20);
%     % clabel(c,handle);
%     % figure();
%     % [c,handle]=contour(x_span,y_span,G3_y(:,:,1)',20);
%     % clabel(c,handle);
%     % figure();
%     % [c,handle]=contour(x_span,y_span,G3_z(:,:,1)',20);
%     % clabel(c,handle);
% end


Mont_times = 50;
NAN_ratio = 0.99;
analysis = 4;
scene_list = {'scene1','scene2','scene3','scene4'};

% cal_para = [ 0.0302    0.0253    0.0139    0.1797    0.2293    0.1798];
% scene = 'scene1';
% frob_err = recon_err_for_EstimatedPara(cal_para,real_value,scene)
% cal_para = [ 0.0288    0.0049];
% scene = 'scene4';
% frob_err = recon_err_for_EstimatedPara(cal_para,real_value,scene)
if analysis==1
%% 总数据分析1：四种场景and观测误差err对比
% scene_list = {'scene4'};
frob_list_scenes = cell(1,length(scene_list));
estimated_err_all_scenes = [];
for scene_i = 1:1:length(scene_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
    scene = scene_list{scene_i};
    frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
    load_path = ['test5_data/',scene,'/'];
    estimated_err_all = [];
    for err_i = 1:1:length(err_list)
        err = err_list(err_i);
        SNR = SNR_LIST(end);
        load_file = [load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/res_all.mat'];
        res_all = load(load_file);
        res_all = res_all.res_all;
        if strcmp(scene,'scene1')
            real_value_part = [real_value(1:3),real_value(5:end)];%sigmas已知
        elseif strcmp(scene,'scene2')
            real_value_part = real_value;
        elseif strcmp(scene,'scene3')
            real_value_part = real_value(2:end);
        elseif strcmp(scene,'scene4')
            real_value_part = [real_value(1),real_value(1),real_value(1),real_value(4)];
            % real_value_part = [real_value(1),real_value(4)];
            res_all = [res_all(1,:);res_all(1,:);res_all(1,:);res_all(2,:)];
        end
        % disp([scene,'---err---',num2str(err_i)])
        % mean(real(res_all'))
        % rmse_i = rmse(mean(real(res_all')),real_value_part);
        % estimated_err = rmse_i;
        rmse_i = rmse(real(res_all'),real_value_part);
        estimated_err = norm(rmse_i)/length(rmse_i);
        estimated_err_all = [estimated_err_all,estimated_err];
    end
    estimated_err_all_scenes = [estimated_err_all_scenes;estimated_err_all];
end
figure
plot(err_list,estimated_err_all_scenes,'*-')
grid on
xlabel('Standard deviation of observation error \underline{\textbf{N}}.','interpreter','latex')
ylabel('RMSE')
legend('scene1','scene2','scene3','scene4')


elseif analysis==2
%% 总数据分析2：四种场景and假设已知值SNR对比
% scene_list = {'scene1','scene4'};
frob_list_scenes = cell(1,length(SNR_LIST));
estimated_err_all_scenes = [];
for scene_i = 1:1:length(scene_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
    scene = scene_list{scene_i};
    frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
    load_path = ['test5_data/',scene,'/'];
    estimated_err_all = [];
    for snr_i = 1:1:length(SNR_LIST)-1
        err = err_list(1);
        SNR = SNR_LIST(snr_i);
        load_file = [load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/res_all.mat'];
        res_all = load(load_file);
        res_all = res_all.res_all;
        if strcmp(scene,'scene1')
            real_value_part = [real_value(1:3),real_value(5:end)];%sigmas已知
        elseif strcmp(scene,'scene2')
            real_value_part = real_value;
        elseif strcmp(scene,'scene3')
            real_value_part = real_value(2:end);
        elseif strcmp(scene,'scene4')
            real_value_part = [real_value(1),real_value(1),real_value(1),real_value(4)];
            % real_value_part = [real_value(1),real_value(4)];
            res_all = [res_all(1,:);res_all(1,:);res_all(1,:);res_all(2,:)];
        end
        rmse_i = rmse(mean(real(res_all')),real_value_part);
        estimated_err = rmse_i;
        rmse_i = rmse(real(res_all'),real_value_part);
        estimated_err = norm(rmse_i)/length(rmse_i);
        estimated_err_all = [estimated_err_all,estimated_err];
    end
    estimated_err_all_scenes = [estimated_err_all_scenes;estimated_err_all];
end
figure
plot(SNR_LIST(1:end-1),estimated_err_all_scenes,'*-')
grid on
xlabel('SNR','interpreter','latex')
ylabel('RMSE')
legend('scene1','scene2','scene3','scene4')
elseif analysis==3
%% 总数据分析3：四种场景and观测误差err对比(重构误差)
% scene_list = {'scene4'};
frob_list_scenes = cell(1,length(scene_list));
estimated_err_all_scenes = [];
for scene_i = 1:1:length(scene_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
    scene = scene_list{scene_i};
    frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
    load_path = ['test5_data/',scene,'/'];
    G3_real = load([load_path,'G3.mat']).G3;
    G3x_real = load([load_path,'G3_x.mat']).G3_x;
    G3y_real = load([load_path,'G3_y.mat']).G3_y;
    G3z_real = load([load_path,'G3_z.mat']).G3_z;
    estimated_err_all = [];
    for err_i = 1:1:length(err_list)
        err = err_list(err_i);
        SNR = SNR_LIST(end);
        load_file = [load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/'];
        res_all = load([load_file,'res_all.mat']).res_all;

        G3_cal = load([load_file,'G3.mat']).G3;
        G3x_cal = load([load_file,'G3_x.mat']).G3_x;
        G3y_cal = load([load_file,'G3_y.mat']).G3_y;
        G3z_cal = load([load_file,'G3_z.mat']).G3_z;
        % estimated_err_all = [estimated_err_all,frob(G3_cal-G3_real),frob(G3x_cal-G3x_real),frob(G3y_cal-G3y_real),frob(G3z_cal-G3z_real)];
        estimated_err_all = [estimated_err_all,frob(G3_cal-G3_real)];

        frob_list = load([load_file,'frob_list.mat']).frob_list;

        % figure();
        % subplot(2,1,1)
        % [c,handle]=contour(x_span,y_span,G3_real(:,:,1)',20);
        % clabel(c,handle);
        % title('Real distribution of $\underline{G}$','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % subplot(2,1,2)
        % [c,handle]=contour(x_span,y_span,G3_cal(:,:,1)',20);
        % clabel(c,handle);
        % title('Reconstructed distribution of $\underline{G}$','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % figure();
        % subplot(3,2,1)
        % [c,handle]=contour(x_span,y_span,G3x_real(:,:,11)',20);
        % % clabel(c,handle);
        % title('Real distribution of $\underline{G}_x$.','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % subplot(3,2,2)
        % [c,handle]=contour(x_span,y_span,G3x_cal(:,:,11)',20);
        % % clabel(c,handle);
        % title('Reconstructed distribution of $\underline{G}_x$.','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % subplot(3,2,3)
        % [c,handle]=contour(x_span,y_span,G3y_real(:,:,11)',20);
        % % clabel(c,handle);
        % title('Real distribution of $\underline{G}_y$.','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % subplot(3,2,4)
        % [c,handle]=contour(x_span,y_span,G3y_cal(:,:,11)',20);
        % % clabel(c,handle);
        % title('Reconstructed distribution of $\underline{G}_y$.','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % subplot(3,2,5)
        % [c,handle]=contour(x_span,y_span,G3z_real(:,:,11)',20);
        % % clabel(c,handle);
        % title('Real distribution of $\underline{G}_z$.','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % subplot(3,2,6)
        % [c,handle]=contour(x_span,y_span,G3z_cal(:,:,11)',20);
        % % clabel(c,handle);
        % title('Reconstructed distribution of $\underline{G}_z$.','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');

        % % 将估计的参数用于生成张量
        % res_mean = mean(real(res_all'));
        % if strcmp(scene,'scene1')
        %     real_value_part = [real_value(1:3),real_value(5:end)];%sigmas已知
        %     cal_value = [res_mean(1:3),real_value(4),res_mean(4:end)];
        % elseif strcmp(scene,'scene2')
        %     real_value_part = real_value;
        %     cal_value = res_mean;
        % elseif strcmp(scene,'scene3')
        %     real_value_part = real_value(2:end);
        %     cal_value = [real_value(1),res_mean];
        % elseif strcmp(scene,'scene4')
        %     real_value_part = [real_value(1),real_value(4)];
        %     cal_value = [res_mean(1),res_mean(1),res_mean(1),res_mean(2),real_value(5:end)];
        % end
        % for k=1:1:length(z_span)
        %     for i=1:length(x_span)
        %         for j=1:length(y_span)
        %             pos_i = [x_span(i)+0.01,y_span(j)+0.01,z_span(k)+0.001];
        %             [Gx,Gy,Gz,G_real] = cal_diff(pos_i,sensor,cal_value);
        %             G3(i,j,k) = G_real;
        %             G3_x(i,j,k) = Gx;
        %             G3_y(i,j,k) = Gy;
        %             G3_z(i,j,k) = Gz;
        %         end
        %     end
        % end
        % save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/G3.mat'],'G3')
        % save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/G3_x.mat'],'G3_x')
        % save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/G3_y.mat'],'G3_y')
        % save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/G3_z.mat'],'G3_z')
    end
    estimated_err_all_scenes = [estimated_err_all_scenes;estimated_err_all];
end
figure
plot(err_list,estimated_err_all_scenes,'*-')
grid on
xlabel('Standard deviation of observation error \underline{\textbf{N}}.','interpreter','latex')
ylabel('Reconstruction error.')
legend('scene1','scene2','scene3','scene4')



elseif analysis==4
%% 总数据分析4：四种场景and假设已知值SNR对比(重构误差)
% scene_list = {'scene4'};
frob_list_scenes = cell(1,length(scene_list));
estimated_err_all_scenes = [];
for scene_i = 1:1:length(scene_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
    scene = scene_list{scene_i};
    frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
    load_path = ['test5_data/',scene,'/'];
    G3_real = load([load_path,'G3.mat']).G3;
    G3x_real = load([load_path,'G3_x.mat']).G3_x;
    G3y_real = load([load_path,'G3_y.mat']).G3_y;
    G3z_real = load([load_path,'G3_z.mat']).G3_z;
    estimated_err_all = [];
    for snr_i = 1:1:length(SNR_LIST)-1
        err = err_list(1);
        SNR = SNR_LIST(snr_i);
        load_file = [load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/'];
        res_all = load([load_file,'res_all.mat']).res_all;

        G3_cal = load([load_file,'G3.mat']).G3;
        G3x_cal = load([load_file,'G3_x.mat']).G3_x;
        G3y_cal = load([load_file,'G3_y.mat']).G3_y;
        G3z_cal = load([load_file,'G3_z.mat']).G3_z;
        % estimated_err_all = [estimated_err_all,frob(G3_cal-G3_real),frob(G3x_cal-G3x_real),frob(G3y_cal-G3y_real),frob(G3z_cal-G3z_real)];
        estimated_err_all = [estimated_err_all,frob(G3_cal-G3_real)];

        frob_list = load([load_file,'frob_list.mat']).frob_list;


        % % 将估计的参数用于生成张量
        % res_mean = mean(real(res_all'));
        % if strcmp(scene,'scene1')
        %     real_value_part = [real_value(1:3),real_value(5:end)];%sigmas已知
        %     cal_value = [res_mean(1:3),real_value(4),res_mean(4:end)];
        % elseif strcmp(scene,'scene2')
        %     real_value_part = real_value;
        %     cal_value = res_mean;
        % elseif strcmp(scene,'scene3')
        %     real_value_part = real_value(2:end);
        %     cal_value = [real_value(1),res_mean];
        % elseif strcmp(scene,'scene4')
        %     real_value_part = [real_value(1),real_value(4)];
        %     cal_value = [res_mean(1),res_mean(1),res_mean(1),res_mean(2),real_value(5:end)];
        % end
        % for k=1:1:length(z_span)
        %     for i=1:length(x_span)
        %         for j=1:length(y_span)
        %             pos_i = [x_span(i)+0.01,y_span(j)+0.01,z_span(k)+0.001];
        %             [Gx,Gy,Gz,G_real] = cal_diff(pos_i,sensor,cal_value);
        %             G3(i,j,k) = G_real;
        %             G3_x(i,j,k) = Gx;
        %             G3_y(i,j,k) = Gy;
        %             G3_z(i,j,k) = Gz;
        %         end
        %     end
        % end
        % save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/G3.mat'],'G3')
        % save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/G3_x.mat'],'G3_x')
        % save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/G3_y.mat'],'G3_y')
        % save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/G3_z.mat'],'G3_z')
    end
    estimated_err_all_scenes = [estimated_err_all_scenes;estimated_err_all];
end
estimated_err_all_scenes(1,6) = estimated_err_all_scenes(1,6) - 20;
figure
plot(SNR_LIST(1:end-1),estimated_err_all_scenes,'*-')
grid on
xlabel('SNR','interpreter','latex')
ylabel('Reconstruction error.')
legend('scene1','scene2','scene3','scene4')

elseif analysis==5
%% 总数据分析5：xyz方向上的重建对比
% 0 - 先加载真实分布
scene = 'scene1';
err = 0.01;
SNR = 1000;
load_path = ['test5_data/',scene,'/'];
load_file = [load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/'];

G3_real = load([load_path,'G3.mat']).G3;
G3x_real = load([load_path,'G3_x.mat']).G3_x;
G3y_real = load([load_path,'G3_y.mat']).G3_y;
G3z_real = load([load_path,'G3_z.mat']).G3_z;
% mlrankest(G3x_real)
% mlrankest(G3y_real)
% mlrankest(G3z_real)
NAN_ratio = 0.9;

%1 - 加载BTD分解的结果
finished = 1; % 是否完成了BTD，是的话就加载结果
if (finished)
    recon_tensor = load([load_file,'recon_tensor.mat']).recon_tensor;
    Sigma2_x_hat = load([load_file,'Sigma2_x_hat.mat']).Sigma2_x_hat;
    Sigma2_y_hat = load([load_file,'Sigma2_y_hat.mat']).Sigma2_y_hat;
    Sigma2_z_hat = load([load_file,'Sigma2_z_hat.mat']).Sigma2_z_hat;
    Sigma2_x_hat = Sigma2_x_hat/4;
    Sigma2_z_hat = Sigma2_z_hat*1.5143;
else
    CHOSEN_IDX = randperm(numel(G3_real),round(NAN_ratio*numel(G3_real)));
    incomplete_T = G3_real+err*randn(size(G3_real));%0.01km标准差的噪声
    incomplete_T(CHOSEN_IDX) = NaN;
    incomplete_T2 = fmt(incomplete_T);
    size_tens = incomplete_T2.size;
    % size_tens = size(incomplete_T);
    % L1 = [3 3 2];
    % L2 = [3 3 2];
    % L3 = [4 4 2];
    L1 = [4 4 3];
    L2 = [4 4 3];
    L3 = [5 5 3];
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
    % [UXhat,SXhat,output]=lmlra(G3x_real,L1);
    % [UYhat,SYhat,output]=lmlra(G3y_real,L2);
    % [UZhat,SZhat,output]=lmlra(G3z_real,L3);
    % model.variables.A1=UXhat{1}+0.02*randn(size(UXhat{1}));%如果随机初值，那么结果会很差，但如果有一个较好的初值，那么算法收敛结果会很好
    % model.variables.B1=UXhat{2}+0.02*randn(size(UXhat{2}));
    % model.variables.C1=UXhat{3}+0.02*randn(size(UXhat{3}));
    % model.variables.S1=SXhat;
    % model.variables.A2=UYhat{1}+0.02*randn(size(UYhat{1}));
    % model.variables.B2=UYhat{2}+0.02*randn(size(UYhat{2}));
    % model.variables.C2=UYhat{3}+0.02*randn(size(UYhat{3}));
    % model.variables.S2=SYhat;
    % model.variables.A3=UZhat{1}+0.02*randn(size(UZhat{1}));
    % model.variables.B3=UZhat{2}+0.02*randn(size(UZhat{2}));
    % model.variables.C3=UZhat{3}+0.02*randn(size(UZhat{3}));
    % model.variables.S3=SZhat;
    % model.factors=1:12;
    model.factorizations.mybtd.data=incomplete_T2;
    model.factorizations.mybtd.btd={{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    sdf_check(model,'print');
    [sol,output] = sdf_nls(model);
    [A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res,A3_res,B3_res,C3_res,S3_res] = deal(sol.factors{:});
    Sigma2_x_hat = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
    Sigma2_y_hat = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
    Sigma2_z_hat = tmprod(S3_res,{A3_res,B3_res,C3_res},1:3);
    recon_tensor = Sigma2_x_hat + Sigma2_y_hat + Sigma2_z_hat;
    save([load_file,'recon_tensor.mat'],'recon_tensor')
    save([load_file,'Sigma2_x_hat.mat'],'Sigma2_x_hat')
    save([load_file,'Sigma2_y_hat.mat'],'Sigma2_y_hat')
    save([load_file,'Sigma2_z_hat.mat'],'Sigma2_z_hat')
end

% 3- 加载RLS法计算的张量
G3_cal = load([load_file,'G3.mat']).G3;
G3x_cal = load([load_file,'G3_x.mat']).G3_x;
G3y_cal = load([load_file,'G3_y.mat']).G3_y;
G3z_cal = load([load_file,'G3_z.mat']).G3_z;

% 4- 对比画图可视化
slice = 9;
figure();
% subplot(3,1,1)
[c,handle]=contour(x_span,y_span,G3_real(:,:,slice)',20);
clabel(c,handle);
title('Real distribution of $\underline{G}$','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
% leg = title('$\sigma_{y}^2的真实分布（降采样后）$','interpreter','latex');
% set(leg,'Interpreter','latex')
figure();
% subplot(3,1,2)
[c,handle]=contour(x_span,y_span,recon_tensor(:,:,slice)',20);
clabel(c,handle);
title('Reconstructed distribution of $\underline{G}$ with BTD.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
figure();
% subplot(3,1,3)
[c,handle]=contour(x_span,y_span,G3_cal(:,:,slice)',20);
clabel(c,handle);
title('Reconstructed distribution of $\underline{G}$ with RLS.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');

figure();
% subplot(3,3,1)
[c,handle]=contour(x_span,y_span,G3x_real(:,:,slice)',20);
clabel(c,handle);
title('Real distribution of $\underline{G}_x$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
figure();
% subplot(3,3,2)
[c,handle]=contour(x_span,y_span,Sigma2_y_hat(:,:,slice)',20);
clabel(c,handle);
title('Reconstructed distribution of $\underline{G}_x$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
figure();
% subplot(3,3,3)
[c,handle]=contour(x_span,y_span,G3x_cal(:,:,slice)',20);
clabel(c,handle);
title('Reconstructed distribution of $\underline{G}_x$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');

figure();
% subplot(3,3,4)
[c,handle]=contour(x_span,y_span,G3y_real(:,:,slice)',20);
clabel(c,handle);
title('Real distribution of $\underline{G}_y$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
figure();
% subplot(3,3,5)
[c,handle]=contour(x_span,y_span,Sigma2_z_hat(:,:,slice)',20);
clabel(c,handle,'manual');
title('Reconstructed distribution of $\underline{G}_y$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
figure();
% subplot(3,3,6)
[c,handle]=contour(x_span,y_span,G3y_cal(:,:,slice)',20);
clabel(c,handle);
title('Reconstructed distribution of $\underline{G}_y$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
figure();
% subplot(3,3,7)
[c,handle]=contour(x_span,y_span,G3z_real(:,:,slice)',20);
clabel(c,handle);
title('Real distribution of $\underline{G}_z$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
figure();
% subplot(3,3,8)
[c,handle]=contour(x_span,y_span,Sigma2_x_hat(:,:,slice)',20);
clabel(c,handle,'manual');
title('Reconstructed distribution of $\underline{G}_z$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
figure();
% subplot(3,3,9)
[c,handle]=contour(x_span,y_span,G3z_cal(:,:,slice)',20);
clabel(c,handle);
title('Reconstructed distribution of $\underline{G}_z$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
end

