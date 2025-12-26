close all
clear all
%%%%%%%%%%%%%%%
% 同样是tdoa定位场景，和test8相比有不一样的误差参数和基站构型
%%%%%%%%%%%%%%%
% helperMIMOBER(4,randn(10),10)
%% 加载原始数据（和test4的数据相同）
load test15_data/tdoa1/G2.mat 
load test15_data/tdoa1/Sigma2_x.mat
load test15_data/tdoa1/Sigma2_y.mat 
load test15_data/tdoa1/Sigma2_z.mat

% load test11_data/tdoa1/G2.mat 
% load test11_data/tdoa1/Sigma2_x.mat
% load test11_data/tdoa1/Sigma2_y.mat 
% load test11_data/tdoa1/Sigma2_z.mat

% 
% % mlrankest(G2)

% % mlrankest(Sigma2_x.^2)
% % mlrankest(Sigma2_y.^2)
% % mlrankest(Sigma2_z.^2)
%% 传感器位置

xt = 0;yt = 0;zt = 0;
x1 = 640;y1 = 1070;z1 = -35;
x2 = -900;y2 = -180;z2 = -27;
x3 = 1000;y3 = -660;z3 = -35;
% L=30;
% xt = 0;yt = 0;zt = 0;
% x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
% x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
% x3 = 0;y3 = -L;z3 = 0.3;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];

fs = 100; %降采样倍数
x_span = -4000:fs:4000;y_span = -4000:fs:4000;z_span = linspace(500,5000,11);
% % vars_real = [30e-3,30e-3,30e-3,5e-3,0.3,0.3,0.3];
vars_real = [18e-3,20e-3,25e-3,0.5,-0.3,0.5,-0.2]; %us us us m

% for k =1:1:length(z_span)
%     for i = 1:1:length(x_span)
%         for j = 1:1:length(y_span)
%             m=x_span(i)+0.01;
%             n=y_span(j)+0.01;
%             p=z_span(k)+0.001;%height=10m
%             emitter = [m,n,p];
%             % if (k==3 && i<40 && i>15 && j<40 && j>15)  
%             if (k==3e3 && i<40 && j<40 )
%                 lambda = 80e-3; %指数分布均值us
%                 % exp_n = exprnd(lambda, 1, 1);
%                 vars_in = vars_real;
%                 vars_in(1) = vars_in(1)+lambda;
%             % elseif (k==1 && i<65 && i>40 && j<65 && j>40)
%             elseif (k>=1 && k<=3 && i>40 && j>40)
%                 lambda = 100e-3; %指数分布均值lambda，方差lambda^2，因此标准差lambda
%                 % exp_n = exprnd(lambda, 1, 1);
%                 vars_in = vars_real;
%                 vars_in(2) = vars_in(2)+lambda;
%             else
%                 vars_in = vars_real;
%             end
%             [Gx_eq,Gy_eq,Gz_eq,G_eq] = cal_diff(emitter,sensor,vars_in);%时间单位us，距离单位m
%             G2(j,i,k) = G_eq;
%             Sigma2_x(j,i,k) = Gx_eq;
%             Sigma2_y(j,i,k) = Gy_eq;
%             Sigma2_z(j,i,k) = Gz_eq;
%         end
%     end
% end
% save('test15_data/tdoa1/G2.mat','G2')
% save('test15_data/tdoa1/Sigma2_x.mat','Sigma2_x')
% save('test15_data/tdoa1/Sigma2_y.mat','Sigma2_y')
% save('test15_data/tdoa1/Sigma2_z.mat','Sigma2_z')
% G2 = sqrt(G2);
% figure
% [c,handle]=contour(x_span,y_span,G2(:,:,1)',20);
% clabel(c,handle);hold on
% plot3(sensor(:,1),sensor(:,2),sensor(:,3),'rp')
% legend('contour','sensors')
% slice_idx = [1,5,10];
% figure
% title_str = [];
% plot_tensor = 10*log10(G2);
% plot_3D_heatmap(plot_tensor,slice_idx,[],x_span,y_span,z_span); hold on
% figure
mlrankest(Sigma2_x)
mlrankest(Sigma2_y)
mlrankest(Sigma2_z)
% size_core = mlrankest(Sigma2_x);%计算sizecore
% size_core = [3,3,2];
% [UXhat,SXhat,output]=lmlra(Sigma2_x,size_core); 
% [UYhat,SYhat,output]=lmlra(Sigma2_y,size_core); 
% size_core = [4,4,2];
% [UZhat,SZhat,output]=lmlra(Sigma2_z,size_core); 


% G2 = G2(41:81, 41:81, 1:11);
% G2 = G2(1:41, 1:41, 1:11);
sub_G2 = G2(41:81, 41:81, 1:11);

Mont_times = 5;
ERR_list = {'2','6','10','14','18','22'};
% NAN_ratio_list = {'0.95'};
% NAN_ratio = 0.99; %80 85 90 93 95 97 99
for index_list = 1:1:length(ERR_list)%TTTTTTTTTTTTTTTTTT
CHOSEN_IDX_LIST = [];
CHOSEN_IDX_LIST2 = [];
snr = str2num(ERR_list{index_list});%TTTTTTTTTTTTTTTTTT
% NAN_ratio = str2num(NAN_ratio_list{index_list});%TTTTTTTTTTTTTTTTTT
NAN_ratio = 0.8; %80 85 90 93 95 97 99
for i=1:1:Mont_times
    CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round((NAN_ratio)*numel(G2)))];%TTTTTTTTTTTTTTTTTT
    CHOSEN_IDX_LIST2 = [CHOSEN_IDX_LIST2;randperm(numel(sub_G2),round((NAN_ratio)*numel(sub_G2)))];%TTTTTTTTTTTTTTTTTT
    % sub_tensor = G2(41:81, 41:81, 1:3);
    % num_samples_to_extract = 100;
    % random_indices_1d = randperm(numel(sub_tensor), num_samples_to_extract);
    % [dim_x, dim_y, dim_z] = size(sub_tensor);
    % [idx_x_sub, idx_y_sub, idx_z_sub] = ind2sub([dim_x, dim_y, dim_z], random_indices_1d);
    % idx_x_original = idx_x_sub + 41 - 1;
    % idx_y_original = idx_y_sub + 41 - 1;
    % idx_z_original = idx_z_sub + 1 - 1;
    % ind = sub2ind(size(G2), idx_x_original, idx_y_original,idx_z_original);
    % nanidx = randperm(numel(G2),round((NAN_ratio)*numel(G2)));
    % nanidx_select = setdiff(nanidx, ind);
    % CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;nanidx_select];%TTTTTTTTTTTTTTTTTT
end

%% 和别的算法作比较:Proposed/FNN/kriging/SVT/NNM-T
Algorithm_list = {'kriging','GeneralGDOP','RBF','NNM-T','Proposed','GeneralBTD','GeneralGDOP2'};
Algorithm_list = {'kriging','GeneralGDOP','RBF','Proposed','GeneralBTD','GeneralGDOP2'};
Algorithm_list = {'NNM-T'};
for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
ContrastAlgorithm=Algorithm_list{algorithm_idx};
frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差  
save_path = ['test15_data/tdoa1/',ContrastAlgorithm,'/fs_10_snr_',ERR_list{index_list},'/'];
% save_path = ['test11_data/tdoa1/',ContrastAlgorithm,'/fs_10_err_',ERR_list{index_list},'/'];%TTTTTTTTTTTTTTTTTT
min_frob_list = 1e8;%当froblist均值最小时存储相应的重构张量
for mont_times = 1:1:Mont_times  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次start
disp_str = [ContrastAlgorithm,':',num2str(mont_times),'%%%%%%%%%%%%%%%%%%'];
disp(disp_str)
%% 按照比例抽取部分G作为观测量
sigma_e = G2*10^(-snr/10);
noise = sigma_e.*randn(size(G2));

incomplete_T = G2+noise;%0.01km标准差的噪声
ALL_ELE = 1:numel(incomplete_T);

% CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
% load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
CHOSEN_IDX = CHOSEN_IDX_LIST(mont_times,:);
CHOSEN_IDX2 = CHOSEN_IDX_LIST2(mont_times,:);
unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
%画出观测数据在全部区域的分布
a = zeros(1,numel(incomplete_T));
a(unselected_elements)=1;
a_3d = reshape(a,size(G2,1),size(G2,2),size(G2,3));
% plot_title = 'The 3-D distribution of the selected grid.';
% [~] = plot_3D_tensor(a_3d,x_span,y_span,z_span,plot_title);
incomplete_T(CHOSEN_IDX) = NaN; %将G2中的值设置为未知


incomplete_T1 = incomplete_T;
incomplete_T1(41:81,41:81,1:11)=NaN;

sigma_e = sub_G2*10^(-snr/10);
noise = sigma_e.*randn(size(sub_G2));
incomplete_T2 = sub_G2+noise;
incomplete_T2(CHOSEN_IDX2) = NaN;


if strcmp(ContrastAlgorithm,'GeneralGDOP')||strcmp(ContrastAlgorithm,'GeneralGDOP2')
    cor_noise = cor_noise_gen(mat2cell(vars_real',ones(1,length(vars_real))),length(unselected_elements));
    eta12_esti = corr(cor_noise(:,1),cor_noise(:,2));
    eta13_esti = corr(cor_noise(:,1),cor_noise(:,3));
    eta23_esti = corr(cor_noise(:,2),cor_noise(:,3));
    sigma_t1 = sqrt(var(cor_noise(:,1)));
    sigma_t2 = sqrt(var(cor_noise(:,2)));
    sigma_t3 = sqrt(var(cor_noise(:,3)));
    sigma_s_esti = 1;
    if ~strcmp(ContrastAlgorithm,'GeneralGDOP')
        snr = 20-(NAN_ratio-0.8)*50;
        sigma_s_esti = sigma_s_esti+sigma_s_esti/sqrt(snr)*randn;%10dB信噪比
    end
    vars = [sigma_t1,sigma_t2,sigma_t3,sigma_s_esti,eta12_esti,eta13_esti,eta23_esti];
    % snr_para = -150*NAN_ratio+190;
    % vars = awgn(vars_real,snr_para);
    % vars = vars_real+str2num(ERR_list{index_list}).*randn(size(vars_real));
    for k =1:1:length(z_span)
        for i = 1:1:length(x_span)
            for j = 1:1:length(y_span)    
                m=x_span(i)+0.01;
                n=y_span(j)+0.01;
                p=z_span(k)+0.001;%height=10m       
                emitter = [m,n,p];
                [Gx_eq,Gy_eq,Gz_eq,G_eq] = cal_diff(emitter,sensor,vars);%时间单位us，距离单位km
                recon_tensor(j,i,k) = G_eq;
            end
        end
    end
    sigma_e = recon_tensor*10^(-snr/10);
    noise = sigma_e.*randn(size(G2));
    recon_tensor = recon_tensor+noise;
    frob_list = [];
    for i = 1:1:size(recon_tensor,3)
        frob_list = [frob_list,frob(sqrt(recon_tensor(:,:,i))-sqrt(G2(:,:,i)))];
    end
    mean(frob_list)
elseif strcmp(ContrastAlgorithm,'Proposed')
    incomplete_T1 = fmt(incomplete_T1);
    L1 = [3 3 3];
    L2 = [4 4 3];
    L3 = [5 5 4];
    L1 = [8 8 3];
    L2 = [8 8 3];
    L3 = [8 8 3];
    poly=12; %
    type = 'polyBTD_nonneg'; %typelist:typicalBTD_nonneg/ typicalBTD/ polyBTD_nonneg/ polyBTD
    [sol,output] = runBTD(incomplete_T1,[L1;L2;L3],poly,type);
    [A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res,A3_res,B3_res,C3_res,S3_res] = deal(sol.factors{:});
    Sigma2_x_hat = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
    Sigma2_y_hat = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
    Sigma2_z_hat = tmprod(S3_res,{A3_res,B3_res,C3_res},1:3);
    recon_tensor = Sigma2_x_hat + Sigma2_y_hat + Sigma2_z_hat;
    
    incomplete_T2 = fmt(incomplete_T2);
    [sol,output] = runBTD(incomplete_T2,[L1;L2;L3],poly,type);
    [A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res,A3_res,B3_res,C3_res,S3_res] = deal(sol.factors{:});
    Sigma2_x_hat2 = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
    Sigma2_y_hat2 = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
    Sigma2_z_hat2 = tmprod(S3_res,{A3_res,B3_res,C3_res},1:3);
    recon_tensor2 = Sigma2_x_hat2 + Sigma2_y_hat2 + Sigma2_z_hat2;

    recon_tensor(41:81,41:81,1:11) = recon_tensor2;
    Sigma2_x_hat(41:81,41:81,1:11) = Sigma2_x_hat2;
    Sigma2_y_hat(41:81,41:81,1:11) = Sigma2_y_hat2;
    Sigma2_z_hat(41:81,41:81,1:11) = Sigma2_z_hat2;


    err_T = abs(G2-recon_tensor);
    err_T2 = log((err_T - min(min(min(err_T)))) / max(max(max(err_T))));
    plot_title='The 3-D distribution of $log(| \hat{\textbf{e}}|)$';
    % [~] = plot_3D_tensor(err_T2,x_span,y_span,z_span,plot_title);


    frob_list = [];
    for i = 1:1:size(recon_tensor,3)
        frob_list = [frob_list,frob(sqrt(recon_tensor(:,:,i))-sqrt(G2(:,:,i)))];
    end
    mean(frob_list) 

    % figure();
    % subplot(2,1,1)
    % [c,handle]=contour(x_span,y_span,real(sqrt(G2(:,:,1))),20);
    % clabel(c,handle);
    % title('Real distribution of $G$','interpreter','latex');
    % xlabel('x(km)');
    % ylabel('y(km)');
    % % leg = title('$\sigma_{y}^2的真实分布（降采样后）$','interpreter','latex');
    % % set(leg,'Interpreter','latex')
    % subplot(2,1,2)
    % [c,handle]=contour(x_span,y_span,real(sqrt(recon_tensor(:,:,1))),20);
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
elseif strcmp(ContrastAlgorithm,'GeneralBTD')
    incomplete_T2 = fmt(incomplete_T);
    size_tens = incomplete_T2.size;
    % size_tens = size(incomplete_T);
    L1 = [5 5 3];
    L2 = [6 6 3];
    L3 = [7 7 3];
    model= struct;
    %% 随机初值
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
    model.factors={ {'A1',@struct_nonneg},...
        {'B1', @struct_nonneg},...
        {'C1', @struct_nonneg},'S1',... 
        {'A2', @struct_nonneg},...
        {'B2', @struct_nonneg},...
        {'C2', @struct_nonneg},'S2',...
        {'A3', @struct_nonneg},...
        {'B3', @struct_nonneg},...
        {'C3', @struct_nonneg},'S3' };
    model.factorizations.mybtd.data=incomplete_T2;
    model.factorizations.mybtd.btd={{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    sdf_check(model,'print');
    [sol,output] = sdf_nls(model);
    [A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res,A3_res,B3_res,C3_res,S3_res] = deal(sol.factors{:});
    Sigma2_x_hat = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
    Sigma2_y_hat = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
    Sigma2_z_hat = tmprod(S3_res,{A3_res,B3_res,C3_res},1:3);
    recon_tensor = Sigma2_x_hat + Sigma2_y_hat + Sigma2_z_hat;
    frob_list = [];
    for i = 1:1:size(recon_tensor,3)
        frob_list = [frob_list,frob(sqrt(recon_tensor(:,:,i))-sqrt(G2(:,:,i)))];
    end
    mean(frob_list)
end
if strcmp(ContrastAlgorithm,'FNN')
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
    recon_tensor = reshape(target_G,dim);
    res_tensor_3d_1 = recon_tensor(:,:,1);
    figure();
    [c,handle]=contour(x_span,y_span,res_tensor_3d_1',20);
    clabel(c,handle);
    title('The reconstructed distribution of $G$ using AI','interpreter','latex');
    xlabel('x(km)');
    ylabel('y(km)');
    disp(frob(G2(:,:,1:11)-recon_tensor))
    % plot_heat_map(res_tensor_3d_1,fs)
    % plot_heat_map(G2(:,:,1),fs)
    frob_list = [];
    for i = 1:1:dim(3)
        frob_list = [frob_list,frob(recon_tensor(:,:,i)-G2(:,:,i))];
    end

elseif strcmp(ContrastAlgorithm,'kriging')
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
        theta = [10 10]; lob = [20 20]; upb = [50 50];
        %变异函数模型为高斯模型
        [dmodel, ~] = dacefit(S, Y, @regpoly2, @corrgauss, theta, lob, upb);
        % [dmodel, ~] = dacefit(S, Y, @regpoly2, @corrgauss, theta);
        
        X = gridsamp([-4000 -4000;4000 4000], size(G2,1));%创建一个szie*size的正方形格网，标注范围为-4000,4000
        % X=[83.731	32.36];     %单点预测的实现
        %格网点的预测值返回在矩阵YX中，预测点的均方根误差返回在矩阵MSE中
        [YX,MSE] = predictor(X, dmodel);
        X1 = reshape(X(:,1),size(G2,1),size(G2,1)); X2 = reshape(X(:,2),size(G2,1),size(G2,1));
        YX = reshape(YX, size(X1));         %size(X1)=400*400
        % figure(1), mesh(X1, X2, YX)         %绘制预测表面
        % hold on,
        % plot3(S(:,1),S(:,2),Y,'.k', 'MarkerSize',10)    %绘制原始散点数据
        % hold off
        % figure(2),mesh(X1, X2, reshape(MSE,size(X1)));  %绘制每个点的插值误差大小
        % figure();
        % [c,handle]=contour(X1(1,:),X2(:,1),YX,20);
        % clabel(c,handle);
        % title('The reconstructed distribution of $G$ using Kriging','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % % disp(frob(G2(:,:,i)-YX))
        recon_tensor(:,:,i) = YX;
        frob_list = [frob_list,frob(sqrt(YX)-sqrt(G2(:,:,i)))];
    end
    mean(frob_list)
 elseif strcmp(ContrastAlgorithm,'SVT')
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
         % figure();
         % [c,handle]=contour(x_span,y_span,X,20);
         % clabel(c,handle);
         % title('The reconstructed distribution of $G$ using SVT','interpreter','latex');
         % xlabel('x(km)');
         % ylabel('y(km)');
         % figure();
         % [c,handle]=contour(x_span,y_span,G2(:,:,i),20);
         % clabel(c,handle);
         % title('The real distribution of $G$','interpreter','latex');
         % xlabel('x(km)');
         % ylabel('y(km)');
         % plot_heat_map(X,fs)
         % plot_heat_map(G2(:,:,i),fs)
         frob_list = [frob_list,frob(X-G2(:,:,i))];
         recon_tensor(:,:,i) = X;
     end
    elseif strcmp(ContrastAlgorithm,'RBF')
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
            X0 = meas_matrix_i(:,1);
            Y0 = meas_matrix_i(:,2);
            values = meas_matrix_i(:,4);
            [X,Y]=meshgrid(linspace(-4000,4000,size(G2,1)));
            %二维RBF插值
            if NAN_ratio>=95
                Rs = 50;
            else
                Rs = 35;
            end
            Rs = 300;
            z3=RBF2(X0,Y0,values,X,Y,'gaussian',Rs,2,'rbf',0.0001);
            YX = reshape(z3, size(G2,1),size(G2,2)); 
            % figure();
            % [c,handle]=contour(x_span,y_span,YX,20);
            % clabel(c,handle);
            % title('The reconstructed distribution of $G$ using rbf','interpreter','latex');
            % xlabel('x(km)');
            % ylabel('y(km)');
            % disp(frob(G2(:,:,i)-YX))
            recon_tensor(:,:,i) = YX;
            frob_list = [frob_list,frob(sqrt(YX)-sqrt(G2(:,:,i)))];
        end
        mean(frob_list)
    elseif strcmp(ContrastAlgorithm,'NNM-T')
        frob_list = [];
        recon_tensor = zeros(size(G2)); 
        for i = 1:1:size(G2,3)
            %1-插值
            A = incomplete_T(:,:,i);
            %找出矩阵A中某半径内的全部连通域
            Meas_matrix = a_3d(:,:,i);
            neighborhoodSize = 10;%局部线性拟合的半径
            kernel = ones(neighborhoodSize);
            InstructionMatrix = conv2(Meas_matrix, kernel, 'same');
            % 使用 bwlabel 函数标记连通区域
            [L, num] = bwlabel(InstructionMatrix);
            stats = regionprops(L, 'Area', 'PixelIdxList');
            %对每一个连通域所在的子矩阵单独进行局部线性拟合插值
            for j=1:1:num
                pixelIdx = stats(j).PixelIdxList;
                %当前连通区域的两个角对应的矩阵中的索引
                area_begin = pixelIdx(1);
                [row_begin, col_begin] = ind2sub(size(InstructionMatrix), area_begin);
                area_end = pixelIdx(end);
                [row_end, col_end] = ind2sub(size(InstructionMatrix), area_end);
                area_A = A(min(row_begin,row_end):max(row_begin,row_end),min(col_begin,col_end):max(col_begin,col_end));
                % 矩阵中的非NaN值的位置
                [rows, cols] = find(~isnan(area_A));
                rows = reshape(rows,length(rows),1);
                cols = reshape(cols,length(cols),1);
                if length(rows)>3
                    values = area_A(~isnan(area_A));
                    values = reshape(values,length(values),1);
                    % 创建插值对象
                    F = scatteredInterpolant(rows, cols, values, 'natural');
                    % 定义插值网格
                    [gridRows, gridCols] = find(isnan(area_A));
                    % 执行插值
                    interpValues = F(gridRows, gridCols);
                    % [X_axis, Y_axis, interpValues]=griddata(rows, cols, values, gridRows, gridCols);
                    % 将插值结果放入原矩阵的对应位置
                    if ~isempty(interpValues)
                        area_A(isnan(area_A)) = interpValues;
                        A(min(row_begin,row_end):max(row_begin,row_end),min(col_begin,col_end):max(col_begin,col_end))=area_A;
                    end
                end
            end
            %2-矩阵复原
            [m,n]=size(A);
            % 开始CVX模型
            cvx_begin
                variable A2(m,n);
                minimize( norm_nuc(A2) );
                subject to
                    norm(A2(~isnan(A)) - A(~isnan(A)))<=100;
                    % A2(~isnan(A)) == A(~isnan(A));
            cvx_end
            recon_tensor(:,:,i) = A2;
            frob_list = [frob_list,frob(sqrt(A2)-sqrt(G2(:,:,i)))];
        end
        mean(frob_list)
end
frob_list_all_Mont = [frob_list_all_Mont;frob_list];
if mean(frob_list)<min_frob_list
    min_frob_list = mean(frob_list);
    save([save_path,'recon_tensor.mat'],'recon_tensor')
    if strcmp(ContrastAlgorithm,'Proposed')
        save([save_path,'sol.mat'],'sol')
    end
end
end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次end
save([save_path,'frob_list_all_Mont.mat'],'frob_list_all_Mont')
end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法end
end

% figure();
% [c,handle]=contour(x_span,y_span,recon_tensor(:,:,1),20);
% % clabel(c,handle);
% title('The reconstructed distribution of $G$','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');
% figure();
% [c,handle]=contour(x_span,y_span,G2(:,:,1),20);
% clabel(c,handle);
% title('The real distribution of $G$','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');


% 生成均值为2000小时的指数分布随机数
lambda = 200;
exp_n = exprnd(lambda, 1000, 1);
histogram(exp_n)
mean_lifetime = mean(exp_n);
variance_lifetime = var(exp_n);
% 计算均值和方差
