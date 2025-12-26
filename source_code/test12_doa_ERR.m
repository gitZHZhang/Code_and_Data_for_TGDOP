clear
close all

%% 加载原始数据（和test4的数据相同）
load test12_data/doa/G2.mat 
load test12_data/doa/Sigma2_x.mat
load test12_data/doa/Sigma2_y.mat 
load test12_data/doa/Sigma2_z.mat
%% 传感器位置
xt = 0;yt = 0;zt = 0;
x1 = 640;y1 = 1070;z1 = -35;
x2 = -900;y2 = -180;z2 = -27;
x3 = 1000;y3 = -660;z3 = -35;
% z1=zt;z2=zt;z3=zt;%不考虑高程差的仿真结果
% p=10;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];

fs = 100; %降采样倍数
x_span = -4000:fs:4000;y_span = -4000:fs:4000;z_span = linspace(500,5000,11);
% for k =1:1:length(z_span)
%     for i = 1:1:length(x_span)
%         for j = 1:1:length(y_span)
%             m=x_span(i)+0.01;
%             n=y_span(j)+0.01;
%             p=z_span(k)+0.001;%height=10m
%             emitter = [m,n,p];
%             value.sigma_theta1 = ones(1,4).*0.1/180*pi;
%             value.sigma_theta2 = ones(1,4).*0.08/180*pi;
%             value.sigma_s = 1e-3;
%             [Gx,Gy,Gz,G] = cal_gdop_doa(emitter,sensor,value);
%             recon_tensor(j,i,k) = G;
%         end
%     end
% end
vars_real = [0.1,0.08,1,0.5,0,0,0];%theta1 theta2 (第3个为1没有任何意义，只是为了保证正定)
% mlrankest(Sigma2_x)


CHOSEN_IDX_LIST = [];
Mont_times = 15;
NAN_ratio_list = {'0.8','0.85','0.9','0.93','0.95','0.97','0.99'};
ERR_list = {'1','2','3','5','8','12','16'};
% NAN_ratio_list = {'0.97'};
for index_list = 1:1:length(ERR_list)%TTTTTTTTTTTTTTTTTT
CHOSEN_IDX_LIST = [];
ERR = str2num(ERR_list{index_list});
NAN_ratio = 0.99;%TTTTTTTTTTTTTTTTTT
% NAN_ratio = 0.8; %80 85 90 93 95 97 99
for i=1:1:Mont_times
    CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round(NAN_ratio*numel(G2)))];
end

%% 和别的算法作比较:Proposed/FNN/kriging/SVT/NNM-T
Algorithm_list = {'Kriging','RBF','NNM-T','Proposed','GeneralBTD','GeneralGDOP','GeneralGDOP2'};
% Algorithm_list = {'GeneralGDOP2'};
Algorithm_list = {'Proposed'};
for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
ContrastAlgorithm=Algorithm_list{algorithm_idx};
frob_list_all_Mont.errG = []; %存储每次蒙特卡洛实验的重构误差  
frob_list_all_Mont.errGx = []; %存储每次蒙特卡洛实验的重构误差 
frob_list_all_Mont.errGy = []; %存储每次蒙特卡洛实验的重构误差 
frob_list_all_Mont.errGz = []; %存储每次蒙特卡洛实验的重构误差 
save_path = ['test12_data/doa/',ContrastAlgorithm,'/fs_10_err_',ERR_list{index_list},'/'];
min_frob_list = 1e8;%当froblist均值最小时存储相应的重构张量
for mont_times = 1:1:Mont_times  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次start
disp_str = [ContrastAlgorithm,':',num2str(mont_times),'%%%%%%%%%%%%%%%%%%'];
disp(disp_str)
%% 按照比例抽取部分G作为观测量
incomplete_T = G2+ERR*randn(size(G2));%0.01km标准差的噪声
incomplete_Tx = Sigma2_x+0.01*randn(size(Sigma2_x));%0.01km标准差的噪声
incomplete_Ty = Sigma2_y+0.01*randn(size(Sigma2_y));%0.01km标准差的噪声
incomplete_Tz = Sigma2_z+0.01*randn(size(Sigma2_z));%0.01km标准差的噪声

ALL_ELE = 1:numel(incomplete_T);
% CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
% load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
CHOSEN_IDX = CHOSEN_IDX_LIST(mont_times,:);
unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
%画出观测数据在全部区域的分布
a = zeros(1,numel(incomplete_T));
a(unselected_elements)=1;
a_3d = reshape(a,size(G2,1),size(G2,2),size(G2,3));

incomplete_T(CHOSEN_IDX) = NaN; %将G2中的值设置为未知
incomplete_Tx(CHOSEN_IDX) = NaN;
incomplete_Ty(CHOSEN_IDX) = NaN;
incomplete_Tz(CHOSEN_IDX) = NaN;

if strcmp(ContrastAlgorithm,'Proposed')
    incomplete_T2x = fmt(incomplete_Tx);
    incomplete_T2y = fmt(incomplete_Ty);
    incomplete_T2z = fmt(incomplete_Tz);
    size_tens = incomplete_T2x.size;
    % size_tens = size(incomplete_T);
    L1 = [5 5 3];
    L2 = [6 6 3];
    L3 = [7 7 3];
    % L1 = [3 3 1];
    % L2 = [3 3 1];
    % L3 = [3 3 2];
    
    model= struct;
    if(NAN_ratio<0.8)
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
    else
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
    end
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
    recon_tensor = sqrt(Sigma2_x_hat.^2 + Sigma2_y_hat.^2 + Sigma2_z_hat.^2);

    frob_list1 = [];
    frob_listx = [];
    frob_listy = [];
    frob_listz = [];
    for i = 1:1:size(recon_tensor,3)
        frob_list1 = [frob_list1,frob(recon_tensor(:,:,i)-G2(:,:,i))];
        frob_listx = [frob_listx,frob(Sigma2_x_hat(:,:,i)-Sigma2_x(:,:,i))];
        frob_listy = [frob_listy,frob(Sigma2_y_hat(:,:,i)-Sigma2_y(:,:,i))];
        frob_listz = [frob_listz,frob(Sigma2_z_hat(:,:,i)-Sigma2_z(:,:,i))];
    end
    frob_list.errG = frob_list1;
    frob_list.errGx = frob_listx;
    frob_list.errGy = frob_listy;
    frob_list.errGz = frob_listz;
    recon_T.G = recon_tensor;
    recon_T.Gx = Sigma2_x_hat;
    recon_T.Gy = Sigma2_y_hat;
    recon_T.Gz = Sigma2_z_hat;
elseif strcmp(ContrastAlgorithm,'GeneralBTD')
    incomplete_T2x = fmt(incomplete_Tx);
    incomplete_T2y = fmt(incomplete_Ty);
    incomplete_T2z = fmt(incomplete_Tz);
    size_tens = incomplete_T2x.size;
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
    recon_tensor = sqrt(Sigma2_x_hat.^2 + Sigma2_y_hat.^2 + Sigma2_z_hat.^2);
    frob_list1 = [];
    frob_listx = [];
    frob_listy = [];
    frob_listz = [];
    for i = 1:1:size(recon_tensor,3)
        frob_list1 = [frob_list1,frob(recon_tensor(:,:,i)-G2(:,:,i))];
        frob_listx = [frob_listx,frob(Sigma2_x_hat(:,:,i)-Sigma2_x(:,:,i))];
        frob_listy = [frob_listy,frob(Sigma2_y_hat(:,:,i)-Sigma2_y(:,:,i))];
        frob_listz = [frob_listz,frob(Sigma2_z_hat(:,:,i)-Sigma2_z(:,:,i))];
    end
    frob_list.errG = frob_list1;
    frob_list.errGx = frob_listx;
    frob_list.errGy = frob_listy;
    frob_list.errGz = frob_listz;
    recon_T.G = recon_tensor;
    recon_T.Gx = Sigma2_x_hat;
    recon_T.Gy = Sigma2_y_hat;
    recon_T.Gz = Sigma2_z_hat;
end



if strcmp(ContrastAlgorithm,'GeneralGDOP')||strcmp(ContrastAlgorithm,'GeneralGDOP2')
    cor_noise = cor_noise_gen(mat2cell(vars_real',ones(1,length(vars_real))),length(unselected_elements));
    eta12_esti = corr(cor_noise(:,1),cor_noise(:,2));
    eta13_esti = corr(cor_noise(:,1),cor_noise(:,3));
    eta23_esti = corr(cor_noise(:,2),cor_noise(:,3));
    sigma_theta1 = sqrt(var(cor_noise(:,1)));
    sigma_theta2 = sqrt(var(cor_noise(:,2)));
    sigma_s_esti = 1;
    Sigma2_x_hat = zeros(size(G2));
    Sigma2_y_hat = zeros(size(G2));
    Sigma2_z_hat = zeros(size(G2));
    if ~strcmp(ContrastAlgorithm,'GeneralGDOP')
        snr = 20-(NAN_ratio-0.8)*50;
        sigma_theta1 = sigma_theta1+sigma_theta1/sqrt(snr)*randn;%10dB信噪比
        sigma_theta2 = sigma_theta2+sigma_theta2/sqrt(snr)*randn;%10dB信噪比
        sigma_s_esti = sigma_s_esti+sigma_s_esti/sqrt(snr)*randn;%10dB信噪比
    end
    for k =1:1:length(z_span)
        for i = 1:1:length(x_span)
            for j = 1:1:length(y_span)
                m=x_span(i)+0.01;
                n=y_span(j)+0.01;
                p=z_span(k)+0.001;%height=10m
                emitter = [m,n,p];
                value.sigma_theta1 = ones(1,4).*sigma_theta1/180*pi;
                value.sigma_theta2 = ones(1,4).*sigma_theta2/180*pi;
                value.sigma_s = sigma_s_esti;
                [Gx,Gy,Gz,G] = cal_gdop_doa(emitter,sensor,value);
                recon_tensor(j,i,k) = sqrt(G);
                Sigma2_x_hat(j,i,k) = sqrt(Gx);
                Sigma2_y_hat(j,i,k) = sqrt(Gy);
                Sigma2_z_hat(j,i,k) = sqrt(Gz);
            end
        end
    end
    recon_tensor = recon_tensor+ERR*randn(size(recon_tensor));
    Sigma2_x_hat = Sigma2_x_hat+ERR*randn(size(Sigma2_x_hat));%0.01km标准差的噪声
    Sigma2_y_hat = Sigma2_y_hat+ERR*randn(size(Sigma2_y_hat));%0.01km标准差的噪声
    Sigma2_z_hat = Sigma2_z_hat+ERR*randn(size(Sigma2_z_hat));%0.01km标准差的噪声
    frob_list1 = [];
    frob_listx = [];
    frob_listy = [];
    frob_listz = [];
    for i = 1:1:size(recon_tensor,3)
        frob_list1 = [frob_list1,frob(recon_tensor(:,:,i)-G2(:,:,i))];
        frob_listx = [frob_listx,frob(Sigma2_x_hat(:,:,i)-Sigma2_x(:,:,i))];
        frob_listy = [frob_listy,frob(Sigma2_y_hat(:,:,i)-Sigma2_y(:,:,i))];
        frob_listz = [frob_listz,frob(Sigma2_z_hat(:,:,i)-Sigma2_z(:,:,i))];
    end
    frob_list.errG = frob_list1;
    frob_list.errGx = frob_listx;
    frob_list.errGy = frob_listy;
    frob_list.errGz = frob_listz;
    recon_T.G = recon_tensor;
    recon_T.Gx = Sigma2_x_hat;
    recon_T.Gy = Sigma2_y_hat;
    recon_T.Gz = Sigma2_z_hat;
elseif strcmp(ContrastAlgorithm,'Kriging')
    %% VS kriging（成功）
    [row_indices, col_indices, z_indices] = ind2sub(size(incomplete_T), unselected_elements);%注意这里的row指引的是y轴，col是x轴
    meas_matrix = [x_span(col_indices)'+0.01, y_span(row_indices)'+0.01, z_span(z_indices)'+0.001,...
        incomplete_Tx(unselected_elements)',incomplete_Ty(unselected_elements)',incomplete_Tz(unselected_elements)'];%[x,y,z,G]
    z_indices_unique = unique(z_indices);%按照不同的高度（z）分别处理，每一个z类似于一个矩阵，分别应用矩阵处理算法
    indices_cell = cell(1,length(z_indices_unique));
    
    frob_list1 = [];
    frob_listx = [];
    frob_listy = [];
    frob_listz = [];
    Sigma2_x_hat = zeros(size(G2));
    Sigma2_y_hat = zeros(size(G2));
    Sigma2_z_hat = zeros(size(G2));
    for p=1:1:3
        for i =1:1:length(z_indices_unique)
            indices_cell{i} = z_indices==z_indices_unique(i);
            meas_matrix_i = meas_matrix(indices_cell{i},[1:3,p+3]);
            %S存储了点位坐标值，Y为观测值
            S = meas_matrix_i(:,1:2);
            Y = meas_matrix_i(:,4);
            theta = [10 10]; lob = [1e-1 1e-1]; upb = [20 20];
            %变异函数模型为高斯模型
            [dmodel, ~] = dacefit(S, Y, @regpoly2, @corrgauss, theta, lob, upb);

            X = gridsamp([-4000 -4000;4000 4000], size(G2,1));%创建一个szie*size的正方形格网，标注范围为-4000,4000
            %格网点的预测值返回在矩阵YX中，预测点的均方根误差返回在矩阵MSE中
            [YX,MSE] = predictor(X, dmodel);
            X1 = reshape(X(:,1),size(G2,1),size(G2,1)); X2 = reshape(X(:,2),size(G2,1),size(G2,1));
            YX = reshape(YX, size(X1));         %size(X1)=400*400
            if p==1
                Sigma2_x_hat(:,:,i) = YX;
                frob_listx = [frob_listx,frob(Sigma2_x_hat(:,:,i)-Sigma2_x(:,:,i))];
            elseif p==2
                Sigma2_y_hat(:,:,i) = YX;
                frob_listy = [frob_listy,frob(Sigma2_y_hat(:,:,i)-Sigma2_y(:,:,i))];
            else
                Sigma2_z_hat(:,:,i) = YX;
                frob_listz = [frob_listz,frob(Sigma2_z_hat(:,:,i)-Sigma2_z(:,:,i))];
            end           
        end
    end
    recon_tensor = sqrt(Sigma2_x_hat.^2 + Sigma2_y_hat.^2 + Sigma2_z_hat.^2);
    frob_list1 = [];
    for i = 1:1:size(recon_tensor,3)
        frob_list1 = [frob_list1,frob(recon_tensor(:,:,i)-G2(:,:,i))];
    end
    frob_list.errG = frob_list1;
    frob_list.errGx = frob_listx;
    frob_list.errGy = frob_listy;
    frob_list.errGz = frob_listz;
    recon_T.G = recon_tensor;
    recon_T.Gx = Sigma2_x_hat;
    recon_T.Gy = Sigma2_y_hat;
    recon_T.Gz = Sigma2_z_hat;
elseif strcmp(ContrastAlgorithm,'RBF')
    [row_indices, col_indices, z_indices] = ind2sub(size(incomplete_T), unselected_elements);%注意这里的row指引的是y轴，col是x轴
    meas_matrix = [x_span(col_indices)'+0.01, y_span(row_indices)'+0.01, z_span(z_indices)'+0.001,...
        incomplete_Tx(unselected_elements)',incomplete_Ty(unselected_elements)',incomplete_Tz(unselected_elements)'];
    indices_cell = cell(1,length(z_indices_unique));
    frob_list1 = [];
    frob_listx = [];
    frob_listy = [];
    frob_listz = [];
    Sigma2_x_hat = zeros(size(G2));
    Sigma2_y_hat = zeros(size(G2));
    Sigma2_z_hat = zeros(size(G2));
    for p=1:1:3
        for i =1:1:length(z_indices_unique)
            indices_cell{i} = z_indices==z_indices_unique(i);
            meas_matrix_i = meas_matrix(indices_cell{i},[1:3,p+3]);
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
            z3=RBF2(X0,Y0,values,X,Y,'gaussian',Rs,2,'rbf',0.0001);
            YX = reshape(z3, size(G2,1),size(G2,2));
            if p==1
                Sigma2_x_hat(:,:,i) = YX;
                frob_listx = [frob_listx,frob(Sigma2_x_hat(:,:,i)-Sigma2_x(:,:,i))];
            elseif p==2
                Sigma2_y_hat(:,:,i) = YX;
                frob_listy = [frob_listy,frob(Sigma2_y_hat(:,:,i)-Sigma2_y(:,:,i))];
            else
                Sigma2_z_hat(:,:,i) = YX;
                frob_listz = [frob_listz,frob(Sigma2_z_hat(:,:,i)-Sigma2_z(:,:,i))];
            end
        end
    end
    recon_tensor = sqrt(Sigma2_x_hat.^2 + Sigma2_y_hat.^2 + Sigma2_z_hat.^2);
    frob_list1 = [];
    for i = 1:1:size(recon_tensor,3)
        frob_list1 = [frob_list1,frob(recon_tensor(:,:,i)-G2(:,:,i))];
    end
    frob_list.errG = frob_list1;
    frob_list.errGx = frob_listx;
    frob_list.errGy = frob_listy;
    frob_list.errGz = frob_listz;
    recon_T.G = recon_tensor;
    recon_T.Gx = Sigma2_x_hat;
    recon_T.Gy = Sigma2_y_hat;
    recon_T.Gz = Sigma2_z_hat;
elseif strcmp(ContrastAlgorithm,'NNM-T')
    frob_list1 = [];
    frob_listx = [];
    frob_listy = [];
    frob_listz = [];
    Sigma2_x_hat = zeros(size(G2));
    Sigma2_y_hat = zeros(size(G2));
    Sigma2_z_hat = zeros(size(G2));
    for p=1:1:3
        for i = 1:1:size(G2,3)
            %1-插值
            if p==1
                A = incomplete_Tx(:,:,i);
            elseif p==2
                A = incomplete_Ty(:,:,i);
            else
                A = incomplete_Tz(:,:,i);
            end
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
            if p==1
                Sigma2_x_hat(:,:,i) = A2;
                frob_listx = [frob_listx,frob(Sigma2_x_hat(:,:,i)-Sigma2_x(:,:,i))];
            elseif p==2
                Sigma2_y_hat(:,:,i) = A2;
                frob_listy = [frob_listy,frob(Sigma2_y_hat(:,:,i)-Sigma2_y(:,:,i))];
            else
                Sigma2_z_hat(:,:,i) = A2;
                frob_listz = [frob_listz,frob(Sigma2_z_hat(:,:,i)-Sigma2_z(:,:,i))];
            end
        end
    end
    recon_tensor = sqrt(Sigma2_x_hat.^2 + Sigma2_y_hat.^2 + Sigma2_z_hat.^2);
    frob_list1 = [];
    for i = 1:1:size(recon_tensor,3)
        frob_list1 = [frob_list1,frob(recon_tensor(:,:,i)-G2(:,:,i))];
    end
    frob_list.errG = frob_list1;
    frob_list.errGx = frob_listx;
    frob_list.errGy = frob_listy;
    frob_list.errGz = frob_listz;
    recon_T.G = recon_tensor;
    recon_T.Gx = Sigma2_x_hat;
    recon_T.Gy = Sigma2_y_hat;
    recon_T.Gz = Sigma2_z_hat;
end
frob_list_all_Mont.errG = [frob_list_all_Mont.errG;frob_list.errG];
frob_list_all_Mont.errGx = [frob_list_all_Mont.errGx;frob_list.errGx];
frob_list_all_Mont.errGy = [frob_list_all_Mont.errGy;frob_list.errGy];
frob_list_all_Mont.errGz = [frob_list_all_Mont.errGz;frob_list.errGz];
% frob_list_all_Mont = [frob_list_all_Mont;frob_list];
if mean(frob_list.errG)<min_frob_list
    min_frob_list = mean(frob_list.errG);
    save([save_path,'recon_T.mat'],'recon_T')
    if strcmp(ContrastAlgorithm,'Proposed')
        save([save_path,'sol.mat'],'sol')
    end
end
end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次end
save([save_path,'frob_list_all_Mont.mat'],'frob_list_all_Mont')
end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法end
end
% size_core = mlrankest(Sigma2_z);%计算sizecore


