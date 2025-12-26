clear
close all

num_stations=4;
station_positions1=zeros(num_stations,3);
station_positions1(1,:)=[114.378753,38.612907,236];%174
station_positions1(2,:)=[114.386107,38.622557,200.577671288129];%171
station_positions1(3,:)=[114.3682315,38.611272,208.631661263305];%172
station_positions1(4,:)=[114.390267,38.60696983,200.687108302994];%173
st2=zeros(num_stations,3); % 存放xyz坐标
st2(1,:)=[0,0,236];%中央主楼和其他三站的xyz
st2(2,:)=[640.441400785537,1071.28855449371,200.577671288129];
st2(3,:)=[-916.435747950129,-181.451672162199,208.631661263305];
st2(4,:)=[1002.94238079953,-659.031578388016,200.687108302994];
st3 = st2;
st3(:,3) = st2(:,3) - st2(1,3);

pos_err = load('test9_data\4Err_data\pos_err.mat').pos_err;
pos_list2 = pos_err.cal_pos;
x_span = linspace(min(pos_list2(:,1)),1000,200);
y_span = linspace(min(pos_list2(:,2)),1000,200);
z_span = linspace(min(pos_list2(:,3)),max(pos_list2(:,3)),10);

%% 训练数据%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delay12_err = load('test9_data\4Err_data\delay12_err.mat').delay12_err;
delay13_err = load('test9_data\4Err_data\delay13_err.mat').delay13_err;
delay14_err = load('test9_data\4Err_data\delay14_err.mat').delay14_err;
% pos_err_train = load('test9_data\4Err_data\pos_err_train.mat').pos_err_train;
% pos_err_test = load('test9_data\4Err_data\pos_err_test.mat').pos_err_test;


delay12 = [];delay13 = [];delay14 = [];
for i =1:1:size(pos_err.index,1)
    idx_i = pos_err.index(i);
    delay12 = [delay12,delay12_err.err(delay12_err.index==idx_i)];
    delay13 = [delay13,delay13_err.err(delay13_err.index==idx_i)];
    delay14 = [delay14,delay14_err.err(delay14_err.index==idx_i)];
end
mean12 = mean(delay12);
sigma12 = sqrt(cov(delay12));
mean13 = mean(delay13);
sigma13 = sqrt(cov(delay13)); %单位ns
mean14 = mean(delay14);
sigma14 = sqrt(cov(delay14));
eta12 = corrcoef(delay12,delay13);eta12 = eta12(1,2);
eta13 = corrcoef(delay12,delay14);eta13 = eta13(1,2);
eta23 = corrcoef(delay13,delay14);eta23 = eta23(1,2);
rate = 3;
M=1;
train_rate = 0.8; %用于训练的数据占比
time_list = [];
ERR_list = [];
recon_rate_list = [];
for m_i = 1:1:M
test_data_index = randperm(size(pos_err.real_pos,1),floor(size(pos_err.real_pos,1)*(1-train_rate)));
train_data_index = setdiff(1:size(pos_err.real_pos,1), test_data_index);
pos_err_train.index = pos_err.index(train_data_index);
pos_err_train.cal_pos = pos_err.cal_pos(train_data_index,:);
pos_err_train.real_pos = pos_err.real_pos(train_data_index,:);
pos_err_train.err = pos_err.err(train_data_index,:);
pos_err_test.index = pos_err.index(test_data_index);
pos_err_test.cal_pos = pos_err.cal_pos(test_data_index,:);
pos_err_test.real_pos = pos_err.real_pos(test_data_index,:);
pos_err_test.err = pos_err.err(test_data_index,:);
time_list_i = [];
ERR_list_i = [];
recon_rate_list_i = [];
%% 开始用测试集数据进行算法对比
Algorithm_list = {'Proposed'};
for alg_i = 1:1:length(Algorithm_list)
    Algorithm = Algorithm_list{alg_i};
    % load_file = ['test10_data\rate=',num2str(train_rate),'/',Algorithm,'/'];
    disp(['Mont time = ',num2str(m_i),'%%%% Algorithm=',Algorithm])
    proposed_alg_list = {'proposed1','proposed1-2','proposed2'};
        %% 2-proposed\
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Proposed 1(拥有误差观测量G) %%%%%%%%%%%%%%%
        tic
        rate = 0;%单位放缩倍数，当取km时为6
        Err_XYZ = pos_err_train.err(:,end).^2;
        
        Err_XYZ_GDOP = [];
        G_esti = NaN(length(x_span),length(y_span),length(z_span));
        for i=1:1:size(pos_err_train.real_pos,1)
            pos_i = pos_err_train.real_pos(i,:);
            x_index = transform_value_to_index(x_span,pos_i(1));
            y_index = transform_value_to_index(y_span,pos_i(2));
            z_index = transform_value_to_index(z_span,pos_i(3));
            G_esti(x_index,y_index,z_index) = Err_XYZ(i);
        end
        incomplete_T2 = fmt(G_esti);
        size_tens = incomplete_T2.size;
        L1 = [5 4 4];
        L2 = [4 5 5];
        L3 = [5 5 4];
        % L1 = [10 4 4];
        % L2 = [4 10 4];
        % L3 = [4 4 4];
        model= struct;
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
        model.factorizations.mybtd.data=incomplete_T2;
        model.factorizations.mybtd.btd={{1,2,3,4},{5,6,7,8},{9,10,11,12}};
        sdf_check(model,'print');
        [sol,output] = sdf_nls(model);
        [A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res,A3_res,B3_res,C3_res,S3_res] = deal(sol.factors{:});
        Sigma2_x_hat = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
        Sigma2_y_hat = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
        Sigma2_z_hat = tmprod(S3_res,{A3_res,B3_res,C3_res},1:3);
        recon_tensor = Sigma2_x_hat + Sigma2_y_hat + Sigma2_z_hat;

        % % recon_tensor = load ('test10_data\GeneralBTD\recon_tensor.mat').recon_tensor;
        % % Sigma2_x_hat = load ('test10_data\GeneralBTD\Sigma2_x_hat.mat').Sigma2_x_hat;
        % % Sigma2_y_hat = load ('test10_data\GeneralBTD\Sigma2_y_hat.mat').Sigma2_y_hat;
        % % Sigma2_z_hat = load ('test10_data\GeneralBTD\Sigma2_z_hat.mat').Sigma2_z_hat;
        % % rate = 0;%单位放缩倍数，当取km时为6
        % Err_XYZ = pos_err_test.err(:,end).^2;
        % x_span = linspace(min(pos_list2(:,1)),1000,200);
        % y_span = linspace(min(pos_list2(:,2)),1000,200);
        % z_span = linspace(min(pos_list2(:,3)),max(pos_list2(:,3)),10);
        % G_esti = NaN(length(x_span),length(y_span),length(z_span));
        % for i=1:1:size(pos_err_test.real_pos,1)
        %     pos_i = pos_err_test.real_pos(i,:);
        %     x_index = transform_value_to_index(x_span,pos_i(1));
        %     y_index = transform_value_to_index(y_span,pos_i(2));
        %     z_index = transform_value_to_index(z_span,pos_i(3));
        %     G_esti(x_index,y_index,z_index) = Err_XYZ(i);
        % end
        % mea_data = G_esti(~isnan(G_esti));
        % esti_data = recon_tensor(~isnan(G_esti));
        % ERR1 = norm(sqrt(mea_data)-sqrt(esti_data));
        % time1 = toc;
        % recon_rate1 = sum(sum(sum(~isnan(recon_tensor))))/numel(recon_tensor);
        save(['test10_data/proposed/',proposed_alg_list{1},'/','recon_tensor.mat'],'recon_tensor')
        save(['test10_data/proposed/',proposed_alg_list{1},'/','Sigma2_x_hat.mat'],'Sigma2_x_hat')
        save(['test10_data/proposed/',proposed_alg_list{1},'/','Sigma2_y_hat.mat'],'Sigma2_y_hat')
        save(['test10_data/proposed/',proposed_alg_list{1},'/','Sigma2_z_hat.mat'],'Sigma2_z_hat')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Proposed 1-2(拥有多方向误差观测量G_p) %%%%%%%%%%%%%%%
        tic
        rate = 3;%单位放缩倍数，当取km时为6
        Err_XYZ = pos_err_train.err(:,end).^2;
        Err_x = pos_err_train.err(:,1).^2;
        Err_y = pos_err_train.err(:,2).^2;
        Err_z = pos_err_train.err(:,3).^2;
        Err_XYZ_GDOP = [];
        Gx_esti = NaN(length(x_span),length(y_span),length(z_span));
        Gy_esti = NaN(length(x_span),length(y_span),length(z_span));
        Gz_esti = NaN(length(x_span),length(y_span),length(z_span));
        for i=1:1:size(pos_err_train.real_pos,1)
            pos_i = pos_err_train.real_pos(i,:);
            x_index = transform_value_to_index(x_span,pos_i(1));
            y_index = transform_value_to_index(y_span,pos_i(2));
            z_index = transform_value_to_index(z_span,pos_i(3));
            Gx_esti(x_index,y_index,z_index) = Err_x(i);
            Gy_esti(x_index,y_index,z_index) = Err_y(i);
            Gz_esti(x_index,y_index,z_index) = Err_z(i);
        end
        incomplete_T2x = fmt(Gx_esti);
        incomplete_T2y = fmt(Gy_esti);
        incomplete_T2z = fmt(Gz_esti);
        size_tens = incomplete_T2x.size;

        L1 = [5 5 3];
        L2 = [6 6 3];
        L3 = [7 7 3];
        L1 = [10 5 3];
        L2 = [10 6 3];
        L3 = [10 7 3];
        model= struct;
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
        recon_tensor = Sigma2_x_hat + Sigma2_y_hat + Sigma2_z_hat;

        % % recon_tensor = load ('test10_data\GeneralBTD\recon_tensor.mat').recon_tensor;
        % % Sigma2_x_hat = load ('test10_data\GeneralBTD\Sigma2_x_hat.mat').Sigma2_x_hat;
        % % Sigma2_y_hat = load ('test10_data\GeneralBTD\Sigma2_y_hat.mat').Sigma2_y_hat;
        % % Sigma2_z_hat = load ('test10_data\GeneralBTD\Sigma2_z_hat.mat').Sigma2_z_hat;
        % rate = 3;%单位放缩倍数，当取km时为6
        % Err_XYZ = pos_err_test.err(:,end).^2*10^(-rate);
        % x_span = linspace(min(pos_list2(:,1)),1000,200);
        % y_span = linspace(min(pos_list2(:,2)),1000,200);
        % z_span = linspace(min(pos_list2(:,3)),max(pos_list2(:,3)),10);
        % G_esti = NaN(length(x_span),length(y_span),length(z_span));
        % for i=1:1:size(pos_err_test.real_pos,1)
        %     pos_i = pos_err_test.real_pos(i,:);
        %     x_index = transform_value_to_index(x_span,pos_i(1));
        %     y_index = transform_value_to_index(y_span,pos_i(2));
        %     z_index = transform_value_to_index(z_span,pos_i(3));
        %     G_esti(x_index,y_index,z_index) = Err_XYZ(i);
        % end
        % mea_data = G_esti(~isnan(G_esti));
        % esti_data = recon_tensor(~isnan(G_esti));
        % ERR1 = norm(sqrt(mea_data)-sqrt(esti_data));
        % time1 = toc;
        % recon_rate1 = sum(sum(sum(~isnan(recon_tensor))))/numel(recon_tensor);
        save(['test10_data/proposed/',proposed_alg_list{2},'/','recon_tensor.mat'],'recon_tensor')
        save(['test10_data/proposed/',proposed_alg_list{2},'/','Sigma2_x_hat.mat'],'Sigma2_x_hat')
        save(['test10_data/proposed/',proposed_alg_list{2},'/','Sigma2_y_hat.mat'],'Sigma2_y_hat')
        save(['test10_data/proposed/',proposed_alg_list{2},'/','Sigma2_z_hat.mat'],'Sigma2_z_hat')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Proposed 2(拥有误差估计值Θ) %%%%%%%%%%%%%%%
        tic
        Err_XYZ = pos_err_test.err(:,end).^2;
        res_G1 = [];
        sigmat2_initial = sigma13*1e-3;%单位us
        res1 = load ('test9_data\5Para_esti\res1.mat').res1;
        sigmt1_res = sqrt(res1(1));
        sigmt3_res = sqrt(res1(2));
        sigms_res = abs(sqrt(res1(3)))*1e3;
        eta12_res = res1(4)*0.01/sigmt1_res/sigmat2_initial;
        eta13_res = res1(5)/sigmt1_res/sigmt3_res;
        eta23_res = res1(6)*0.01/sigmat2_initial/sigmt3_res;
        vars = [sigmt1_res,sigmat2_initial,sigmt3_res,sigms_res,eta12_res,eta13_res,eta23_res];
        for k = 1:1:length(z_span)
            for i = 1:1:length(x_span)
                for j = 1:1:length(y_span)
                    pos_i = [x_span(i),y_span(j),z_span(k)];
                    [Gx_eq,Gy_eq,Gz_eq,G_eq] = cal_diff(pos_i,st3,vars);%时间单位us，距离单位km
                    recon_tensor(j,i,k) = G_eq;
                    Sigma2_x_hat(j,i,k) = Gx_eq;
                    Sigma2_y_hat(j,i,k) = Gy_eq;
                    Sigma2_z_hat(j,i,k) = Gz_eq;
                end
            end
        end
        save(['test10_data/proposed/',proposed_alg_list{3},'/','recon_tensor.mat'],'recon_tensor')
        save(['test10_data/proposed/',proposed_alg_list{3},'/','Sigma2_x_hat.mat'],'Sigma2_x_hat')
        save(['test10_data/proposed/',proposed_alg_list{3},'/','Sigma2_y_hat.mat'],'Sigma2_y_hat')
        save(['test10_data/proposed/',proposed_alg_list{3},'/','Sigma2_z_hat.mat'],'Sigma2_z_hat')
        % recon_tensor = G_cal;
        % % recon_tensor = load ('test10_data\GeneralGDOP\G_cal.mat').G_cal;
        % G_esti = NaN(length(x_span),length(y_span),length(z_span));
        % for i=1:1:size(pos_err_test.real_pos,1)
        %     pos_i = pos_err_test.real_pos(i,:);
        %     x_index = transform_value_to_index(x_span,pos_i(1));
        %     y_index = transform_value_to_index(y_span,pos_i(2));
        %     z_index = transform_value_to_index(z_span,pos_i(3));
        %     G_esti(x_index,y_index,z_index) = Err_XYZ(i);
        % end
        % mea_data = G_esti(~isnan(G_esti));
        % esti_data = recon_tensor(~isnan(G_esti));
        % time2=toc;
        % ERR2 = norm(sqrt(mea_data)-sqrt(esti_data));
        % recon_rate2 = sum(sum(sum(~isnan(recon_tensor))))/numel(recon_tensor);
        % ERR = [ERR1;ERR2];
        % time = [time1;time2];
        % recon_rate = [recon_rate1;recon_rate2];
        % % tic
        % % res_G2 = [];
        % % Err_XYZ = pos_err_test.err(:,end).^2*1e-6;
        % % sigmat2_initial = sigma13*1e-3;%单位us
        % % res1 = load ('test9_data\5Para_esti\res1.mat').res1;
        % % sigmt1_res = sqrt(res1(1));
        % % sigmt3_res = sqrt(res1(2));
        % % sigms_res = abs(sqrt(res1(3)));
        % % eta12_res = res1(4)*0.01/sigmt1_res/sigmat2_initial;
        % % eta13_res = res1(5)/sigmt1_res/sigmt3_res;
        % % eta23_res = res1(6)*0.01/sigmat2_initial/sigmt3_res;
        % % vars = [sigmt1_res,sigmat2_initial,sigmt3_res,sigms_res,eta12_res,eta13_res,eta23_res];
        % % for i=1:1:size(pos_err_test.real_pos,1)
        % %     pos_i = pos_err_test.real_pos(i,:);
        % %     [Gx_eq,Gy_eq,Gz_eq,G_eq] = cal_diff(pos_i*1e-3,st3*1e-3,vars);%时间单位us，距离单位km
        % %     % res_i = Err_XY(i)-(Gx_eq + Gy_eq);
        % %     res_i = Err_XYZ(i)-G_eq;
        % %     res_G2 = [res_G2;res_i];
        % % end
        % % ERR2 = norm(res_G2); %已知模型解析解
        % % time2 = toc;
        % % % recon_tensor = load ('test9_data\7ReconTensor_unknown_model\G_cal.mat').recon_tensor;
        % % % ERR2 = load ('test9_data\7ReconTensor_unknown_model\ERR.mat').ERR %未知模型
    % time_list_i = [time_list_i;time];
    % ERR_list_i = [ERR_list_i;ERR];
    % recon_rate_list_i = [recon_rate_list_i;recon_rate];
end
% time_list = [time_list,time_list_i];
% ERR_list = [ERR_list,ERR_list_i];
% recon_rate_list = [recon_rate_list,recon_rate_list_i];
end
disp('finish')

