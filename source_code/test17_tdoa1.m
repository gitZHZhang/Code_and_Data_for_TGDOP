close all
clear all
%%%%%%%%%%%%%%%
% 同样是tdoa定位场景，和test8相比有不一样的误差参数和基站构型
%%%%%%%%%%%%%%%
% helperMIMOBER(4,randn(10),10)
%% 加载原始数据（和test4的数据相同）


%% 传感器位置

xt = 0;yt = 0;zt = 0;
x1 = 640;y1 = 1070;z1 = -35;
x2 = -900;y2 = -180;z2 = -27;
x3 = 1000;y3 = -660;z3 = -35;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];

fs = 10; %降采样倍数
x_span = -4000:fs:4000;y_span = -4000:fs:4000;z_span = linspace(500,5000,11);
vars_real = [18e-3,20e-3,25e-3,0.5,-0.3,0.5,-0.2]; %us us us m
% for k =1:1:length(z_span)
%     for i = 1:1:length(x_span)
%         for j = 1:1:length(y_span)
%             m=x_span(i)+0.01;
%             n=y_span(j)+0.01;
%             p=z_span(k)+0.001;%height=10m
%             emitter = [m,n,p];
%             [Gx_eq,Gy_eq,Gz_eq,G_eq] = cal_diff(emitter,sensor,vars_real);%时间单位us，距离单位m
%             G2(j,i,k) = G_eq;
%             Sigma2_x(j,i,k) = Gx_eq;
%             Sigma2_y(j,i,k) = Gy_eq;
%             Sigma2_z(j,i,k) = Gz_eq;
%         end
%     end
% end





Mont_times = 5;
down_samples_list = [2,4,6,8,10,12,14,16,18,20];
NAN_ratio_list = {'0.6','0.7','0.8','0.85','0.9','0.93','0.95','0.97','0.99'};
NAN_ratio_list = {'0.6','0.7'};
res_cell1 = cell(length(down_samples_list),length(NAN_ratio_list));
res_cell2 = cell(length(down_samples_list),length(NAN_ratio_list));

for ds_idx = 1:1:length(down_samples_list)
    load test17_data/tdoa1/G2.mat
    load test17_data/tdoa1/Sigma2_x.mat
    load test17_data/tdoa1/Sigma2_y.mat
    load test17_data/tdoa1/Sigma2_z.mat
    down_samples_i = down_samples_list(ds_idx);
    G2 = G2(1:down_samples_i:end,1:down_samples_i:end,:);
    Sigma2_x = Sigma2_x(1:down_samples_i:end,1:down_samples_i:end,:);
    Sigma2_y = Sigma2_y(1:down_samples_i:end,1:down_samples_i:end,:);
    Sigma2_z = Sigma2_z(1:down_samples_i:end,1:down_samples_i:end,:);

    
    % NAN_ratio = 0.99; %80 85 90 93 95 97 99
    for index_list = 1:1:length(NAN_ratio_list)%TTTTTTTTTTTTTTTTTT
        CHOSEN_IDX_LIST = [];
        ERR = 0.01;
        NAN_ratio = str2num(NAN_ratio_list{index_list});%TTTTTTTTTTTTTTTTTT
        % NAN_ratio = 0.8; %80 85 90 93 95 97 99
        for i=1:1:Mont_times
            CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round((NAN_ratio)*numel(G2)))];%TTTTTTTTTTTTTTTTTT
        end

        %% 和别的算法作比较:Proposed/FNN/kriging/SVT/NNM-T
        Algorithm_list = {'Proposed','GeneralBTD'};
        Algorithm_list = {'Proposed'};
        for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
            ContrastAlgorithm=Algorithm_list{algorithm_idx};
            frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
            save_path = ['test11_data/tdoa1/',ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/'];
            % save_path = ['test11_data/tdoa1/',ContrastAlgorithm,'/fs_10_err_',ERR_list{index_list},'/'];%TTTTTTTTTTTTTTTTTT
            min_frob_list = 1e8;%当froblist均值最小时存储相应的重构张量
            for mont_times = 1:1:Mont_times  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次start
                disp_str = [ContrastAlgorithm,':',num2str(mont_times),'%%%%%%%%%%%%%%%%%%'];
                disp(disp_str)
                
                %% 按照比例抽取部分G作为观测量
                incomplete_T = G2+ERR*randn(size(G2));%0.01km标准差的噪声
                ALL_ELE = 1:numel(incomplete_T);
                % CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
                % load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
                CHOSEN_IDX = CHOSEN_IDX_LIST(mont_times,:);
                unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
                %画出观测数据在全部区域的分布
                a = zeros(1,numel(incomplete_T));
                a(unselected_elements)=1;
                a_3d = reshape(a,size(G2,1),size(G2,2),size(G2,3));
                % plot_title = 'The 3-D distribution of the selected grid.';
                % [~] = plot_3D_tensor(a_3d,x_span,y_span,z_span,plot_title);
                incomplete_T(CHOSEN_IDX) = NaN; %将G2中的值设置为未知

                if strcmp(ContrastAlgorithm,'Proposed')
                    incomplete_T2 = fmt(incomplete_T);
                    size_tens = incomplete_T2.size;
                    % size_tens = size(incomplete_T);
                    L1 = [5 4 3];
                    L2 = [6 5 3];
                    L3 = [6 5 3];
                    L1 = [5 4 4];
                    L2 = [4 5 5];
                    L3 = [5 5 4];
                    % L1 = [15 2 2];
                    % L2 = [2 15 2];
                    % L3 = [2 2 15];
                    % L1 = [6 4 3];
                    % L2 = [6 6 3];
                    % L3 = [6 5 3];
                    % L1 = [10 10 3];
                    % L2 = [12 12 3];
                    % L3 = [15 15 5];
                    model= struct;
                    if(NAN_ratio<0)
                    % if(NAN_ratio<0.8)
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

                        m=8;%%m=2:9  4.5913e+03 2.1511e+03 2.0862e+03 1.1906e+03 861.0972 826.5696 761.6267 1.2590e+03
                        % m=4;%m=6:579 m=5:321 m=4:576
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
                        % model.factors={ {'A1',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
                        %     {'B1',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
                        %     {'C1',  @(z,task) struct_poly(z,task,t2),@struct_nonneg},...
                        %     {'S1',@struct_nonneg},...
                        %     {'A2',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
                        %     {'B2',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
                        %     {'C2',  @(z,task) struct_poly(z,task,t2),@struct_nonneg},...
                        %     {'S2',@struct_nonneg},...
                        %     {'A3',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
                        %     {'B3',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
                        %     {'C3',  @(z,task) struct_poly(z,task,t2),@struct_nonneg},...
                        %     {'S3',@struct_nonneg} };
                        model.factors={ {'A1',  @(z,task) struct_poly(z,task,t1)},...
                            {'B1',  @(z,task) struct_poly(z,task,t1)},...
                            {'C1',  @(z,task) struct_poly(z,task,t2)},...
                            {'S1'},...
                            {'A2',  @(z,task) struct_poly(z,task,t1),},...
                            {'B2',  @(z,task) struct_poly(z,task,t1)},...
                            {'C2',  @(z,task) struct_poly(z,task,t2)},...
                            {'S2'},...
                            {'A3',  @(z,task) struct_poly(z,task,t1)},...
                            {'B3',  @(z,task) struct_poly(z,task,t1)},...
                            {'C3',  @(z,task) struct_poly(z,task,t2)},...
                            {'S3'} };
                    end

                    %% 以先验信息为初值的因子矩阵
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
                    % % subplot(2,1,1)
                    % [c,handle]=contour(x_span,y_span,G2(:,:,1),20);
                    % clabel(c,handle);
                    % title('Real distribution of $G$','interpreter','latex');
                    % xlabel('x(km)');
                    % ylabel('y(km)');
                    % % leg = title('$\sigma_{y}^2的真实分布（降采样后）$','interpreter','latex');
                    % % set(leg,'Interpreter','latex')
                    % subplot(2,1,2)
                    % [c,handle]=contour(x_span,y_span,recon_tensor(:,:,1),20);
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
                end
                frob_list_all_Mont = [frob_list_all_Mont;frob_list];
                if strcmp(ContrastAlgorithm,'Proposed')
                    res_cell1{ds_idx,index_list} = frob_list_all_Mont;
                else
                    res_cell2{ds_idx,index_list} = frob_list_all_Mont;
                end
            end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次end
            % save([save_path,'frob_list_all_Mont.mat'],'frob_list_all_Mont')
        end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法end
    end
end



