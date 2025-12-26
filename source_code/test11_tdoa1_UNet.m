close all
clear all
% 神经网络算法的训练数据是所有蒙特卡洛实验的数据总和，而非神经网络算法的激活数据是单次蒙特卡洛实验的数据，是明显
% 少于神经网络算法的输入的。显然这样的对比是不公平的，因为神经网络算法是有远高于其余算法的输入信息的，否则无法收敛，
% 但为了表现所提算法的优越性，这里仍将神经网络方法用作对比。

% UNet分析的具体步骤for Ratio模式
%1：在本程序中更改需要的NAN_ratio，然后将mode调整为train，点击运行，生产训练集数据
%3：打开python程序，将mode更改为train，将NAN_ratio更改为对应值，点击运行，则开始训练
%4：训练完成后，将mode更改为test，则开始用训练好的模型进行推理，此时推理结果放入outdata文件夹
%5：回到matlab，将mode改为evaluate，则对刚才的推理结果进行误差分析，并存储frob_list_all_Mont
% UNet分析的具体步骤for SNR模式
%1：将mode调整为train，influence_trigger改为SNR，点击运行，仅生产测试集数据
%3：打开python程序，将mode更改为test，将待处理代码被注释掉部分恢复，点击运行，此时推理结果放入outdata文件夹
%5：回到matlab，将mode改为evaluate，则对刚才的推理结果进行误差分析，并存储frob_list_all_Mont


%% 加载原始数据（和test4的数据相同）
fs = 100; %降采样倍数
x_span = -4000:fs:4000;y_span = -4000:fs:4000;z_span = linspace(500,5000,11);
load test11_data/tdoa1/G2.mat 
load test11_data/tdoa1/Sigma2_x.mat
load test11_data/tdoa1/Sigma2_y.mat 
load test11_data/tdoa1/Sigma2_z.mat
G2 = sqrt(G2(1:80,1:80,:));
%% 传感器位置
xt = 0;yt = 0;zt = 0;
x1 = 640;y1 = 1070;z1 = -35;
x2 = -900;y2 = -180;z2 = -27;
x3 = 1000;y3 = -660;z3 = -35;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];
%% 按照比例抽取部分G作为观测量
influence_trigger = 'SNR'; % SNR or Ratio
mode = 'evaluate'; % train or evaluate
% ContrastAlgorithm='UNet';
ContrastAlgorithm='ViT'; % UNet or ViT

out_path0 = 'test11_data/tdoa1/';
CHOSEN_IDX_LIST = [];
Mont_times = 50;
NAN_ratio_0 = 0.8;
err_0 = 0.01;


SNR_list = {'2','6','10','14','18','22'};

if strcmp(influence_trigger,'SNR')
%% SNR对UNet算法的影响%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for err_idx = 1:1:length(SNR_list)
    % err = SNR_list(err_idx);
    snr = str2num(SNR_list{err_idx});%TTTTTTTTTTTTTTTTTT
    sigma_e = G2*10^(-snr/10);
    
    if strcmp(mode,'train')
        %% 1-生成神经网络训练数据集
        % %把张量中已知测量值的向量索引对应到张量行列高索引上
        % disp(['Generating the train data with snr=',num2str(snr)])
        % CHOSEN_IDX_LIST = [];
        % for i=1:1:Mont_times
        %     CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round(NAN_ratio_0*numel(G2)))];
        % end
        % frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
        % save_path = [out_path0,ContrastAlgorithm,'/fs_10_snr_',num2str(snr),'/'];
        % if ~exist([save_path,'data'],'dir')
        %     mkdir([save_path,'data'])
        % end
        % if ~exist([save_path,'label'],'dir')
        %     mkdir([save_path,'label'])
        % end
        % for mont_times = 1:1:Mont_times  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次start
        %     noise = sigma_e.*randn(size(G2));
        %     incomplete_T = G2+noise;%0.01km标准差的噪声
        %     % incomplete_T = G2+err*randn(size(G2));%0.01km标准差的噪声
        %     ALL_ELE = 1:numel(incomplete_T);
        %     % CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
        %     % load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
        %     CHOSEN_IDX = CHOSEN_IDX_LIST(mont_times,:);
        %     unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
        %     measurements = zeros(size(incomplete_T));
        %     measurements(unselected_elements) = incomplete_T(unselected_elements);
        %     for i=1:1:size(G2,3)
        %         data = measurements(:,:,i);
        %         label = G2(:,:,i);
        %         % data_save_path = [save_path,'data'];
        %         save([[save_path,'data/'],['data_',num2str(mont_times),'_',num2str(i),'.mat']],'data')
        %         save([[save_path,'label/'],['label_',num2str(mont_times),'_',num2str(i),'.mat']],'label')
        %     end
        % end
        %% 2-生成神经网络测试集
        disp(['Generating the test data with snr=',num2str(snr)])
        CHOSEN_IDX_LIST = [];
        for i=1:1:Mont_times
            CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round(NAN_ratio_0*numel(G2)))];
        end

        frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
        save_path = [out_path0,ContrastAlgorithm,'/fs_10_snr_',num2str(snr),'/'];
        if ~exist([save_path,'testdata'],'dir')
            mkdir([save_path,'testdata'])
        end
        if ~exist([save_path,'testlabel'],'dir')
            mkdir([save_path,'testlabel'])
        end
        for mont_times = 1:1:Mont_times  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次start
            noise = sigma_e.*randn(size(G2));
            incomplete_T = G2+noise;%0.01km标准差的噪声
            % incomplete_T = G2+err*randn(size(G2));%0.01km标准差的噪声
            ALL_ELE = 1:numel(incomplete_T);
            % CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
            % load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
            CHOSEN_IDX = CHOSEN_IDX_LIST(mont_times,:);
            unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
            measurements = zeros(size(incomplete_T));
            measurements(unselected_elements) = incomplete_T(unselected_elements);
            for i=1:1:size(G2,3)
                data = measurements(:,:,i);
                label = G2(:,:,i);
                % data_save_path = [save_path,'testdata'];
                if ~exist([save_path,'testdata/z_',num2str(i)],'dir')
                    mkdir([save_path,'testdata/z_',num2str(i)])
                end
                if ~exist([save_path,'testlabel/z_',num2str(i)],'dir')
                    mkdir([save_path,'testlabel/z_',num2str(i)])
                end
                save([[save_path,'testdata/z_',num2str(i),'/'],['data_',num2str(mont_times),'_',num2str(i),'.mat']],'data')
                save([[save_path,'testlabel/z_',num2str(i),'/'],['label_',num2str(mont_times),'_',num2str(i),'.mat']],'label')
            end
        end
    end
    %% 测试训练好的网络性能
    if strcmp(mode,'evaluate')
        disp(['Evaluating the test data with saved model and snr=',num2str(snr)])
        frob_list_all_Mont = [];
        for i=1:1:size(G2,3)
            read_path = [out_path0,ContrastAlgorithm,'/fs_10_snr_',num2str(snr),'/','outdata/z_',num2str(i)];
            dirs_list = dir(read_path);
            frob_list = [];
            for j=3:1:length(dirs_list)
                dirs_i = dirs_list(j).name;
                load ([read_path,'/',dirs_i]);
                frob_list = [frob_list;frob(res_matrix-G2(:,:,i))];
            end
            frob_list_all_Mont = [frob_list_all_Mont,frob_list];
        end
        save_path = [out_path0,ContrastAlgorithm,'/fs_10_snr_',num2str(snr),'/'];
        save([save_path,'frob_list_all_Mont.mat'],'frob_list_all_Mont')

        % figure();
        % [c,handle]=contour(x_span,y_span,G2(:,:,1),20);
        % clabel(c,handle);
        % title('Real distribution of $G$','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % 
        % figure();
        % [c,handle]=contour(x_span,y_span,res_matrix,20);
        % clabel(c,handle);
        % title('Real distribution of $G$','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
    end
end
end


NAN_ratio_list = [0.8,0.85,0.9,0.93,0.95,0.97,0.99];

if strcmp(influence_trigger,'Ratio')
%% NaNRatio对UNet算法的影响%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ratio_idx = 1:1:length(NAN_ratio_list)
    NAN_ratio = NAN_ratio_list(ratio_idx);
    if strcmp(mode,'train')
        %% 1-生成神经网络训练数据集
        %把张量中已知测量值的向量索引对应到张量行列高索引上
        disp(['Generating the train data with ratio=',num2str(NAN_ratio)])
        CHOSEN_IDX_LIST = [];
        for i=1:1:Mont_times
            CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round(NAN_ratio*numel(G2)))];
        end
        frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
        save_path = [out_path0,ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/'];
        if ~exist([save_path,'data'],'dir')
            mkdir([save_path,'data'])
        end
        if ~exist([save_path,'label'],'dir')
            mkdir([save_path,'label'])
        end
        for mont_times = 1:1:Mont_times  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次start
            incomplete_T = G2+err_0*randn(size(G2));%0.01km标准差的噪声
            ALL_ELE = 1:numel(incomplete_T);
            % CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
            % load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
            CHOSEN_IDX = CHOSEN_IDX_LIST(mont_times,:);
            unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
            measurements = zeros(size(incomplete_T));
            measurements(unselected_elements) = incomplete_T(unselected_elements);
            for i=1:1:size(G2,3)
                data = measurements(:,:,i);
                label = G2(:,:,i);
                % data_save_path = [save_path,'data'];
                save([[save_path,'data/'],['data_',num2str(mont_times),'_',num2str(i),'.mat']],'data')
                save([[save_path,'label/'],['label_',num2str(mont_times),'_',num2str(i),'.mat']],'label')
            end
        end
        %% 2-生成神经网络测试集
        disp(['Generating the test data with ratio=',num2str(NAN_ratio)])
        CHOSEN_IDX_LIST = [];
        for i=1:1:Mont_times
            CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round(NAN_ratio*numel(G2)))];
        end

        frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
        save_path = [out_path0,ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/'];
        if ~exist([save_path,'testdata'],'dir')
            mkdir([save_path,'testdata'])
        end
        if ~exist([save_path,'testlabel'],'dir')
            mkdir([save_path,'testlabel'])
        end
        for mont_times = 1:1:Mont_times  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次start
            incomplete_T = G2+err_0*randn(size(G2));%0.01km标准差的噪声
            ALL_ELE = 1:numel(incomplete_T);
            % CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
            % load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
            CHOSEN_IDX = CHOSEN_IDX_LIST(mont_times,:);
            unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
            measurements = zeros(size(incomplete_T));
            measurements(unselected_elements) = incomplete_T(unselected_elements);
            for i=1:1:size(G2,3)
                data = measurements(:,:,i);
                label = G2(:,:,i);
                % data_save_path = [save_path,'testdata'];
                if ~exist([save_path,'testdata/z_',num2str(i)],'dir')
                    mkdir([save_path,'testdata/z_',num2str(i)])
                end
                if ~exist([save_path,'testlabel/z_',num2str(i)],'dir')
                    mkdir([save_path,'testlabel/z_',num2str(i)])
                end
                save([[save_path,'testdata/z_',num2str(i),'/'],['data_',num2str(mont_times),'_',num2str(i),'.mat']],'data')
                save([[save_path,'testlabel/z_',num2str(i),'/'],['label_',num2str(mont_times),'_',num2str(i),'.mat']],'label')
            end
        end
    end
    %% 测试训练好的网络性能
    if strcmp(mode,'evaluate')
        disp(['Evaluating the test data with saved model and ratio=',num2str(NAN_ratio)])
        frob_list_all_Mont = [];
        for i=1:1:size(G2,3)
            read_path = [out_path0,ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/','outdata/z_',num2str(i)];
            dirs_list = dir(read_path);
            frob_list = [];
            for j=3:1:length(dirs_list)
                dirs_i = dirs_list(j).name;
                load ([read_path,'/',dirs_i]);
                frob_list = [frob_list;frob(res_matrix-G2(:,:,i))];
            end
            frob_list_all_Mont = [frob_list_all_Mont,frob_list];
        end
        save_path = [out_path0,ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/'];
        save([save_path,'frob_list_all_Mont.mat'],'frob_list_all_Mont')

        % figure();
        % [c,handle]=contour(x_span,y_span,G2(:,:,1),20);
        % clabel(c,handle);
        % title('Real distribution of $G$','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
        % 
        % figure();
        % [c,handle]=contour(x_span,y_span,res_matrix,20);
        % clabel(c,handle);
        % title('Real distribution of $G$','interpreter','latex');
        % xlabel('x(km)');
        % ylabel('y(km)');
    end
end
end
