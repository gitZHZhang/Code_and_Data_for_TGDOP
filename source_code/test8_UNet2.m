close all
clear all
% 神经网络算法的训练数据是所有蒙特卡洛实验的数据总和，而非神经网络算法的激活数据是单次蒙特卡洛实验的数据，是明显
% 少于神经网络算法的输入的。显然这样的对比是不公平的，因为神经网络算法是有远高于其余算法的输入信息的，否则无法收敛，
% 但为了表现所提算法的优越性，这里仍将神经网络方法用作对比。

% UNet分析的具体步骤
%1：在本程序中更改需要的NAN_ratio，然后将mode调整为train，点击运行，生产训练集数据
%2：将mode调整为test，点击运行，生产测试集数据
%3：打开python程序，将mode更改为train，将NAN_ratio更改为对应值，点击运行，则开始训练
%4：训练完成后，将mode更改为test，则开始用训练好的模型进行推理，此时推理结果放入outdata文件夹
%5：回到matlab，将mode改为evaluate，则对刚才的推理结果进行误差分析，并存储frob_list_all_Mont


%% 加载原始数据（和test4的数据相同）
load G.mat 
load sigma2_x.mat
load sigma2_y.mat 
load sigma2_z.mat

fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;z_span = linspace(10,20,11);
for i=1:1:size(G,1)/fs
    for j=1:1:size(G,2)/fs
        for k=1:1:size(G,3)
            G2(j,i,k) = G(fs*i,fs*j,k);
            Sigma2_x(j,i,k) = sigma2_x(fs*i,fs*j,k);
            Sigma2_y(j,i,k) = sigma2_y(fs*i,fs*j,k);
            Sigma2_z(j,i,k) = sigma2_z(fs*i,fs*j,k);
        end
    end
end
% figure
% mlrankest(Sigma2_y)
% size_core = mlrankest(Sigma2_x);%计算sizecore
% size_core = [3,3,2];
% [UXhat,SXhat,output]=lmlra(Sigma2_x,size_core); 
% [UYhat,SYhat,output]=lmlra(Sigma2_y,size_core); 
% size_core = [4,4,2];
% [UZhat,SZhat,output]=lmlra(Sigma2_z,size_core); 


%% 传感器位置
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
x3 = 0;y3 = -L;z3 = 0.3;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];
%% 按照比例抽取部分G作为观测量
CHOSEN_IDX_LIST = [];
Mont_times = 50;
NAN_ratio = 0.99;

err_list = [16];
for err_idx = 1:1:length(err_list)
    err = err_list(err_idx);
    mode = 'evaluate'; %train or test or evaluate
    for i=1:1:Mont_times
        CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round(NAN_ratio*numel(G2)))];
    end
    
    
    %% 1-生成神经网络训练数据集
    %把张量中已知测量值的向量索引对应到张量行列高索引上
    
    Algorithm_list = {'UNet'};
    if strcmp(mode,'train')
    disp(['Generating the train data with err=',num2str(err)])
    for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
        ContrastAlgorithm=Algorithm_list{algorithm_idx};
        frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差  
        save_path = ['test8_data/',ContrastAlgorithm,'/fs_10_err_',num2str(err),'/'];
        if ~exist([save_path,'data'],'dir')
            mkdir([save_path,'data'])
        end
        if ~exist([save_path,'label'],'dir')
            mkdir([save_path,'label'])
        end
        for mont_times = 1:1:Mont_times  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次start
            incomplete_T = G2+err*randn(size(G2));%0.01km标准差的噪声
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
    end
    end
    %% 2-生成神经网络测试集
    if strcmp(mode,'test')
    disp(['Generating the test data with err=',num2str(err)])
    CHOSEN_IDX_LIST = [];
    for i=1:1:Mont_times
        CHOSEN_IDX_LIST = [CHOSEN_IDX_LIST;randperm(numel(G2),round(NAN_ratio*numel(G2)))];
    end
    
    for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
        ContrastAlgorithm=Algorithm_list{algorithm_idx};
        frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差  
        save_path = ['test8_data/',ContrastAlgorithm,'/fs_10_err_',num2str(err),'/'];
        if ~exist([save_path,'testdata'],'dir')
            mkdir([save_path,'testdata'])
        end
        if ~exist([save_path,'testlabel'],'dir')
            mkdir([save_path,'testlabel'])
        end
        for mont_times = 1:1:Mont_times  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%每种算法循环若干次start
            incomplete_T = G2+0.01*randn(size(G2));%0.01km标准差的噪声
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
    end
    %% 测试训练好的网络性能
    if strcmp(mode,'evaluate')
    disp(['Evaluating the test data with saved model and err=',num2str(err)])
    frob_list_all_Mont = [];
    ContrastAlgorithm=Algorithm_list{1};
    for i=1:1:size(G2,3)
        read_path = ['test8_data/',ContrastAlgorithm,'/fs_10_err_',num2str(err),'/','outdata/z_',num2str(i)];
        dirs_list = dir(read_path);
        frob_list = [];
        for j=3:1:length(dirs_list)
            dirs_i = dirs_list(j).name;
            load ([read_path,'/',dirs_i]);
            frob_list = [frob_list;frob(res_matrix-G2(:,:,i))];
        end
        frob_list_all_Mont = [frob_list_all_Mont,frob_list];
    end
    save_path = ['test8_data/UNet/fs_10_err_',num2str(err),'/'];
    save([save_path,'frob_list_all_Mont.mat'],'frob_list_all_Mont')
    
    figure();
    [c,handle]=contour(x_span,y_span,G2(:,:,1),20);
    clabel(c,handle);
    title('Real distribution of $G$','interpreter','latex');
    xlabel('x(km)');
    ylabel('y(km)');
    
    figure();
    [c,handle]=contour(x_span,y_span,res_matrix,20);
    clabel(c,handle);
    title('Real distribution of $G$','interpreter','latex');
    xlabel('x(km)');
    ylabel('y(km)');
    end
end
