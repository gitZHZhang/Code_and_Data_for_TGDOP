close all
clear all
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

Mont_times = 50;
NAN_ratio = 0.99;
analysis = 3;
Algorithm_list = {'kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet','GeneralGDOP'};
line_styles = {'-','--','-.',':'};
markers = {'o','+','*','s','^','x','p','d'};
color_order = [0    0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];
if analysis==1
%% 总数据分析1：99%缺失值算法箱线图对比
Algorithm_list = {'kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet'};
frob_list_all_mean = [];
frob_list_all_var = [];
frob_list_algorithms = cell(1,length(Algorithm_list));
for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
    ContrastAlgorithm=Algorithm_list{algorithm_idx};
    frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
    load_path = ['test8_data/',ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/'];
    load_file = [load_path,'frob_list_all_Mont.mat'];
    frob_list_all_Mont_i = load(load_file);
    frob_list_all_Mont_i = frob_list_all_Mont_i.frob_list_all_Mont;
    frob_list_algorithms{algorithm_idx} = frob_list_all_Mont_i;
    frob_list_all_mean = [frob_list_all_mean;mean(frob_list_all_Mont_i)];
    frob_list_all_var = [frob_list_all_var;var(frob_list_all_Mont_i)];
end

% 需要分组的数据，size 11*4
color_all=slanCL(10);
filledcolor = [];
fillbox = 0;
figure
% 箱线图，仅边框有颜色
for i = 1:1:length(frob_list_algorithms)
% for i = 1:1:2
    group_i = frob_list_algorithms{i};
    % edgecolor_i = [0,0,0];
    filledcolor_i = color_all(i,:);
    filledcolor = [filledcolor;repmat(filledcolor_i, size(group_i,2), 1)];
    % box_i = boxplot(group_i,'Colors',edgecolor_i,'Widths',0.4);
    box_i = boxplot(group_i,'Colors',filledcolor_i,'Widths',0.4,'Symbol','o','OutlierSize',0.5);
    h = findobj(gca, 'Tag', 'Median');
    median_v = get(h,'YData');
    median_v = median_v(1:size(group_i,2));
    median_v = flip([median_v{:}]);
    set(box_i,'LineWidth',1.5);
    hold on;
    plot(1:1:size(group_i,2), median_v(1:2:end), '-*', 'Color', filledcolor_i)
    % median_v(end)
end
xticklabels(10:1:20)
if(fillbox)%对box进行颜色填充
    boxobj = findobj(gca, 'Tag', 'Box');
    for i = 1:length(boxobj)
        patch(get(boxobj(i), 'XData'), get(boxobj(i), 'YData'), filledcolor(length(boxobj)-i+1,:), 'FaceAlpha', 0.5)
    end
end
legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet')
% legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet','GeneralGDOP')
xlabel('Height (km)')
ylabel('Reconstruction error.')
title('Boxplot of reconstruction errors across various algorithms.')

elseif analysis==2
%% 总数据分析2：99%缺失值算法还原度可视化
contour_num = 15; %等值线数目
% contour_values = linspace(3,200,contour_num); %指定数值的等值线列表，列表长度即等值线数目
contour_values = contour_num;
bias = 20; %用于确定参考源的位置，在等值线图中选取4个点来展示值
text_bias = 5; %文本位置偏置，可以保证数字不和图标重合

figure
[c,handle]=contour(x_span,y_span,G2(:,:,11),contour_values);
clabel(c,handle);
title('Real distribution of $G(:,:,end)$.','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');
hold on
add_points(G2(:,:,11),bias,text_bias)

for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
    ContrastAlgorithm=Algorithm_list{algorithm_idx};
    if strcmp(ContrastAlgorithm,'UNet')
        read_path = ['test8_data/UNet/fs_10_ratio_',num2str(NAN_ratio),'/','outdata/z_11/'];
        dir_list = dir(read_path);
        recon_tensor_i = load([read_path,dir_list(3).name]);
        figure
        % contour_values = linspace(3,200,contour_num);
        [c,handle]=contour(x_span,y_span,recon_tensor_i.res_matrix,contour_values);
        clabel(c,handle);
        title(ContrastAlgorithm,'interpreter','latex');
        xlabel('x(km)');
        ylabel('y(km)');
        hold on
        add_points(recon_tensor_i.res_matrix,bias,text_bias)
    else
        load_path = ['test8_data/',ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/'];
        load_file = [load_path,'recon_tensor.mat'];
        recon_tensor_i = load(load_file);
        recon_tensor_i = recon_tensor_i.recon_tensor;
        figure
        % contour_values = linspace(3,200,contour_num);
        [c,handle]=contour(x_span,y_span,recon_tensor_i(:,:,11),contour_values);
        if frob(G2(:,:,11)-recon_tensor_i(:,:,11))<2e3
            clabel(c,handle);
        end
        title(ContrastAlgorithm,'interpreter','latex');
        xlabel('x(km)');
        ylabel('y(km)');
        hold on
        add_points(recon_tensor_i(:,:,11),bias,text_bias)
    end
    

end
elseif analysis==3
%% 总数据分析3：不同缺失值算法对比
Algorithm_list = {'kriging','RBF','NNM-T','Proposed','GeneralBTD','GeneralGDOP','UNet'};
ratio_list = [0.8,0.85,0.9,0.93,0.95,0.97,0.99];
% Algorithm_list = {'Proposed'};
frob_algorithms_ratios_all_mean = [];
for ratio =1:1:length(ratio_list)
    ratio_i = ratio_list(ratio);
    frob_algorithms_all_mean = [];
    for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
        ContrastAlgorithm=Algorithm_list{algorithm_idx};
        frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
        load_path = ['test8_data/',ContrastAlgorithm,'/fs_10_ratio_',num2str(ratio_i),'/'];
        load_file = [load_path,'frob_list_all_Mont.mat'];
        frob_list_all_Mont_i = load(load_file);
        frob_list_all_Mont_i = frob_list_all_Mont_i.frob_list_all_Mont;
        frob_algorithms_all_mean = [frob_algorithms_all_mean,mean(mean(frob_list_all_Mont_i))];       
    end
    frob_algorithms_ratios_all_mean = [frob_algorithms_ratios_all_mean;frob_algorithms_all_mean];
end
figure
for algorithm_idx = 1:1:length(Algorithm_list)
    plot(ratio_list*100,frob_algorithms_ratios_all_mean(:,algorithm_idx),'*-');
    hold on
end
legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet','GeneralGDOP')
grid on
xlabel('Percentage of missing observations (%).')
ylabel('Reconstruction error.')

elseif analysis==4
%% 总数据分析4：不同噪声标准差算法对比
err_list = [0.01,0.5,1,1.5,2,2.5,3,5,8,12,16];
% Algorithm_list = {'UNet'};
Algorithm_list = {'kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet'};
frob_algorithms_ratios_all_mean = [];
for err =1:1:length(err_list)
    err_i = err_list(err);
    frob_algorithms_all_mean = [];
    for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
        ContrastAlgorithm=Algorithm_list{algorithm_idx};
        frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
        load_path = ['test8_data/',ContrastAlgorithm,'/fs_10_err_',num2str(err_i),'/'];
        load_file = [load_path,'frob_list_all_Mont.mat'];
        frob_list_all_Mont_i = load(load_file);
        frob_list_all_Mont_i = frob_list_all_Mont_i.frob_list_all_Mont;
        frob_algorithms_all_mean = [frob_algorithms_all_mean,mean(mean(frob_list_all_Mont_i))];       
    end
    frob_algorithms_ratios_all_mean = [frob_algorithms_ratios_all_mean;frob_algorithms_all_mean];
end
figure
for algorithm_idx = 1:1:length(Algorithm_list)
    plot(err_list,frob_algorithms_ratios_all_mean(:,algorithm_idx),'*-');
    hold on
end
legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet')
grid on
xlabel('Standard deviation of observation error \underline{\textbf{N}}.','interpreter','latex')
ylabel('Reconstruction error.')

frob_diff = abs(diff(double(frob_algorithms_ratios_all_mean),1,1));
err_diff = diff(err_list,1,2)';
err_diff = repmat(err_diff,1,size(frob_diff,2));
Slope = mean(frob_diff./err_diff,1);%判断鲁棒性的参数
end

