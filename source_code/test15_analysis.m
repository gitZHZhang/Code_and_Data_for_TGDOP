close all
clear all


fs = 100; %降采样倍数\
x_span = -4000:fs:4000;y_span = -4000:fs:4000;z_span = linspace(500,5000,11);
scene_list = {'tdoa1'};
% scene_list = {'doa','tdoa1'};

Mont_times = 50;
NAN_ratio = 0.99;
analysis = 1;
line_styles = {'-','--','-.',':'};
markers = {'o','+','*','s','^','x','p','d'};
% Algorithm_list = {'Kriging','RBF','NNM-T','Proposed','GeneralBTD','GeneralGDOP','UNet'};
Algorithm_list = {'Kriging','RBF','NNM-T','Proposed','GeneralBTD','GeneralGDOP2','UNet','ViT'};
% Algorithm_list = {'Kriging','RBF','Proposed','GeneralBTD','GeneralGDOP2'};
% Algorithm_list = {'Kriging','RBF','Proposed','GeneralBTD','GeneralGDOP','GeneralGDOP2'};
% Algorithm_list = {'Kriging','RBF','NNM-T','GeneralBTD','GeneralGDOP2','Proposed'};
nan99_reconERR_values=[];
nan99_perfoemance_improved=[];
nan99_Slope_values=[];
nan99_Slope_improved=[];
print_matrix = [];
color_order = [0    0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840;
    0 0.5 0];
% Algorithm_list = {'kriging','RBF','NNM-T','GeneralBTD','GeneralGDOP'};
for scene_idx = 1:1:length(scene_list)
    scene_i = scene_list{scene_idx};
    load_path0 = ['test15_data/',scene_i,'/'];
    if analysis==1
        
        %% 总数据分析1：99%缺失值算法箱线图对比
        % Algorithm_list = {'Kriging','RBF','NNM-T','Proposed','GeneralBTD'};
        frob_list_all_mean = [];
        frob_list_all_var = [];
        frob_list_algorithms = cell(1,length(Algorithm_list));
        for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
            ContrastAlgorithm=Algorithm_list{algorithm_idx};
            frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
            load_path = [load_path0,ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/'];
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
            % box_i = boxplot(group_i,'Colors',filledcolor_i,'Widths',0.4,'Symbol','o','OutlierSize',0.5);
            box_i = boxplot(group_i,'Colors',color_order(i,:),'Widths',0.4,'Symbol','o','OutlierSize',0.5);
            h = findobj(gca, 'Tag', 'Median');
            median_v = get(h,'YData');
            median_v = median_v(1:size(group_i,2));
            median_v = flip([median_v{:}]);
            set(box_i,'LineWidth',1.5);
            hold on;
            % plot(1:1:size(group_i,2), median_v(1:2:end), '-*', 'Color', filledcolor_i)
            plot(1:1:size(group_i,2), median_v(1:2:end), 'Marker',markers{i},'LineStyle',line_styles{4-mod(i,4)},'Color',color_order(i,:))
            hold on;
            % color_order = get(gca,'colororder');
            % median_v(end)
        end
        xticklabels(round(z_span./1e3,2))
        if(fillbox)%对box进行颜色填充
            boxobj = findobj(gca, 'Tag', 'Box');
            for i = 1:length(boxobj)
                patch(get(boxobj(i), 'XData'), get(boxobj(i), 'YData'), filledcolor(length(boxobj)-i+1,:), 'FaceAlpha', 0.5)
            end
        end

        legend_label = Algorithm_list;
        if sum(strcmp(Algorithm_list, 'GeneralGDOP'))~=0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP')}='AccurateGDOP';
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='WGDOP';
        elseif sum(strcmp(Algorithm_list, 'GeneralGDOP'))==0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='WGDOP';
        end
        legend(legend_label)
        
        % legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet','GeneralGDOP')
        xlabel('Height (km)')
        ylabel('MFNE (m)')
        title('Boxplot of reconstruction errors across various algorithms.')

    elseif analysis==2
        %% 总数据分析2：不同缺失值算法对比
        ratio_list = [0.8,0.85,0.9,0.93,0.95,0.97,0.99];
        % ratio_list = [0.8,0.85,0.9,0.93,0.95,0.97];
        % Algorithm_list = {'UNet'};
        frob_algorithms_ratios_all_mean = [];
        for ratio =1:1:length(ratio_list)
            ratio_i = ratio_list(ratio);
            frob_algorithms_all_mean = [];
            for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
                ContrastAlgorithm=Algorithm_list{algorithm_idx};
                frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
                load_path = [load_path0,ContrastAlgorithm,'/fs_10_ratio_',num2str(ratio_i),'/'];
                load_file = [load_path,'frob_list_all_Mont.mat'];
                frob_list_all_Mont_i = load(load_file);
                frob_list_all_Mont_i = frob_list_all_Mont_i.frob_list_all_Mont;
                frob_algorithms_all_mean = [frob_algorithms_all_mean,mean(mean(frob_list_all_Mont_i))];
            end
            frob_algorithms_ratios_all_mean = [frob_algorithms_ratios_all_mean;frob_algorithms_all_mean];
        end
        
        figure
        for algorithm_idx = 1:1:length(Algorithm_list)
            % plot(ratio_list*100,frob_algorithms_ratios_all_mean(:,algorithm_idx),'*-');
            plot(ratio_list*100,frob_algorithms_ratios_all_mean(:,algorithm_idx),'Marker',markers{algorithm_idx},'LineStyle',line_styles{4-mod(algorithm_idx,4)},'Color',color_order(algorithm_idx,:));
            hold on
        end
        nan99_reconERR_values = [nan99_reconERR_values;round(frob_algorithms_ratios_all_mean(end,:),2)];
        proposed_idx = find(strcmp(Algorithm_list, 'Proposed'));
        nan99_perfoemance_improved = [nan99_perfoemance_improved;round((frob_algorithms_ratios_all_mean(end,:)-frob_algorithms_ratios_all_mean(end,proposed_idx))./frob_algorithms_ratios_all_mean(end,:)*100,2)];
        MFNE_nan99 = round(frob_algorithms_ratios_all_mean(end,:),2);
        print_matrix = [print_matrix,frob_algorithms_ratios_all_mean(:,proposed_idx)];
        % legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','GeneralGDOP','UNet')
        legend_label = Algorithm_list;
        if sum(strcmp(Algorithm_list, 'GeneralGDOP'))~=0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP')}='AccurateGDOP';
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='WGDOP';
        elseif sum(strcmp(Algorithm_list, 'GeneralGDOP'))==0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='WGDOP';
        end
        % legend(legend_label)
        legend(legend_label,'Location','northwest')
        % legend('Kriging','RBF','NNM-T','GeneralBTD','GeneralGDOP')
        grid on
        xlabel('MER (%)')
        ylabel('MFNE (m)')
    elseif analysis==3
        %% 总数据分析3：不同噪声标准差算法对比
        err_list = [0.01,1,2,3,5,8,12,16];
        % err_list = [0.01,1,2,3,5,8,12];
        % Algorithm_list = {'UNet'};
        % Algorithm_list = {'kriging','RBF','NNM-T','Proposed','GeneralBTD'};
        frob_algorithms_ratios_all_mean = [];
        for err =1:1:length(err_list)
            err_i = err_list(err);
            frob_algorithms_all_mean = [];
            for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
                ContrastAlgorithm=Algorithm_list{algorithm_idx};
                frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
                load_path = [load_path0,ContrastAlgorithm,'/fs_10_err_',num2str(err_i),'/'];
                load_file = [load_path,'frob_list_all_Mont.mat'];
                frob_list_all_Mont_i = load(load_file);
                frob_list_all_Mont_i = frob_list_all_Mont_i.frob_list_all_Mont;
                frob_algorithms_all_mean = [frob_algorithms_all_mean,mean(mean(frob_list_all_Mont_i))];
            end
            frob_algorithms_ratios_all_mean = [frob_algorithms_ratios_all_mean;frob_algorithms_all_mean];
        end
        figure
        for algorithm_idx = 1:1:length(Algorithm_list)
            plot(err_list,frob_algorithms_ratios_all_mean(:,algorithm_idx),'Marker',markers{algorithm_idx},'LineStyle',line_styles{4-mod(algorithm_idx,4)});
            hold on
        end
        
        % legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet')
        legend_label = Algorithm_list;
        if sum(strcmp(Algorithm_list, 'GeneralGDOP'))~=0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP')}='AccurateGDOP';
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='WGDOP';
        elseif sum(strcmp(Algorithm_list, 'GeneralGDOP'))==0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='WGDOP';
        end
        % hl = legend(legend_label, 'Location', 'east');  %图例
        % set(hl,'Box','off','location','NorthOutside','NumColumns',4);
        legend(legend_label)
        grid on
        xlabel('$SNR (dB)$','interpreter','latex')
        ylabel('MFNE (m)')
        % breakyaxis([2000 8000]);  % 截断纵坐标
        
        frob_diff = abs(diff(double(frob_algorithms_ratios_all_mean),1,1));
        err_diff = diff(err_list,1,2)';
        err_diff = repmat(err_diff,1,size(frob_diff,2));
        Slope = mean(frob_diff./err_diff,1);%判断鲁棒性的参数
        nan99_Slope_values = [nan99_Slope_values;round(Slope,2)];
        proposed_idx = find(strcmp(Algorithm_list, 'Proposed'));
        nan99_Slope_improved = [nan99_Slope_improved,round((Slope(:)-Slope(proposed_idx))./Slope(:)*100,2)];
    elseif analysis==3.5
        %% 总数据分析3：不同噪声标准差算法对比
        snr_list = [2,6,10,14,18,22];
        % err_list = [0.01,1,2,3,5,8,12];
        % Algorithm_list = {'UNet'};
        % Algorithm_list = {'kriging','RBF','NNM-T','Proposed','GeneralBTD'};
        frob_algorithms_ratios_all_mean = [];
        for err =1:1:length(snr_list)
            err_i = snr_list(err);
            frob_algorithms_all_mean = [];
            for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
                ContrastAlgorithm=Algorithm_list{algorithm_idx};
                frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
                load_path = [load_path0,ContrastAlgorithm,'/fs_10_snr_',num2str(err_i),'/'];
                load_file = [load_path,'frob_list_all_Mont.mat'];
                frob_list_all_Mont_i = load(load_file);
                frob_list_all_Mont_i = frob_list_all_Mont_i.frob_list_all_Mont;
                frob_algorithms_all_mean = [frob_algorithms_all_mean,mean(mean(frob_list_all_Mont_i))];
            end
            frob_algorithms_ratios_all_mean = [frob_algorithms_ratios_all_mean;frob_algorithms_all_mean];
        end
        figure
        for algorithm_idx = 1:1:length(Algorithm_list)
            plot(snr_list,frob_algorithms_ratios_all_mean(:,algorithm_idx),'Marker',markers{algorithm_idx},'LineStyle',line_styles{4-mod(algorithm_idx,4)},'Color',color_order(algorithm_idx,:));
            hold on
        end
        
        % legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet')
        legend_label = Algorithm_list;
        if sum(strcmp(Algorithm_list, 'GeneralGDOP'))~=0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP')}='AccurateGDOP';
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='WGDOP';
        elseif sum(strcmp(Algorithm_list, 'GeneralGDOP'))==0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='WGDOP';
        end
        % hl = legend(legend_label, 'Location', 'east');  %图例
        % set(hl,'Box','off','location','NorthOutside','NumColumns',4);
        legend(legend_label)
        grid on
        xlabel('SNR (dB)','interpreter','latex')
        ylabel('MFNE (m)')
        % breakyaxis([2000 8000]);  % 截断纵坐标
        
        frob_diff = abs(diff(double(frob_algorithms_ratios_all_mean),1,1));
        err_diff = diff(snr_list,1,2)';
        err_diff = repmat(err_diff,1,size(frob_diff,2));
        Slope = mean(frob_diff./err_diff,1);%判断鲁棒性的参数
        nan99_Slope_values = [nan99_Slope_values;round(Slope,2)];
        proposed_idx = find(strcmp(Algorithm_list, 'Proposed'));
        nan99_Slope_improved = [nan99_Slope_improved,round((Slope(:)-Slope(proposed_idx))./Slope(:)*100,2)];

    elseif analysis==4
        %% 总数据分析4：99%缺失值算法还原度可视化
        if strcmp(scene_i,'tdoa1')
            load ([load_path0,'G2.mat'])
            contour_num = 15; %等值线数目
            % contour_values = linspace(3,200,contour_num); %指定数值的等值线列表，列表长度即等值线数目
            contour_values = contour_num;
            bias = 20; %用于确定参考源的位置，在等值线图中选取4个点来展示值
            text_bias = 5; %文本位置偏置，可以保证数字不和图标重合

            figure
            [c,handle]=contour(x_span,y_span,G2(:,:,11),contour_values);
            clabel(c,handle);
            title('Real distribution of $\underline{G}(:,:,11)$.','interpreter','latex');
            xlabel('x(km)');
            ylabel('y(km)');
            hold on
            % add_points(G2(:,:,11),bias,text_bias)

            for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
                ContrastAlgorithm=Algorithm_list{algorithm_idx};
                load_path = [load_path0,ContrastAlgorithm,'/fs_10_ratio_',num2str(NAN_ratio),'/'];
                load_file = [load_path,'recon_tensor.mat'];
                recon_tensor_i = load(load_file);
                recon_tensor_i = recon_tensor_i.recon_tensor;
                figure
                % contour_values = linspace(3,200,contour_num);
                [c,handle]=contour(x_span,y_span,recon_tensor_i(:,:,11),contour_values);
                if frob(G2(:,:,11)-recon_tensor_i(:,:,11))<2e3
                    clabel(c,handle);
                else
                    clabel(c,handle,'manual') % 手动标签，摁enter退出
                end
                if strcmp(ContrastAlgorithm,'GeneralGDOP')
                    title('AccurateGDOP','interpreter','latex');
                elseif strcmp(ContrastAlgorithm,'GeneralGDOP2')
                    title('WGDOP','interpreter','latex');
                else
                    title(ContrastAlgorithm,'interpreter','latex');
                end
                xlabel('x(km)');
                ylabel('y(km)');
                hold on
                % add_points(recon_tensor_i(:,:,11),bias,text_bias)
            end
        end
    elseif analysis==5
        %% 总数据分析5：99%缺失值算法方向信息还原度可视化
        if strcmp(scene_i,'tdoa1')
            bias = 20; %用于确定参考源的位置，在等值线图中选取4个点来展示值
            text_bias = 5; %文本位置偏置，可以保证数字不和图标重合
            G2 = sqrt(load ([load_path0,'G2.mat']).G2);
            recon_T = sqrt(load ([load_path0,'Proposed/fs_10_ratio_0.99/recon_tensor.mat']).recon_tensor);
            slice_idx = [1,4,7,10];
            
            figure
            % plot_tensor = 10*log10(Sigma2_x+10);
            plot_tensor = 10*log10(G2);
            title_str = 'The heatmap of real $\underline{\textbf{G}}$.';
            plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)
            figure
            % plot_tensor = 10*log10(Sigma2_x+10);
            plot_tensor = 10*log10(recon_T);
            title_str = 'The heatmap of reconstructed $\underline{\textbf{G}}$.';
            plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)


            % figure
            % plot_tensor = log10(recon_Tx+10);
            % % plot_tensor = recon_Tx;
            % title_str = 'The heatmap of reconstructed $\underline{\textbf{G}}_x$.';
            % plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)


        end
    end
end

