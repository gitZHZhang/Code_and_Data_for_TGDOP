close all
clear all


fs = 100; %降采样倍数
x_span = -4000:fs:4000;y_span = -4000:fs:4000;z_span = linspace(500,5000,11);
scene_list = {'tdoa1'};
% scene_list = {'doa','tdoa1'};

Mont_times = 50;
NAN_ratio = 0.99;
analysis = 2;
line_styles = {'-','--','-.',':'};
markers = {'o','+','*','s','^','x','p','d'};
% Algorithm_list = {'Kriging','RBF','NNM-T','Proposed','GeneralBTD','GeneralGDOP','UNet'};
Algorithm_list = {'Kriging','RBF','NNM-T','Proposed','GeneralBTD','GeneralGDOP','GeneralGDOP2'};
Algorithm_list = {'Proposed'};
nan99_reconERR_values=[];
nan99_perfoemance_improved=[];
nan99_Slope_values=[];
nan99_Slope_improved=[];
print_MFNE1 = [];
print_MFNE2 = [];
% Algorithm_list = {'kriging','RBF','NNM-T','GeneralBTD','GeneralGDOP'};
for scene_idx = 1:1:length(scene_list)
    scene_i = scene_list{scene_idx};
    load_path0 = ['test16_data/',scene_i,'/'];
    if analysis==1
        color_order = [0    0.4470    0.7410;
            0.8500    0.3250    0.0980;
            0.9290    0.6940    0.1250;
            0.4940    0.1840    0.5560;
            0.4660    0.6740    0.1880;
            0.3010    0.7450    0.9330;
            0.6350    0.0780    0.1840];
        %% 总数据分析1：99%缺失值算法箱线图对比
        Algorithm_list = {'Kriging','RBF','NNM-T','Proposed','GeneralBTD'};
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
            plot(1:1:size(group_i,2), median_v(1:2:end), 'Marker',markers{i},'LineStyle',line_styles{4-mod(i,4)})
            hold on;
            % color_order = get(gca,'colororder');
            % median_v(end)
        end
        xticklabels(10:1:20)
        if(fillbox)%对box进行颜色填充
            boxobj = findobj(gca, 'Tag', 'Box');
            for i = 1:length(boxobj)
                patch(get(boxobj(i), 'XData'), get(boxobj(i), 'YData'), filledcolor(length(boxobj)-i+1,:), 'FaceAlpha', 0.5)
            end
        end

        legend(Algorithm_list)
        % legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet','GeneralGDOP')
        xlabel('Height (km)')
        ylabel('Reconstruction error.')
        title('Boxplot of reconstruction errors across various algorithms.')

    elseif analysis==2
        
        %% 总数据分析2：不同缺失值算法对比
        ratio_list = [0.8,0.85,0.9,0.95,0.99];
        % ratio_list = [0.8];
        % Algorithm_list = {'GeneralGDOP2'};
        frob_algorithms_ratios_all_mean = [];
        frob_algorithms_ratios_all_rms = [];
        for ratio =1:1:length(ratio_list)
            ratio_i = ratio_list(ratio);
            frob_algorithms_all_mean = [];
            frob_algorithms_all_rms = [];
            for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
                ContrastAlgorithm=Algorithm_list{algorithm_idx};
                load_path = [load_path0,ContrastAlgorithm,'/fs_10_ratio_',num2str(ratio_i),'/'];
                load_file = [load_path,'frob_list_all_Mont.mat'];
                frob_list_all_Mont_i = load(load_file);        
                frob_list_all_Mont_i = frob_list_all_Mont_i.frob_list_all_Mont;
                
                % slice = [1,2,9];
                % frob_list_all_Mont = frob_list_all_Mont_i;
                % frob_list_all_Mont.errG = frob_list_all_Mont.errG(slice,:);
                % frob_list_all_Mont.errGx = frob_list_all_Mont.errGx(slice,:);
                % frob_list_all_Mont.errGy = frob_list_all_Mont.errGy(slice,:);
                % frob_list_all_Mont.errGz = frob_list_all_Mont.errGz(slice,:);

                % load_path_11 = ['test11_data/',scene_i,'/'];
                % load_path11 = [load_path_11,ContrastAlgorithm,'/fs_10_ratio_',num2str(ratio_i),'/'];
                % load_file11 = [load_path11,'frob_list_all_Mont.mat'];
                % frob_list_all_Mont_i11 = load(load_file11);        
                % frob_list_all_Mont_i11 = frob_list_all_Mont_i11.frob_list_all_Mont;

                
                % frob_algorithms_all_mean = [frob_algorithms_all_mean,mean(mean(frob_list_all_Mont_i.errG))];
                % frob_algorithms_all_rms = [frob_algorithms_all_rms,mean(rms([mean(frob_list_all_Mont_i.errGx,2),mean(frob_list_all_Mont_i.errGy,2),mean(frob_list_all_Mont_i.errGz,2)],2))];
                % frob_algorithms_all_mean = [frob_algorithms_all_mean,mean(rms(frob_list_all_Mont_i.errG,2))];
                % frob_algorithms_all_rms = [frob_algorithms_all_rms,mean(rms(frob_list_all_Mont_i.errGx,2))+mean(rms(frob_list_all_Mont_i.errGy,2))+mean(rms(frob_list_all_Mont_i.errGz,2)) ];
                frob_algorithms_all_mean = [frob_algorithms_all_mean,mean(mean(frob_list_all_Mont_i.errG,2))];
                frob_algorithms_all_rms = [frob_algorithms_all_rms,(mean(mean(frob_list_all_Mont_i.errGx))+mean(mean(frob_list_all_Mont_i.errGy))+mean(mean(frob_list_all_Mont_i.errGz)))/3 ];
            end
            frob_algorithms_ratios_all_mean = [frob_algorithms_ratios_all_mean;frob_algorithms_all_mean];
            frob_algorithms_ratios_all_rms = [frob_algorithms_ratios_all_rms;frob_algorithms_all_rms];
        end
        print_MFNE1 = [print_MFNE1,frob_algorithms_ratios_all_mean(:,1)];
        print_MFNE2 = [print_MFNE2,frob_algorithms_ratios_all_rms(:,1)];
        figure
        for algorithm_idx = 1:1:length(Algorithm_list)
            % plot(ratio_list*100,frob_algorithms_ratios_all_mean(:,algorithm_idx),'*-');
            plot(ratio_list*100,frob_algorithms_ratios_all_rms(:,algorithm_idx),'Marker',markers{algorithm_idx},'LineStyle',line_styles{4-mod(algorithm_idx,4)});
            hold on
        end
        nan99_reconERR_values = [nan99_reconERR_values;round(frob_algorithms_ratios_all_mean(end,:),2)];
        proposed_idx = find(strcmp(Algorithm_list, 'Proposed'));
        nan99_perfoemance_improved = [nan99_perfoemance_improved;round((frob_algorithms_ratios_all_mean(end,:)-frob_algorithms_ratios_all_mean(end,proposed_idx))./frob_algorithms_ratios_all_mean(end,:)*100,2)];

        % legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','GeneralGDOP','UNet')
        legend_label = Algorithm_list;
        if sum(strcmp(Algorithm_list, 'GeneralGDOP'))~=0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP')}='AccurateGDOP';
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='GeneralGDOP';
        end
        legend(legend_label)
        % legend('Kriging','RBF','NNM-T','GeneralBTD','GeneralGDOP')
        grid on
        xlabel('NaNRatio (%).')
        ylabel('Reconstruction error.')
    elseif analysis==3
        %% 总数据分析3：不同噪声标准差算法对比
        err_list = [0.01,1,2,3,5,8,12,16];
        Algorithm_list = {'Proposed'};
        % Algorithm_list = {'kriging','RBF','NNM-T','Proposed','GeneralBTD'};
        frob_algorithms_ratios_all_mean = [];
        frob_algorithms_ratios_all_rms = [];
        for err =1:1:length(err_list)
            err_i = err_list(err);
            frob_algorithms_all_mean = [];
            frob_algorithms_all_rms = [];
            for algorithm_idx = 1:1:length(Algorithm_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
                ContrastAlgorithm=Algorithm_list{algorithm_idx};
                frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
                load_path = [load_path0,ContrastAlgorithm,'/fs_10_err_',num2str(err_i),'/'];
                load_file = [load_path,'frob_list_all_Mont.mat'];
                frob_list_all_Mont_i = load(load_file);
                frob_list_all_Mont_i = frob_list_all_Mont_i.frob_list_all_Mont;
                frob_algorithms_all_mean = [frob_algorithms_all_mean,mean(mean(frob_list_all_Mont_i.errG))];
                frob_algorithms_all_rms = [frob_algorithms_all_rms,mean(rms([mean(frob_list_all_Mont_i.errGx,2),mean(frob_list_all_Mont_i.errGy,2),mean(frob_list_all_Mont_i.errGz,2)],2))];
            end
            frob_algorithms_ratios_all_mean = [frob_algorithms_ratios_all_mean;frob_algorithms_all_mean];
            frob_algorithms_ratios_all_rms = [frob_algorithms_ratios_all_rms;frob_algorithms_all_rms];
        end
        print_matrix = [print_matrix,frob_algorithms_ratios_all_mean];
        figure
        for algorithm_idx = 1:1:length(Algorithm_list)
            plot(err_list,frob_algorithms_ratios_all_mean(:,algorithm_idx),'Marker',markers{algorithm_idx},'LineStyle',line_styles{4-mod(algorithm_idx,4)});
            hold on
        end
        % legend('Kriging','RBF','NNM-T','Proposed','GeneralBTD','UNet')
        legend_label = Algorithm_list;
        if sum(strcmp(Algorithm_list, 'GeneralGDOP'))~=0 && sum(strcmp(Algorithm_list, 'GeneralGDOP2'))~=0
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP')}='AccurateGDOP';
            legend_label{strcmp(Algorithm_list, 'GeneralGDOP2')}='GeneralGDOP';
        end
        legend(legend_label)
        grid on
        xlabel('Standard deviation of observation error \underline{\textbf{N}}.','interpreter','latex')
        ylabel('Reconstruction error.')

        frob_diff = abs(diff(double(frob_algorithms_ratios_all_mean),1,1));
        err_diff = diff(err_list,1,2)';
        err_diff = repmat(err_diff,1,size(frob_diff,2));
        Slope = mean(frob_diff./err_diff,1);%判断鲁棒性的参数
        nan99_Slope_values = [nan99_Slope_values;round(Slope,2)];
        proposed_idx = find(strcmp(Algorithm_list, 'Proposed'));
        nan99_Slope_improved = [nan99_Slope_improved,round((Slope(:)-Slope(proposed_idx))./Slope(:)*100,2)];

    elseif analysis==4
        %% 总数据分析4：99%缺失值算法还原度可视化
        if strcmp(scene_i,'tdoa1')
            G2 = load ([load_path0,'G2.mat']).G2;
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
                load_file = [load_path,'recon_T.mat'];
                recon_tensor_i = load(load_file);
                recon_tensor_i = recon_tensor_i.recon_T;
                recon_tensor_i = recon_tensor_i.G;
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
                    title('GeneralGDOP','interpreter','latex');
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
            Sigma2_x = sqrt(load ([load_path0,'Sigma2_x.mat']).Sigma2_x);
            Sigma2_y = sqrt(load ([load_path0,'Sigma2_y.mat']).Sigma2_y);
            Sigma2_z = sqrt(load ([load_path0,'Sigma2_z.mat']).Sigma2_z);
            contour_num = 10; %等值线数目
            % contour_values = linspace(3,200,contour_num); %指定数值的等值线列表，列表长度即等值线数目
            contour_values = contour_num;
            bias = 20; %用于确定参考源的位置，在等值线图中选取4个点来展示值
            text_bias = 5; %文本位置偏置，可以保证数字不和图标重合
            recon_T = load ([load_path0,'Proposed/fs_10_ratio_0.99/recon_T.mat']).recon_T;
            recon_T.Gx(recon_T.Gx<0)=0;
            recon_T.Gy(recon_T.Gy<0)=0;
            recon_T.Gz(recon_T.Gz<0)=0;
            % recon_Tx = recon_T.Gx;
            recon_Tx = real(sqrt(recon_T.Gx));
            recon_Ty = real(sqrt(recon_T.Gy));
            recon_Tz = real(sqrt(recon_T.Gz));
            
            %% Generate 3D heatmap images
            % [X,Y,Z] = meshgrid(x_span,y_span,z_span);
            % im1 = Sigma2_x(:,:,1);  %// extract the slice at Z=5.  im1 size is [25x50]
            % figure() ;
            % imagesc(flipud(im1));title('Z=1');
            % colormap('jet');
            % colorbar;
            slice_idx = [1,4,7,10];
            
            figure
            plot_tensor = 10*log10(Sigma2_x+10);
            % plot_tensor = Sigma2_x;
            title_str = 'The heatmap of real $\underline{\textbf{G}}_x$.';
            plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)


            % figure
            % plot_tensor = log10(recon_Tx+10);
            % % plot_tensor = recon_Tx;
            % title_str = 'The heatmap of reconstructed $\underline{\textbf{G}}_x$.';
            % plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)
    
            figure
            plot_tensor = 10*log10(Sigma2_y+10); %这里计算dB取10而不是20，是因为本来我们希望绘画的是sqrt(G),对应的dB是减半的
            % plot_tensor = Sigma2_y;
            title_str = 'The heatmap of real $\underline{\textbf{G}}_y$.';
            plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)

            figure
            plot_tensor = 10*log10(Sigma2_z+10);
            % plot_tensor = Sigma2_z;
            title_str = 'The heatmap of real $\underline{\textbf{G}}_z$.';
            plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)

            figure
            plot_tensor = 10*log10(recon_Tx+10);
            % plot_tensor = recon_Tx;
            title_str = 'The heatmap of reconstructed $\underline{\textbf{G}}_x$.';
            plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)

            figure
            plot_tensor = 10*log10(recon_Ty+10);
            % plot_tensor = recon_Ty;
            title_str = 'The heatmap of reconstructed $\underline{\textbf{G}}_y$.';
            plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)

            figure
            plot_tensor = 10*log10(recon_Tz+10);
            % plot_tensor = recon_Tz;
            title_str = 'The heatmap of reconstructed $\underline{\textbf{G}}_z$.';
            plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)
            
            figure
            [c,handle]=contour(x_span,y_span,Sigma2_x(:,:,1),contour_values);
            clabel(c,handle);
            title('$\underline{G}_x(:,:,1)$.','interpreter','latex');
            xlabel('x(km)');
            ylabel('y(km)');
            %axis equal
            figure
            [c,handle]=contour(x_span,y_span,Sigma2_y(:,:,1),contour_values);
            clabel(c,handle);
            title('$\underline{G}_y(:,:,1)$.','interpreter','latex');
            xlabel('x(km)');
            ylabel('y(km)');
            %axis equal
            figure
            [c,handle]=contour(x_span,y_span,Sigma2_z(:,:,1),contour_values);
            clabel(c,handle);
            title('$\underline{G}_z(:,:,1)$.','interpreter','latex');
            xlabel('x(km)');
            ylabel('y(km)');
            %axis equal
      
            
            figure
            [c,handle]=contour(x_span,y_span,recon_Tx(:,:,1),contour_values);
            clabel(c,handle);
            title('Reconstructed $\underline{G}_x(:,:,1)$.','interpreter','latex');
            xlabel('x(km)');
            ylabel('y(km)');
            %axis equal
            figure
            [c,handle]=contour(x_span,y_span,recon_Ty(:,:,1),contour_values);
            clabel(c,handle);
            title('Reconstructed $\underline{G}_y(:,:,1)$.','interpreter','latex');
            xlabel('x(km)');
            ylabel('y(km)');
            %axis equal
            figure
            [c,handle]=contour(x_span,y_span,recon_Tz(:,:,1),contour_values);
            clabel(c,handle);
            title('Reconstructed $\underline{G}_z(:,:,1)$.','interpreter','latex');
            xlabel('x(km)');
            ylabel('y(km)');
            %axis equal


        end
    end
end
print_matrix = round(print_matrix,2);
