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
rate_list = [0.4,0.6,0.8];
% rate_list = [0.6,0.8];
trigger = 1;



if trigger==1
    print_for_latex = zeros(4*4,3);
    for flight = 1:1:4
        dir_pos = ['test18_data/flight',num2str(flight),'/'];
        %% 1-比较训练数据比例对不同算法的影响
        time_all = [];
        ERR_all = [];
        recon_rate_all = [];
        for rate_idx = 1:1:length(rate_list)
            rate_i = rate_list(rate_idx);
            load_path = [dir_pos,'rate=',num2str(rate_i),'/'];
            time_list = load([load_path,'time_list.mat']).time_list;
            ERR_list = load([load_path,'ERR_list.mat']).ERR_list;
            recon_rate_list = load([load_path,'recon_rate_list.mat']).recon_rate_list;
            % 去除含NaN的列
            nanMask = isnan(ERR_list);
            colsWithNaN = any(nanMask, 1);% 使用 any 函数沿着行的方向检查每列是否包含 NaN
            time_list(:,colsWithNaN)=[];
            ERR_list(:,colsWithNaN)=[];
            recon_rate_list(:,colsWithNaN)=[];

            time_all = [time_all,mean(time_list,2)];
            ERR_all = [ERR_all,mean(ERR_list,2)];
            recon_rate_all = [recon_rate_all,mean(recon_rate_list,2)];

        end
        disp(dir_pos)
        enhance = (ERR_all-ERR_all(2,:))./ERR_all;
        %----格式化数据，便于直接复制到latex中
        cnt = 0;
        for i = 1:1:size(ERR_all,1)
            if i ~= 2
                cnt = cnt + 1;
                % print_for_latex = [print_for_latex;round(ERR_all(i,:),2);round(enhance(i,:)*100,2)];
                % print_for_latex = [print_for_latex;round(enhance(i,:)*100,2)];
                print_for_latex((cnt-1)*4+flight,:) = round(enhance(i,:)*100,2);
                
            end
        end
        format long g
        % print_for_latex = [print_for_latex;round(ERR_all(2,:),2)]

    end
    print_for_latex
elseif trigger == 2
    %% 2-实验地图和场景可视化
    flight = 1;
    dir_pos = ['E:\zzh清华\时差定位\河北UAV\origin_data_20220224\data\flight',num2str(flight)];
    text1=[dir_pos,'\',num2str(flight),'.xlsx'];
    auto1=importdata(text1);
    auto2=auto1.data;%位置信息
    auto3=auto1.textdata;%时间信息
    time1=auto2(:,1)/1000;
    time2=auto3(2:end,2);
    time2=char(time2);
    time2_0=time2(:,end-7:end);
    time2_1=str2num(time2_0(:,end-1:end));
    time2_2=str2num(time2_0(:,end-4:end-3));
    time2_3=str2num(time2_0(:,end-7:end-6));
    char1=time2_1(1);
    second1=[1];%存放字符串矩阵不同字符出现位置，这里用unique函数可以直接获取
    for k1=1:length(time2_1)
        if time2_1(k1)~=char1
            second1=[second1;k1];
            char1=time2_1(k1);
        end
    end
    dronedata=zeros(length(second1),3);
    % dronedata数据格式分别是 1.时 2.分 3.秒 4.纬度 5.经度 6高度 7.UAV到主楼的距离 8.UAV高度
    for k3=1:length(second1)
        dronedata(k3,1)=auto2(second1(k3),3);%lat
        dronedata(k3,2)= auto2(second1(k3),4);%lon
        dronedata(k3,3)= auto2(second1(k3),5)*0.3048;%原单位是feet，变成米
    end
    wm=webmap('World Imagery');
    zoomLevel=3;
    wmcenter(wm,28,124);
    wmzoom(wm,zoomLevel);
    wmline(dronedata(:,1), dronedata(:,2), ...
        'Color', 'red', ...  % 鲜艳的红色 (Hex Color Code)
        'Width', 3, ...          % 线条宽度为3个像素
        'Description', 'UAV Trajectory'); % 在图层管理器中显示的名称
    wmmarker(station_positions1(:,2), station_positions1(:,1), ...
        'Color','yellow',...
        'FeatureName','stations',...
        'IconScale', 1.2, ...           % 图标缩放比例
        'Description', 'stations');   % 鼠标悬停时显示的描述


elseif trigger==3
    %% 3-所提算法的重建可视化
    pos_err = load('test9_data\4Err_data\pos_err.mat').pos_err;
    pos_list2 = pos_err.cal_pos;
    x_span = linspace(min(pos_list2(:,1)),1000,200);
    y_span = linspace(min(pos_list2(:,2)),1000,200);
    z_span = linspace(min(pos_list2(:,3)),max(pos_list2(:,3)),10);
    proposed_alg_list = {'proposed1','proposed1-2','proposed2'};
    for alg_idx = 2:1:length(proposed_alg_list)
        load_path = ['test10_data/proposed/',proposed_alg_list{alg_idx},'/'];
        recon_tensor = sqrt(load([load_path,'recon_tensor.mat']).recon_tensor);
        Sigma2_x_hat = sqrt(load([load_path,'Sigma2_x_hat.mat']).Sigma2_x_hat);
        Sigma2_y_hat = sqrt(load([load_path,'Sigma2_y_hat.mat']).Sigma2_y_hat);
        Sigma2_z_hat = sqrt(load([load_path,'Sigma2_z_hat.mat']).Sigma2_z_hat);
        slice_idx = [1,4,7,10];

        figure
        plot_3D_realData_track(pos_err,st2,x_span,y_span,z_span)
        title_str = 'The heatmap of $\underline{\textbf{G}}$.';
        % plot_tensor = log(recon_tensor+1);
        plot_tensor = 10*log10(recon_tensor);
        plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span); hold on
        legend('GPS trajectory','Positioning trajectory','Sensors','Location','BestOutside');
        legend('boxoff')
        xlabel('x(m)')
        ylabel('y(m)')
        zlabel('z(m)')

        figure
        title_str = 'The heatmap of $\underline{\textbf{G}}_x$.';
        % plot_tensor = log(Sigma2_x_hat+1);
        plot_tensor = 10*log10(Sigma2_x_hat);
        plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)
        xlabel('x(m)')
        ylabel('y(m)')
        zlabel('z(m)')

        figure
        title_str = 'The heatmap of $\underline{\textbf{G}}_y$.';
        % plot_tensor = log(Sigma2_y_hat+1);
        plot_tensor = 10*log10(Sigma2_y_hat);
        plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)
        xlabel('x(m)')
        ylabel('y(m)')
        zlabel('z(m)')

        figure
        title_str = 'The heatmap of $\underline{\textbf{G}}_z$.';
        % plot_tensor = log(Sigma2_z_hat+1);
        plot_tensor = 10*log10(Sigma2_z_hat);
        plot_3D_heatmap(plot_tensor,slice_idx,title_str,x_span,y_span,z_span)
        xlabel('x(m)')
        ylabel('y(m)')
        zlabel('z(m)')
        
    end
    % figure();
    % [c,handle]=contour(x_span*1e-3,y_span*1e-3,Gx_cal(:,:,1)',100); hold on
    % clabel(c,handle,'manual') % 手动标签，摁enter退出
    % % v = [0.01,0.02,0.0522621,0.134494,0.267897392624881,0.622787];
    % % clabel(c,handle,v);
    % plot(pos_err.real_pos(:,1)*1e-3,pos_err.real_pos(:,2)*1e-3,'.-'); hold on
    % plot(pos_err.cal_pos(:,1)*1e-3,pos_err.cal_pos(:,2)*1e-3,'.-'); hold on
    % plot(st2(:,1)*1e-3,st2(:,2)*1e-3,'rp','MarkerSize',10,'MarkerFaceColor','r'); hold on;
    % legend('contour','GPS trajectory','Positioning trajectory','stations')
    % title('Reconstructed distribution of $G_y$','interpreter','latex');
    % xlabel('x(km)');
    % ylabel('y(km)');
    
end