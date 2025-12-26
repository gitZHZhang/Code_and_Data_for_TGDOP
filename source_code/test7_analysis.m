close all
clear all
L=30;
xt = 0;yt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);
x3 = 0;y3 = -L;
sensor = [xt,yt;x1,y1;x2,y2;x3,y3];
y=-400:1:400;x=-400:1:400;

trigger = 2;
switch trigger
    case 0
        %% 0 计算GDOP的2维空间分布
        [gdop,G,sigma2_x,sigma2_y] = deal(zeros(length(x),length(y)));
        for i=1:length(x)
            for j=1:length(y)
                m=x(i)+0.01;
                n=y(j)+0.01;
                emitter = [m,n];
                [gdop_ji,~,~,~] = cal_gdop(emitter,sensor);
                gdop(j,i) = gdop_ji;
            end
        end
        figure();
        plot(sensor(:,1),sensor(:,2),'r.');hold on
        %contour(Z,n),n指定了等高线的条数，用于绘制矩阵的等高线
        [c,handle]=contour(x,y,gdop,20);
        clabel(c,handle);
        xlabel('x方向(单位:km)');
        ylabel('y方向(单位:km)');
        title('GDOP');
        hold on;

    case 1
        %% 1 根据误差协方差矩阵给出误差椭圆&与采样数据做对比（解析vs数值）
        emitter_i=[100,100];
        [gdop,~,~,P] = cal_gdop(emitter_i,sensor);
        %1-计算得到的误差椭圆方程组
        figure
        Pe=0.5;%CEP概率圆
        [f1,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
        fimplicit(f1,[min(x),max(x),min(y),max(y)],'b.-');hold on
        Pe=0.99;
        [f2,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
        fimplicit(f2,[min(x),max(x),min(y),max(y)],'r.-');hold on
        %2-数值采样得到的误差椭圆(与计算结果一致)
        n=1000;
        samples = mvnrnd(emitter_i, P, n);
        % samples = mvnrnd([0,0], P, n);
        scatter(samples(:,1),samples(:,2));hold on

        emitter_i=[-100,-100];
        [gdop,~,~,P] = cal_gdop(emitter_i,sensor);
        %1-计算得到的误差椭圆方程组
        figure
        Pe=0.5;%CEP概率圆
        [f1,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
        fimplicit(f1,[min(x),max(x),min(y),max(y)],'b.-');hold on
        Pe=0.99;
        [f2,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
        fimplicit(f2,[min(x),max(x),min(y),max(y)],'r.-');hold on
        n=1000;
        samples = mvnrnd(emitter_i, P, n);
        scatter(samples(:,1),samples(:,2));hold on

        emitter_i=[-100,100];
        [gdop,~,~,P] = cal_gdop(emitter_i,sensor);
        %1-计算得到的误差椭圆方程组
        figure
        Pe=0.5;%CEP概率圆
        [f1,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
        fimplicit(f1,[min(x),max(x),min(y),max(y)],'b.-');hold on
        Pe=0.99;
        [f2,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
        fimplicit(f2,[min(x),max(x),min(y),max(y)],'r.-');hold on
        n=1000;
        samples = mvnrnd(emitter_i, P, n);
        scatter(samples(:,1),samples(:,2));hold on

        emitter_i=[100,-100];
        [gdop,~,~,P] = cal_gdop(emitter_i,sensor);
        %1-计算得到的误差椭圆方程组
        figure
        Pe=0.5;%CEP概率圆
        [f1,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
        fimplicit(f1,[min(x),max(x),min(y),max(y)],'b.-');hold on
        Pe=0.99;
        [f2,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
        fimplicit(f2,[min(x),max(x),min(y),max(y)],'r.-');hold on
        n=1000;
        samples = mvnrnd(emitter_i, P, n);
        scatter(samples(:,1),samples(:,2));hold on
    case 2
        %% 2 多次实验统计误差提升程度
        MC=2000;%蒙特卡洛实验次数
        n=25;    %同一目标的定位点数
        measured_err = [];
        estimated_err = [];
        real_pos_list = [];
        n_list = [2,3,5,7,10,15,20,25];
        % n_list = [10];
        mean_measured_err_versus_n = [];
        mean_estimated_err_versus_n = [];
        all_improvement_versus_n = [];
        for n_index = 1:1:length(n_list)
            n=n_list(n_index);
            load(['test7_data/n=',num2str(n),'/measured_err.mat'])
            load(['test7_data/n=',num2str(n),'/estimated_err.mat'])
            mean_measured_err_versus_n = [mean_measured_err_versus_n,mean(measured_err)];
            mean_estimated_err_versus_n = [mean_estimated_err_versus_n,mean(estimated_err)];
            improvement_n = (measured_err-estimated_err)./measured_err;
            all_improvement_versus_n = [all_improvement_versus_n,improvement_n'];
            % figure
            % % subplot(2,2,3)
            % plot(estimated_err,'b.');hold on
            % plot(measured_err,'r.');hold on
            % plot(ones(1,length(estimated_err))*mean(estimated_err),'b-');hold on
            % plot(ones(1,length(measured_err))*mean(measured_err),'r-');hold on
            % h=legend('$$\hat{\textbf{dr}}$$','$$\textbf{dr}$$','$$\mu(\hat{\textbf{dr}})$$','$$\mu(\textbf{dr})$$');
            % set(h,'Interpreter','latex')
            % xlabel('Monte Carlo index')
            % ylabel('Positioning error (km)')
            % title(['Results of 2000 Monte-Carlo experiments with N = ',num2str(n)])
            % 
            % figure
            % % subplot(2,2,4)
            % histogram(estimated_err);hold on
            % histogram(measured_err)
            % h=legend('$$\hat{\textbf{dr}}$$','$$\textbf{dr}$$');
            % set(h,'Interpreter','latex')
            % xlabel('Positioning error (km)')
            % ylabel('Count')
            % title(['Histograms of 2000 Monte-Carlo experiments with N = ',num2str(n)])
        end
        % figure
        % plot(n_list,(mean_measured_err_versus_n - mean_estimated_err_versus_n)./mean_measured_err_versus_n);

        figure
        box_i = boxplot(all_improvement_versus_n,'Widths',0.4,'Symbol','o','OutlierSize',0.5);
        h = findobj(gca, 'Tag', 'Median');
        median_v = get(h,'YData');
        median_v = median_v(1:size(all_improvement_versus_n,2));
        median_v = flip([median_v{:}]);
        set(box_i,'LineWidth',1.5);
        hold on;
        % plot(1:1:size(group_i,2), median_v(1:2:end), '-*', 'Color', filledcolor_i)
        plot(1:1:size(all_improvement_versus_n,2),median_v(1:2:end))
        hold on;
        xticks(1:1:length(n_list))
        xlabeltick = {};
        for n_i = 1:1:length(n_list)
            xlabeltick{n_i} = num2str(n_list(n_i));
        end
        xticklabels(xlabeltick)
        ylim([-0.5,1.5])
        xlabel('N')
        ylabel('improvement (%)')
        title('Boxplots of the improvement under differ N.')
    case 3
        %% 3 给出若干个定位结果，根据GDOP给出其真值最可能的位置
        % %1- 单次实验
        % real_pos = [100,100];%真实位置
        % [gdop,~,~,P] = cal_gdop(real_pos,sensor);
        % n=2;
        % measured_pos = mvnrnd(real_pos, P, n);%观测到的定位结果
        % figure
        % % subplot(2,2,1)
        % plot(100,100,'rp','MarkerSize',12);hold on
        % Pe = 0.5;
        % [f3,~,~,~]=Draw_err_ellipse(P,Pe,real_pos);
        % fimplicit(f3,[98.5,101.5,98.5,101.5],'b.-');hold on
        % Pe = 0.95;
        % [f4,~,~,~]=Draw_err_ellipse(P,Pe,real_pos);
        % fimplicit(f4,[98.5,101.5,98.5,101.5],'g.-');hold on
        % scatter(measured_pos(:,1),measured_pos(:,2));hold on
        % xlabel('x(km)')
        % ylabel('y(km)')
        % legend('Real position','Ellipse of 50% probability error contours','Ellipse of 95% probability error contours','Location results','Location','northwest')
        % loss = [];
        % for grid_x = 90:0.1:110
        %     for grid_y = 90:0.1:110
        %         [gdop,~,~,P_i] = cal_gdop([grid_x,grid_y],sensor);
        %         [f4,theta,sigma1_2,sigma2_2]=Draw_err_ellipse(P_i,Pe,[grid_x,grid_y]);
        %         % theta = 1/2*atan(2*P(1,2)/(P(1,1)-P(2,2)));
        %         % if grid_y<0
        %         %     theta = theta - pi/2;
        %         % end
        %         % sigma1_2 = (2*P(1,1)*P(2,2)-2*P(1,2)^2) / (P(1,1)+P(2,2)-sqrt((P(1,1)-P(2,2))^2+4*P(1,2)^2));
        %         % sigma2_2 = (2*P(1,1)*P(2,2)-2*P(1,2)^2) / (P(1,1)+P(2,2)+sqrt((P(1,1)-P(2,2))^2+4*P(1,2)^2));
        %         sum_term = 0;
        %         for i=1:1:n
        %             pos_i = measured_pos(i,:);
        %             sum_term = sum_term + ((pos_i(1)-grid_x)*cos(theta)+(pos_i(2)-grid_y)*sin(theta))^2/sigma1_2 + ...
        %                 (-(pos_i(1)-grid_x)*sin(theta)+(pos_i(2)-grid_y)*cos(theta))^2/sigma2_2;
        %         end
        %         loss_i = -n*log(2*pi*sqrt(sigma1_2)*sqrt(sigma2_2))-1/2*sum_term;
        %         loss = [loss;grid_x,grid_y,loss_i];
        %     end
        % end
        % loss(:,3) = exp(loss(:,3));
        % [a,idx]=max(loss(:,3));
        % estimated_pos = loss(idx,1:2);
        % figure
        % % subplot(2,2,2)
        % scatter3(loss(:,1),loss(:,2),loss(:,3))
        % text(loss(idx,1),loss(idx,2),loss(idx,3),['[',num2str(loss(idx,1)),',',num2str(loss(idx,2)),',',num2str(floor(loss(idx,3))),']'],'Color','b');
        % xlabel('x(km)')
        % ylabel('y(km)')
        % zlabel('Likelihood value')
end