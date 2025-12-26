close all
clear all


xt = 0;yt = 0;zt = 0;
x1 = 640;y1 = 1070;z1 = -35;
x2 = -900;y2 = -180;z2 = -27;
x3 = 1000;y3 = -660;z3 = -35;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];
fs = 100; %降采样倍数
x_span = -4000:fs:4000;y_span = -4000:fs:4000;z_span = linspace(500,5000,11);
vars_real = [18e-3,20e-3,25e-3,0.5,-0.3,0.5,-0.2]; %us us us m
% y=-400:1:400;x=-400:1:400;


% vars_real = [30e-3,20e-3,5e-3,1.0,0.5,0.4,0.2]; %us us us m
% real_pos = [-2000,1000,500];%真实位置
% P1 = cal_diff_P(real_pos,sensor,vars_real);
% delta1 = P1 - cal_diff_P(real_pos+1,sensor,vars_real);
% real_pos = [-200,100,500];%真实位置
% P2 = cal_diff_P(real_pos,sensor,vars_real);
% delta2 = P2 - cal_diff_P(real_pos+1,sensor,vars_real);
% pinv_P1addP2 = pinv(P1+P2);
% W = P2*pinv_P1addP2;
% tic
% W2_real = (P2+delta2)*pinv(P1+P2+delta1+delta2);
% toc
% tic
% W2_deduce = W + pinv_P1addP2*delta2-P2*pinv_P1addP2*(delta1+delta2)*pinv_P1addP2;
% toc

trigger = 2;
switch trigger
    case 0
        gdop = sqrt(load('test11_data/tdoa1/G2.mat').G2);
        figure();
        plot(sensor(:,1),sensor(:,2),'r.');hold on
        %contour(Z,n),n指定了等高线的条数，用于绘制矩阵的等高线
        [c,handle]=contour(x_span,y_span,gdop(:,:,1),20);
        clabel(c,handle);
        xlabel('x方向(单位:m)');
        ylabel('y方向(单位:m)');
        title('GDOP');
        hold on;

    case 1
        %% case1-根据误差协方差矩阵给出误差椭圆&与采样数据做对比（解析vs数值）
        %% 1*sigma Pe=67;2*sigma Pe=95;3*sigma Pe=99
        emitter_i=[500,500,500];

        P = cal_diff_P(emitter_i,sensor,vars_real);
        %1-计算得到的误差椭圆方程组
        figure
        Pe=0.5;%CEP概率圆
        [f1,~,~,~]=Draw_err_ellipse(P(1:2,1:2),Pe,emitter_i);
        fimplicit(f1,[min(x_span),max(x_span),min(y_span),max(y_span)],'b.-');hold on
        Pe=0.95;
        [f2,~,~,~]=Draw_err_ellipse(P(1:2,1:2),Pe,emitter_i);
        fimplicit(f2,[min(x_span),max(x_span),min(y_span),max(y_span)],'r.-');hold on
        %2-数值采样得到的误差椭圆(与计算结果一致)
        n=1000;
        samples = mvnrnd(emitter_i(1:2), P(1:2,1:2), n);
        % samples = mvnrnd([0,0], P, n);
        scatter(samples(:,1),samples(:,2));hold on

        emitter_i=[-2000,1000,500];
        P = cal_diff_P(emitter_i,sensor,vars_real);
        %1-计算得到的误差椭圆方程组
        Pe=0.5;%CEP概率圆
        [f1,~,~,~]=Draw_err_ellipse(P(1:2,1:2),Pe,emitter_i);
        fimplicit(f1,[min(x_span),max(x_span),min(y_span),max(y_span)],'b.-');hold on
        Pe=0.95;
        [f2,~,~,~]=Draw_err_ellipse(P(1:2,1:2),Pe,emitter_i);
        fimplicit(f2,[min(x_span),max(x_span),min(y_span),max(y_span)],'r.-');hold on
        %2-数值采样得到的误差椭圆(与计算结果一致)
        n=1000;
        samples = mvnrnd(emitter_i(1:2), P(1:2,1:2), n);
        % samples = mvnrnd([0,0], P, n);
        scatter(samples(:,1),samples(:,2));hold on
        plot(sensor(:,1),sensor(:,2),'p');hold on
        legend('50%概率圆','95%概率圆','定位点集1','','','定位点集2','基站')


    case 2
        %% case2-定位增强：给出若干个定位结果，根据GDOP给出其真值最可能的位置
        % %1- 单次实验
        
        n=20;

        % real_pos = [-2000,1000,500];%真实位置
        % % real_pos = [-2010,1010,500];%真实位置
        % vars_real = [30e-3,20e-3,5e-3,1.0,0.5,0.4,0.2]; %us us us m
        % P = cal_diff_P(real_pos,sensor,vars_real);
        % real_pos = real_pos(1:2);
        % P = P(1:2,1:2);
        % Pe = 0.95;
        % [f3,~,~,~]=Draw_err_ellipse(P,Pe,real_pos);
        % measured_pos1 = mvnrnd(real_pos, P, n);

        real_pos = [-2000,1000,500];%真实位置
        % real_pos = [-200,100,500];%真实位置
        vars_real = [18e-3,20e-3,25e-3,0.5,-0.3,0.5,-0.2]; %us us us m
        P = cal_diff_P(real_pos,sensor,vars_real);
        real_pos = real_pos(1:2);
        P = P(1:2,1:2);
        Pe = 0.95;
        [f4,~,~,~]=Draw_err_ellipse(P,Pe,real_pos);
        measured_pos = mvnrnd(real_pos, P, n);
        measured_pos = [measured_pos;real_pos+200];
        % measured_pos = [measured_pos1;measured_pos2];

        figure
        plot(real_pos(1),real_pos(2),'rp','MarkerSize',12);hold on
        % fimplicit(f3,[98.5,101.5,98.5,101.5],'b.-');hold on
        % fimplicit(f4,[98.5,101.5,98.5,101.5],'g.-');hold on
        % fimplicit(f3,[real_pos(1)-500,real_pos(1)+500,real_pos(2)-500,real_pos(2)+500],'b.-');hold on
        fimplicit(f4,[real_pos(1)-500,real_pos(1)+500,real_pos(2)-500,real_pos(2)+500],'g.-');hold on
        scatter(measured_pos(:,1),measured_pos(:,2));hold on
        xlabel('x(m)')
        ylabel('y(m)')
        legend('Real position','Ellipse of 95% probability error contours','Location results','Location','northwest')
        
        [estimated_pos,loss] = PerformanceEnhancementWithMAP(measured_pos,real_pos,sensor,vars_real,Pe);
        [a,idx]=max(loss(:,3));
        % 创建网格
        [X, Y] = meshgrid(unique(loss(:,1)), unique(loss(:,2))); % 创建网格
        Z = griddata(loss(:,1), loss(:,2), loss(:,3), X, Y); % 使用 griddata 插值

        % 绘制三维曲面
        figure; % 创建新图形窗口
        surf(X, Y, Z); % 绘制曲面
        text(loss(idx,1),loss(idx,2),loss(idx,3),['[',num2str(loss(idx,1)),',',num2str(loss(idx,2)),',',num2str(round(loss(idx,3))),']'],'Color','b');
        xlabel('X(km)'); % x 轴标签
        ylabel('Y(km)'); % y 轴标签
        zlabel('Likelihood value'); % z 轴标签
        title('Loss Function'); % 图形标题
        colorbar; % 显示颜色条
        shading interp

        % % 2- 多次实验统计误差提升程度
        MC=100;%蒙特卡洛实验次数
        n=20;    %同一目标的定位点数
        finished = 1; %1表示已完成统计，直接加载即可；0表示重新开始统计
        measured_err = [];
        estimated_err = [];
        real_pos_list = [];
        for i=1:1:MC
            disp(i)
            index = randperm(numel(a), 1);
            real_pos = [x_span(randperm(numel(x_span), 1)),y_span(randperm(numel(y_span), 1)),500];
            vars_real = [18e-3,20e-3,25e-3,0.5,-0.3,0.5,-0.2]; %us us us m
            P = cal_diff_P(real_pos,sensor,vars_real);
            real_pos = real_pos(1:2);
            P = P(1:2,1:2);

            measured_pos = mvnrnd(real_pos, P, n);%观测到的定位结果
            Pe = 0.95;
            [estimated_pos,loss] = PerformanceEnhancementWithMAP(measured_pos,real_pos,sensor,vars_real,Pe);
            estimated_err = [estimated_err;estimated_pos(1)-real_pos(1),estimated_pos(2)-real_pos(2),pdist2(estimated_pos,real_pos)];
            measured_err = [measured_err;mean(pdist2(measured_pos,real_pos)),pdist2(mean(measured_pos),real_pos)];
            % measured_err = [measured_err,mean(pdist2(mean(measured_pos,1),real_pos))];
            real_pos_list = [real_pos_list;real_pos];
        end
        save_path = ['test7_data/n=',num2str(n),'/'];
        save([save_path,'estimated_err.mat'],'estimated_err')
        save([save_path,'measured_err.mat'],'measured_err')
        % figure
        % plot(estimated_err,'r.');hold on
        % plot(measured_err,'b.')
    case 3
        %% case3-置信度：当真实位置已知时：置信度是指相对于真实位置，每个定位结果的可信程度
        emitter_i=[100,100];%真实位置
        [gdop,~,~,P] = cal_gdop(emitter_i,sensor);%计算真实位置的协方差
        n=10;%定位次数
        samples = mvnrnd(emitter_i, P, n);%模拟的定位结果
        gridsize = 0.1;% 定位时所选用的网格大小:0.1km
        plot_trigger = 1; %绘画真实位置处的概率分布函数
        [confidence] = CalConfidence_withCalib(emitter_i,samples,sensor,gridsize,plot_trigger);
       
    case 4
        % case3是真实位置已知的情况，很多情况下要假设真实位置未知，此时需要用定位结果作为真实位置进行推算
        % 所以当真实位置未知时，置信度是指相对于最终定位结果，当前定位值的可信程度
        %% case4-置信度：当真实位置未知时：置信度是指相对于最终定位结果，每个定位结果的可信程度

        % 1-距离d（即GDOP的影响）、信噪比snr（传输信道决定的）到a的映射模型：
        % 1.1- GDOP到权重的映射：k1=exp(-k*gdop), 设置gdop=0时k1=1，gdop=3km时k1=0.5（半衰）
        k = log(0.5)/(-3);
        k1 = @(x) exp(-k.*x);
        % 1.2- snr到权重的映射：k2=1/(1+exp(-k*snr)) # sigmoid函数, 设置snr=10时k2=0.6
        k = log(1/0.6-1)/(-10);
        k2 = @(x) 1./(1+exp(-k*x));


        snr_list = [0,5,10,15,20];
        n_list = [5,10,15,20,25,30];
        f1 = figure(1);
        f2 = figure(2);
        % 将变化过程存成视频文件%%%%%%%%%%%
        vidOut1 = VideoWriter('f1', 'MPEG-4'); %MPEG-4
        vidOut1.FrameRate = 1; %帧率
        open(vidOut1);
        vidOut2 = VideoWriter('f2', 'MPEG-4'); %MPEG-4
        vidOut2.FrameRate = 1; %帧率
        open(vidOut2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % for idx = 1:1:length(snr_list)%%%%%%%%%%%%%%snr_experiment
        for idx = 1:1:length(n_list)%%%%%%%%%%%%%%samples_num_experiment
            % 2-计算权重系数K(n)
            % snr = snr_list(idx); %dB %%%%%%%%%%%%%%snr_experiment
            % n = 10; %定位样本数 %%%%%%%%%%%%%%snr_experiment
            snr = 15; %dB %%%%%%%%%%%%%%samples_num_experiment
            n = n_list(idx); %定位样本数 %%%%%%%%%%%%%%samples_num_experiment
            emitter_i=[100,100]; % 真实位置
            [gdop,~,~,P] = cal_gdop(emitter_i,sensor);%计算真实位置的协方差
            samples = mvnrnd(emitter_i, P, n);%模拟的定位结果
            % load samples_30.mat
            Pe = 0.95;
            [estimated_emitter,loss] = PerformanceEnhancementWithMAP(samples,emitter_i,sensor,Pe);% 综合定位结果估计得到的位置
            for i =1:1:size(samples,1)
                smaple_i = samples(i,:);
                [gdop,~,~,~] = cal_gdop(smaple_i,sensor);
                a(i) = k1(gdop)*k2(snr);
            end
            K = a.^(1/n);
    
            % 3-求解置信度
            gridsize = 0.1;% 定位时所选用的网格大小:0.1km
            plot_trigger = 0;
            % 真实位置已知时的置信度
            [confidence1] = CalConfidence_withCalib(emitter_i,samples,sensor,gridsize,plot_trigger);
            % 用综合定位结果估计得到的位置作为真实位置时的置信度
            [confidence2] = CalConfidence_withCalib(estimated_emitter,samples,sensor,gridsize,plot_trigger);
            confidence2 = confidence2.*K;
    
            % 4-分析与画图
            [f,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
            figure(1)
            clf %先清空下图窗
            fimplicit(f,[emitter_i(1)-3,emitter_i(1)+3,emitter_i(2)-3,emitter_i(2)+3],'r.-');hold on
            plot(emitter_i(1),emitter_i(2),'rp','MarkerSize',8);hold on
            scatter(samples(:,1),samples(:,2),36,confidence1','filled');hold on
            xlabel('x(km)')
            ylabel('y(km)')
            legend('Ellipse of 95% probability error contours','real position','positioning outputs')
            % title(['snr=',num2str(snr)]) %%%%%%%%%%%%%%snr_experiment
            title(['n=',num2str(n)]) %%%%%%%%%%%%%%samples_num_experiment
            [f,~,~,~]=Draw_err_ellipse(P,Pe,emitter_i);
            colorbar; % 显示颜色条
            xlim([98,102]);
            ylim([98,102]);
            f1_frame = getframe(f1); % f.cdata 即为图像数据，获取视频帧
            writeVideo(vidOut1, uint8(f1_frame.cdata)); %写入视频帧

            figure(2)
            clf
            fimplicit(f,[emitter_i(1)-3,emitter_i(1)+3,emitter_i(2)-3,emitter_i(2)+3],'r.-');hold on
            plot(estimated_emitter(1),estimated_emitter(2),'rp','MarkerSize',8);hold on
            scatter(samples(:,1),samples(:,2),36,confidence2','filled');hold on
            xlabel('x(km)')
            ylabel('y(km)')
            legend('Ellipse of 95% probability error contours','processed position','positioning outputs')
            % title(['snr=',num2str(snr)]) %%%%%%%%%%%%%%snr_experiment
            title(['n=',num2str(n)]) %%%%%%%%%%%%%%samples_num_experiment
            colorbar; % 显示颜色条
            xlim([98,102]);
            ylim([98,102]);
            pause(0.05)
            f2_frame = getframe(f2); % f.cdata 即为图像数据
            writeVideo(vidOut2, uint8(f2_frame.cdata));

        end
        close(vidOut1);
        close(vidOut2);

end