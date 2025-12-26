clear;
close all;
flight = 4;
dir_pos = 'E:\zzh清华\时差定位\河北UAV\origin_data_20220224\data\flight';
delay12=load([dir_pos,num2str(flight),'\delay12_new']).delay12;
delay13=load([dir_pos,num2str(flight),'\delay13_new']).delay13;
delay14=load([dir_pos,num2str(flight),'\delay14_new']).delay14;
start_num=3;
piece_num=10;
delay12_set = [];delay13_set = [];delay14_set = [];

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

delay = 39; % 测量数据和GPS数据之间的时间延迟

%% 0：对时差作处理得到delay1k_set，并对处理后的时差作定位得到position_set
% plot_en = 1;
% position_set = [];
% for i=start_num:(length(delay12)/10)
%     a1_com=[]; a2_com=[]; a3_com=[];
%     delay12_part = [];delay13_part = [];delay14_part = [];
%     kk=(i-1)*10+1:(i-1)*10+10;
%     if kk(end)<length(delay12)
%         delay12_first =[delay12{(i-1)*10+1:(i-1)*10+10}];
%         delay13_first =[delay13{(i-1)*10+1:(i-1)*10+10}];
%         delay14_first =[delay14{(i-1)*10+1:(i-1)*10+10}];
%         delay12_second = delay_process1(delay12_first,piece_num);
%         delay13_second = delay_process1(delay13_first,piece_num);
%         delay14_second = delay_process1(delay14_first,piece_num);
%         for j=1:1:length(delay12_second)
%             delay12_ij = delay12_second{j};
%             delay13_ij = delay13_second{j};
%             delay14_ij = delay14_second{j};
%             if ~isempty(delay12_second{j}) && ~isempty(delay13_second{j}) && ~isempty(delay14_second{j})
%                 % plot(i,delay13_ij(2:end),'b.'); hold on;
%                 for k1 = 2:1:length(delay12_ij)
%                     for k2 = 2:1:length(delay13_ij)
%                         for k3 = 2:1:length(delay14_ij)
%                             delay12_temp = delay12_ij(k1);
%                             delay13_temp = delay13_ij(k2);
%                             delay14_temp = delay14_ij(k3);
%                             delay12_part = [delay12_part;delay12_temp];
%                             delay13_part = [delay13_part;delay13_temp];
%                             delay14_part = [delay14_part;delay14_temp];
%                             tdoa3=-[delay12_temp*0.3,delay13_temp*0.3,delay14_temp*0.3];
%                             height=st2(1,3);
%                             position1=jiexi4stations1(st3,tdoa3,plot_en,height);
%                             if position1(1)~=0
%                                 position_set = [position_set;i,position1];
%                             end
%                         end
%                     end
%                 end
%             end
%         end
%     end
%     if ~isempty(delay13_part)
%         delay12_set = [delay12_set;position_process(delay12_part,'all2',i)];
%         delay13_set = [delay13_set;position_process(delay13_part,'all2',i)];
%         delay14_set = [delay14_set;position_process(delay14_part,'all2',i)];
%     else
%         delay12_set = [delay12_set;[i,0]];
%         delay13_set = [delay13_set;[i,0]];
%         delay14_set = [delay14_set;[i,0]];
%     end
% end
% delay12_set = unique(delay12_set,'rows');
% delay13_set = unique(delay13_set,'rows');
% delay14_set = unique(delay14_set,'rows');
% position_set = unique(position_set,'rows');
% figure
% scatter(delay12_set(:,1),delay12_set(:,2))
% figure
% scatter(delay13_set(:,1),delay13_set(:,2))
% figure
% scatter(delay14_set(:,1),delay14_set(:,2))

delay12_list = load('test9_data\1Meas_data\delay12_set.mat').delay12_set;
delay13_list = load('test9_data\1Meas_data\delay13_set.mat').delay13_set;
delay14_list = load('test9_data\1Meas_data\delay14_set.mat').delay14_set;
pos_list = load('test9_data\1Meas_data\position_set.mat').position_set;


%% 1：处理飞行记录数据
% text1=[dir_pos,num2str(flight),'\',num2str(flight),'.xlsx'];
% auto1=importdata(text1);
% auto2=auto1.data;%位置信息
% auto3=auto1.textdata;%时间信息
% time1=auto2(:,1)/1000;
%
% % 记录时间
% time2=auto3(2:end,2);
% time2=char(time2);
% time2_0=time2(:,end-7:end);
% time2_1=str2num(time2_0(:,end-1:end));
% time2_2=str2num(time2_0(:,end-4:end-3));
% time2_3=str2num(time2_0(:,end-7:end-6));
% char1=time2_1(1);
% second1=[1];%存放字符串矩阵不同字符出现位置，这里用unique函数可以直接获取
% for k1=1:length(time2_1)
%     if time2_1(k1)~=char1
%         second1=[second1;k1];
%         char1=time2_1(k1);
%     end
% end
%
% longitude1=zeros(length(second1),1);
% latitude1=zeros(length(second1),1);
% height1=zeros(length(second1),1);
% time_s=zeros(length(second1),1);
% time_m=zeros(length(second1),1);
% time_h=zeros(length(second1),1);
%
% for k2=1:length(second1)
%     longitude1(k2)=auto2(second1(k2),3);
%     latitude1(k2)=auto2(second1(k2),4);
%     height1(k2)=auto2(second1(k2),5);
%     time_h(k2)=time2_3(second1(k2));
%     time_m(k2)=time2_2(second1(k2));
%     time_s(k2)=time2_1(second1(k2));
% end
% dronedata=zeros(length(second1),8);
% % dronedata数据格式分别是 1.时 2.分 3.秒 4.纬度 5.经度 6高度 7.UAV到主楼的距离 8.UAV高度
% for k3=1:length(second1)
%     dronedata(k3,1)= time_h(k3);
%     dronedata(k3,2)= time_m(k3);
%     dronedata(k3,3)= time_s(k3);
%     dronedata(k3,4)=longitude1(k3);
%     dronedata(k3,5)= latitude1(k3);
%     dronedata(k3,6)= height1(k3)*0.3048;%原单位是feet，变成米
% end
%
% % txt_res = [];
% % for i=1:1:size(dronedata,1)
% %     txt_res_part = [0,0,dronedata(i,5),dronedata(i,4),dronedata(i,6),1000,1000];
% %     disp_content = [num2str(0),',',num2str(0),',',...
% %         num2str(dronedata(i,5)),',', num2str(dronedata(k3,4)),',', num2str(dronedata(k3,6)),',',...
% %         num2str(1000),',',num2str(1000)];
% %     disp(disp_content);
% %     txt_res = [txt_res;txt_res_part];
% % end
% % txt_res = txt_res(3:end,:);
% % dlmwrite('D:\GisAnalyse\data\trace.txt',txt_res,'precision','%.8f','delimiter',',');
% % dlmwrite('D:\GisAnalyse\data\stations.txt',[zeros(4,2),station_positions1,1000*ones(4,2)],'precision','%.8f','delimiter',',');
%
% % writematrix(txt_res,'D:\GisAnalyse\data\trace.txt','Delimiter',',')
% b1_matrix=[];
% for pp=1:length(dronedata(:,1))
%     %       b1=[dronedata(pp,5),dronedata(pp,4),dronedata(pp,6)+100.51-1];
%     b1=[dronedata(pp,5),dronedata(pp,4),dronedata(pp,6)+160];
%     X0Y0Z0=[-2059888.16201274,4545478.00190905,3958977.77427503];
%     L0B0H0=[114.378753,38.612907,236];
%     LBH=b1;
%     b1_matrix=[b1_matrix;LBH_xyz(L0B0H0,X0Y0Z0,LBH)];
% end
%
% %用来存储每一秒TDOA的矩阵
% delay12_1=zeros(1,length(dronedata(:,1)));
% delay13_1=zeros(1,length(dronedata(:,1)));
% delay14_1=zeros(1,length(dronedata(:,1)));
% delay23_1=zeros(1,length(dronedata(:,1)));
% delay24_1=zeros(1,length(dronedata(:,1)));
% delay34_1=zeros(1,length(dronedata(:,1)));
%
% for k3=1:length(dronedata(:,1))
%
%     %      a1=[dronedata(k3,5),dronedata(k3,4),dronedata(k3,6)+100.51-1];
%     %       a1=[dronedata(pp,5),dronedata(pp,4),dronedata(pp,6)];
%     %      %经纬高转xyz
%     %       X0Y0Z0=[-2059888.16201274,4545478.00190905,3958977.77427503];
%     %       L0B0H0=[114.378753,38.612907,236];
%     %       LBH=a1;
%     %       a1=LBH_xyz(L0B0H0,X0Y0Z0,LBH);
%     delay12_1(k3)=(sqrt((b1_matrix(k3,:)-st2(1,:))*(b1_matrix(k3,:)-st2(1,:))')-sqrt((b1_matrix(k3,:)-st2(2,:))*(b1_matrix(k3,:)-st2(2,:))'))/0.3;
%     delay13_1(k3)=(sqrt((b1_matrix(k3,:)-st2(1,:))*(b1_matrix(k3,:)-st2(1,:))')-sqrt((b1_matrix(k3,:)-st2(3,:))*(b1_matrix(k3,:)-st2(3,:))'))/0.3;
%     delay14_1(k3)=(sqrt((b1_matrix(k3,:)-st2(1,:))*(b1_matrix(k3,:)-st2(1,:))')-sqrt((b1_matrix(k3,:)-st2(4,:))*(b1_matrix(k3,:)-st2(4,:))'))/0.3;
%
%     dronedata(k3,7)=sqrt((b1_matrix(k3,:)-st2(1,:))*(b1_matrix(k3,:)-st2(1,:))');
%     dronedata(k3,8)=b1_matrix(k3,3);
%     plot3(b1_matrix(k3,1),b1_matrix(k3,2),b1_matrix(k3,3),'r.'); hold on;
% end
delay12_GPS = load('test9_data\2GPS_data\delay12_1.mat').delay12_1;
delay13_GPS = load('test9_data\2GPS_data\delay13_1.mat').delay13_1;
delay14_GPS = load('test9_data\2GPS_data\delay14_1.mat').delay14_1;
delay12_GPS(1:2) = delay12_GPS(3);delay13_GPS(1:2) = delay13_GPS(3);delay14_GPS(1:2) = delay14_GPS(3);
pos_GPS = load('test9_data\2GPS_data\GPS_pos.mat').b1_matrix;
pos_GPS(1,:) = pos_GPS(3,:);pos_GPS(2,:) = pos_GPS(3,:);

% figure
% plot(1+delay:1:delay+length(delay12_GPS),delay12_GPS,'b*');hold on
% plot(delay12_list(:,1),delay12_list(:,2),'r*');
% figure
% plot(1+delay:1:delay+length(delay13_GPS),delay13_GPS,'b*');hold on
% plot(delay13_list(:,1),delay13_list(:,2),'r*');
% figure
% plot(1+delay:1:delay+length(delay14_GPS),delay14_GPS,'b*');hold on
% plot(delay14_list(:,1),delay14_list(:,2),'r*');

%% 3：根据GPS记载的数据，对时差作进一步分选，选出干净的，并定位
% delay12_list2 = [];
% delay13_list2 = [];
% delay14_list2 = [];
% % 3-1时差分选
% for i = 1:1:length(delay12_GPS)
%     idx = i + delay;
%     delay12_list_temp = delay12_list(delay12_list(:,1)==idx,2);
%     delay13_list_temp = delay13_list(delay13_list(:,1)==idx,2);
%     delay14_list_temp = delay14_list(delay14_list(:,1)==idx,2);
%     if ~isempty(delay12_list_temp)
%         loss = dist(delay12_list_temp,delay12_GPS(i));
%         [min_value,min_idx]=min(loss);
%         if min_value<100
%             delay12_list2 = [delay12_list2;idx,delay12_list_temp(min_idx)];
%         end
%     end
%     if ~isempty(delay13_list_temp)
%         loss = dist(delay13_list_temp,delay13_GPS(i));
%         [min_value,min_idx]=min(loss);
%         if min_value<100
%             delay13_list2 = [delay13_list2;idx,delay13_list_temp(min_idx)];
%         end
%     end
%     if ~isempty(delay14_list_temp)
%         loss = dist(delay14_list_temp,delay14_GPS(i));
%         [min_value,min_idx]=min(loss);
%         if min_value<100
%             delay14_list2 = [delay14_list2;idx,delay14_list_temp(min_idx)];
%         end
%     end
% end
% % 3-2定位
% start_idx = delay+1;
% end_idx = max([delay12_list2(:,1);delay13_list2(:,1);delay14_list2(:,1)]);
% pos_list2 = [];
% for i = start_idx:1:end_idx
%     delay12_temp = delay12_list2(delay12_list2(:,1)==i,2);
%     delay13_temp = delay13_list2(delay13_list2(:,1)==i,2);
%     delay14_temp = delay14_list2(delay14_list2(:,1)==i,2);
%     if ~isempty(delay12_temp) & ~isempty(delay13_temp) & ~isempty(delay14_temp)
%         tdoa3=-[delay12_temp*0.3,delay13_temp*0.3,delay14_temp*0.3];
%         height=st2(1,3);
%         position1=jiexi4stations1(st3,tdoa3,0,height);
%         pos_list2 = [pos_list2;i,position1];
%     end
% end

delay12_list2 = load('test9_data\3Selected_data\delay12_list2.mat').delay12_list2;
delay13_list2 = load('test9_data\3Selected_data\delay13_list2.mat').delay13_list2;
delay14_list2 = load('test9_data\3Selected_data\delay14_list2.mat').delay14_list2;
pos_list2 = load('test9_data\3Selected_data\pos_list2.mat').pos_list2;


% figure
% plot(1+delay:1:delay+length(delay12_GPS),delay12_GPS,'b*');hold on
% plot(delay12_list2(:,1),delay12_list2(:,2),'r*');
% figure
% plot(1+delay:1:delay+length(delay13_GPS),delay13_GPS,'b*');hold on
% plot(delay13_list2(:,1),delay13_list2(:,2),'r*');
% figure
% plot(1+delay:1:delay+length(delay14_GPS),delay14_GPS,'b*');hold on
% plot(delay14_list2(:,1),delay14_list2(:,2),'r*');
% figure
% plot3(st2(:,1),st2(:,2),st2(:,3),'rp','MarkerSize',10,'MarkerFaceColor','r'); hold on;
% plot3(pos_GPS(:,1),pos_GPS(:,2),pos_GPS(:,3),'b*');hold on
% plot3(pos_list2(:,2),pos_list2(:,3),pos_list2(:,4),'g*')

%% 4：误差分析（时差误差+定位误差）
% start_idx = delay+1;
% end_idx = max([delay12_list2(:,1);delay13_list2(:,1);delay14_list2(:,1)]);
% delay12_err = struct('index',[],'cal_TDOA',[],'real_TDOA',[],'err',[]);
% delay13_err = struct('index',[],'cal_TDOA',[],'real_TDOA',[],'err',[]);
% delay14_err = struct('index',[],'cal_TDOA',[],'real_TDOA',[],'err',[]);
% pos_err = struct('index',[],'cal_pos',[],'real_pos',[],'err',[]);
% for i=start_idx:1:end_idx
%     delay12_temp = delay12_list2(delay12_list2(:,1)==i,2);
%     delay13_temp = delay13_list2(delay13_list2(:,1)==i,2);
%     delay14_temp = delay14_list2(delay14_list2(:,1)==i,2);
%     pos_temp = pos_list2(pos_list2(:,1)==i,2:4);
%     delay12_true = delay12_GPS(i-delay);
%     delay13_true = delay13_GPS(i-delay);
%     delay14_true = delay14_GPS(i-delay);
%     pos_true = pos_GPS(i-delay,:);
%     if ~isempty(delay12_temp)
%         delay12_err.index = [delay12_err.index;i];
%         delay12_err.cal_TDOA = [delay12_err.cal_TDOA;delay12_temp];
%         delay12_err.real_TDOA = [delay12_err.real_TDOA;delay12_true];
%         delay12_err.err = [delay12_err.err;delay12_temp - delay12_true];
%     end
%     if ~isempty(delay13_temp)
%         delay13_err.index = [delay13_err.index;i];
%         delay13_err.cal_TDOA = [delay13_err.cal_TDOA;delay13_temp];
%         delay13_err.real_TDOA = [delay13_err.real_TDOA;delay13_true];
%         delay13_err.err = [delay13_err.err;delay13_temp - delay13_true];
%     end
%     if ~isempty(delay14_temp)
%         delay14_err.index = [delay14_err.index;i];
%         delay14_err.cal_TDOA = [delay14_err.cal_TDOA;delay14_temp];
%         delay14_err.real_TDOA = [delay14_err.real_TDOA;delay14_true];
%         delay14_err.err = [delay14_err.err;delay14_temp - delay14_true];
%     end
%     if ~isempty(pos_temp)
%         pos_err.index = [pos_err.index;i]; % idx, dx, dy, dz, dr
%         pos_err.cal_pos = [pos_err.cal_pos;pos_temp];
%         pos_err.real_pos = [pos_err.real_pos;pos_true];
%         pos_err.err = [pos_err.err;pos_temp-pos_true,norm(pos_temp-pos_true)];
%     end
% end
% test_data_index = randperm(size(pos_err.real_pos,1),floor(size(pos_err.real_pos,1)*0.2));
% train_data_index = setdiff(1:size(pos_err.real_pos,1), test_data_index);
% pos_err_train.index = pos_err.index(train_data_index);
% pos_err_train.cal_pos = pos_err.cal_pos(train_data_index,:);
% pos_err_train.real_pos = pos_err.real_pos(train_data_index,:);
% pos_err_train.err = pos_err.err(train_data_index,:);
% pos_err_test.index = pos_err.index(test_data_index);
% pos_err_test.cal_pos = pos_err.cal_pos(test_data_index,:);
% pos_err_test.real_pos = pos_err.real_pos(test_data_index,:);
% pos_err_test.err = pos_err.err(test_data_index,:);

delay12_err = load('test9_data\4Err_data\delay12_err.mat').delay12_err;
delay13_err = load('test9_data\4Err_data\delay13_err.mat').delay13_err;
delay14_err = load('test9_data\4Err_data\delay14_err.mat').delay14_err;
pos_err = load('test9_data\4Err_data\pos_err.mat').pos_err;
% pos_err = load('test9_data\4Err_data\pos_err_train.mat').pos_err_train;

mean12 = mean(delay12_err.err)
sigma12 = sqrt(cov(delay12_err.err))
mean13 = mean(delay13_err.err)
sigma13 = sqrt(cov(delay13_err.err)) %单位ns
mean14 = mean(delay14_err.err)
sigma14 = sqrt(cov(delay14_err.err))
eta12 = corrcoef(delay12_err.cal_TDOA(find(ismember(delay12_err.index, pos_err.index))),...
    delay13_err.cal_TDOA(find(ismember(delay13_err.index, pos_err.index))))
eta13 = corrcoef(delay12_err.cal_TDOA(find(ismember(delay12_err.index, pos_err.index))),...
    delay14_err.cal_TDOA(find(ismember(delay14_err.index, pos_err.index))))
eta23 = corrcoef(delay13_err.cal_TDOA(find(ismember(delay13_err.index, pos_err.index))),...
    delay14_err.cal_TDOA(find(ismember(delay14_err.index, pos_err.index))))


Err_XY = vecnorm(pos_err.real_pos(:,1:2)-pos_err.cal_pos(:,1:2),2,2); %第1个参数2表示2范数，第2个表示逐行求范数
Err_XY = Err_XY.^2*1e-6; %单位km
Err_XYZ = pos_err.err(:,end).^2*1e-6;
Err_XY_GDOP = [];
Err_XYZ_GDOP = [];

% 先假设几个参数值（包括时差误差的观测量），依此从所有观测到的误差中筛选出一部分基本符合GDOP分布规律的数据
var_value = [sigma12*1e-3,sigma13*1e-3,sigma14*1e-3,1*1e-3,0,0,0];%变量列表，eta初始化为0，eta必须取常数而不是变量
for i=1:1:size(pos_err.real_pos,1)
    pos_i = pos_err.real_pos(i,:);
    [Gx,Gy,Gz,G] = cal_diff(pos_i*1e-3,st3*1e-3,var_value);
    Err_XY_GDOP = [Err_XY_GDOP;Gx+Gy];
    Err_XYZ_GDOP = [Err_XYZ_GDOP;G];
end
figure
plot3(st2(:,1)*1e-3,st2(:,2)*1e-3,zeros(4,1),'rp','MarkerSize',10,'MarkerFaceColor','r'); hold on;
plot3(pos_err.real_pos(:,1),pos_err.real_pos(:,2),Err_XY(:),'g.');hold on
plot3(pos_err.real_pos(:,1),pos_err.real_pos(:,2),Err_XY_GDOP(:),'b.')
legend('mea','real')
selected_pos = [];
selected_Err = [];
selected_Err_GDOP = [];
for i=1:1:size(Err_XY_GDOP,1)
    if abs(Err_XY_GDOP(i)-Err_XY(i))<0.02e-3 %0.02m
        selected_pos = [selected_pos;pos_err.real_pos(i,:)];
        selected_Err = [selected_Err;Err_XY(i,:)];
        selected_Err_GDOP = [selected_Err_GDOP;Err_XY_GDOP(i,:)];
    end
end
% for i=1:1:size(Err_XYZ_GDOP,1)
%     if abs(Err_XY_GDOP(i)-Err_XYZ(i))<0.2e-3 %0.02m
%         selected_pos = [selected_pos;pos_err.real_pos(i,:)];
%         selected_Err = [selected_Err;Err_XYZ(i,:)];
%         selected_Err_GDOP = [selected_Err_GDOP;Err_XYZ_GDOP(i,:)];
%     end
% end

%% 5:参数估计
sigmat1_initial = sigma12*1e-3;%单位us
sigmat2_initial = sigma13*1e-3;%单位us
sigmat3_initial = sigma14*1e-3;%单位us
syms sigmat1 sigmat2 sigmat3
syms sigmas
syms eta12 eta13 eta23
var_value = [sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23];%变量列表，eta初始化为0，eta必须取常数而不是变量
constant = [];
coeffs = [];
b = [];
for i=1:1:size(selected_pos,1)
    pos_i = selected_pos(i,:);
    % 计算G关于各变量的偏微分以及G_cal关于变量的表达式，是真实表达式
    [Gx_eq,Gy_eq,Gz_eq,G_eq] = cal_diff(pos_i*1e-3,st3*1e-3,var_value);%时间单位us，距离单位km
    G_mea = selected_Err(i,:);%实际测量值
    % 在表达式中代入变量和预设的已知量，得到计算的G的表达式
    G_Cal = subs(Gx_eq+Gy_eq, sigmat2,sigmat2_initial);
    % G_Cal = subs(G_eq, sigmat2,sigmat2_initial);
    var_name = [sigmat1,sigmat3,sigmas,eta12,eta13,eta23];%G的表达式中有哪些多项式的项
    constant = [constant;double(subs(G_Cal, var_name,zeros(1,length(var_name))))];%求常数项式系数
    var_name = [sigmat1^2,sigmat3^2,sigmas^2,eta12*sigmat1,eta13*sigmat1*sigmat3,eta23*sigmat3];%G的表达式中有哪些多项式的项

    % G_Cal = subs(Gx_eq+Gy_eq, [sigmat1,sigmat2,sigmat3],[sigmat1_initial,sigmat2_initial,sigmat3_initial]);
    % var_name = [sigmas,eta12,eta13,eta23];%G的表达式中有哪些多项式的项
    % constant = [constant;double(subs(G_Cal, var_name,zeros(1,length(var_name))))];%求常数项式系数
    % var_name = [sigmas^2,eta12,eta13,eta23];%G的表达式中有哪些多项式的项

    coeff = double(expr_coeff(G_Cal, var_name));
    coeffs =  [coeffs;coeff];
    b = [b;G_mea];

end
res1 = pinv(coeffs)*(b-constant);

res1 = load ('test9_data\5Para_esti\res1.mat').res1;
sigmt1_res = sqrt(res1(1));
sigmt3_res = sqrt(res1(2));
sigms_res = abs(sqrt(res1(3)));
eta12_res = res1(4)*0.01/sigmt1_res/sigmat2_initial;
eta13_res = res1(5)/sigmt1_res/sigmt3_res;
eta23_res = res1(6)*0.1/sigmat2_initial/sigmt3_res;

% sigms_res = abs(sqrt(res1(1)));
% eta12_res = res1(2)/sigmat1_initial/sigmat2_initial;
% eta13_res = res1(3)/sigmat1_initial/sigmat3_initial;
% eta23_res = res1(4)/sigmat2_initial/sigmat3_initial;

%% 6：张量重建
x_span = linspace(min(pos_list2(:,2)),max(pos_list2(:,2)),200);
x_span = linspace(min(pos_list2(:,2)),1000,200);
y_span = linspace(min(pos_list2(:,3)),max(pos_list2(:,3)),200);
y_span = linspace(min(pos_list2(:,3)),1000,200);
z_span = linspace(min(pos_list2(:,4)),max(pos_list2(:,4)),10);
vars = [sigmt1_res,sigmat2_initial,sigmt3_res,sigms_res,eta12_res,eta13_res,eta23_res];
% for k = 1:1:length(z_span)
%     for i = 1:1:length(x_span)
%         for j = 1:1:length(y_span)
%             pos_i = [x_span(i),y_span(j),z_span(k)];
%             [Gx_eq,Gy_eq,Gz_eq,G_eq] = cal_diff(pos_i*1e-3,st3*1e-3,vars);%时间单位us，距离单位km
%             G_cal(i,j,k) = G_eq;
%             Gx_cal(i,j,k) = Gx_eq;
%             Gy_cal(i,j,k) = Gy_eq;
%             Gz_cal(i,j,k) = Gz_eq;
%         end
%     end
% end
% pos_i_real = pos_err.real_pos(185,:);
% pos_i_TDOA = pos_err.cal_pos(185,:);
% [Gx_eq,Gy_eq,Gz_eq,G_eq] = cal_diff(pos_i_real*1e-3,st3*1e-3,vars);%时间单位us，距离单位km
% sqrt([Gx_eq,Gy_eq,Gz_eq])*1e3
% pos_i_real - pos_i_TDOA

G_cal = load ('test9_data\6ReconTensor\G_cal.mat').G_cal;
Gx_cal = load ('test9_data\6ReconTensor\Gx_cal.mat').Gx_cal;
Gy_cal = load ('test9_data\6ReconTensor\Gy_cal.mat').Gy_cal;
Gz_cal = load ('test9_data\6ReconTensor\Gz_cal.mat').Gz_cal;

% figure();
% [c,handle]=contour(x_span*1e-3,y_span*1e-3,G_cal(:,:,1)',100); hold on
% clabel(c,handle,'manual') % 手动标签，摁enter退出
% % v = [0.01,0.02,0.0522621,0.134494,0.267897392624881,0.622787];
% % clabel(c,handle,v);
% plot(pos_err.real_pos(:,1)*1e-3,pos_err.real_pos(:,2)*1e-3,'.-'); hold on
% plot(pos_err.cal_pos(:,1)*1e-3,pos_err.cal_pos(:,2)*1e-3,'.-'); hold on
% plot(st2(:,1)*1e-3,st2(:,2)*1e-3,'rp','MarkerSize',10,'MarkerFaceColor','r'); hold on;
% legend('contour','GPS trajectory','Positioning trajectory','stations')
% title('Reconstructed distribution of $G$','interpreter','latex');
% xlabel('x(km)');
% ylabel('y(km)');

figure();
[c,handle]=contour(x_span*1e-3,y_span*1e-3,Gx_cal(:,:,1)',100); hold on
clabel(c,handle,'manual') % 手动标签，摁enter退出
% v = [0.01,0.02,0.0522621,0.134494,0.267897392624881,0.622787];
% clabel(c,handle,v);
plot(pos_err.real_pos(:,1)*1e-3,pos_err.real_pos(:,2)*1e-3,'.-'); hold on
plot(pos_err.cal_pos(:,1)*1e-3,pos_err.cal_pos(:,2)*1e-3,'.-'); hold on
plot(st2(:,1)*1e-3,st2(:,2)*1e-3,'rp','MarkerSize',10,'MarkerFaceColor','r'); hold on;
legend('contour','GPS trajectory','Positioning trajectory','stations')
title('Reconstructed distribution of $G_y$','interpreter','latex');
xlabel('x(km)');
ylabel('y(km)');



