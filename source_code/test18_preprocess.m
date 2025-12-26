clear;
close all;
% 该脚本用于从原始定位结果和GPS结果中处理提取四组flight的时差、定位误差
flight = 3;
dir_pos = ['E:\zzh清华\时差定位\河北UAV\origin_data_20220224\data\flight',num2str(flight)];
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



%% 0：对时差作处理得到delay1k_set，并对处理后的时差作定位得到position_set

delay12_list = load([dir_pos,'/delay12_processed.mat']).delay12_set;
delay13_list = load([dir_pos,'/delay13_processed.mat']).delay13_set;
delay14_list = load([dir_pos,'/delay14_processed.mat']).delay14_set;


%% 1：处理飞行记录数据
delay12_GPS = load([dir_pos,'/delay12_GPS.mat']).delay12_1;
delay13_GPS = load([dir_pos,'/delay13_GPS.mat']).delay13_1;
delay14_GPS = load([dir_pos,'/delay14_GPS.mat']).delay14_1;
pos_GPS = load([dir_pos,'/pos_GPS.mat']).b1_matrix;

% delay12_GPS(1:2) = delay12_GPS(3);delay13_GPS(1:2) = delay13_GPS(3);delay14_GPS(1:2) = delay14_GPS(3);
% pos_GPS(1,:) = pos_GPS(3,:);pos_GPS(2,:) = pos_GPS(3,:);

% Delay计算
delay = 83; % 测量数据和GPS数据之间的时间延迟 flight1:41  flight2:16  flight3:83  flight4:39
figure
plot(1+delay:1:delay+length(delay12_GPS),delay12_GPS,'b*');hold on
plot(delay12_list(:,1),delay12_list(:,2),'r*');
figure
plot(1+delay:1:delay+length(delay13_GPS),delay13_GPS,'b*');hold on
plot(delay13_list(:,1),delay13_list(:,2),'r*');
figure
plot(1+delay:1:delay+length(delay14_GPS),delay14_GPS,'b*');hold on
plot(delay14_list(:,1),delay14_list(:,2),'r*');

%% 3：根据GPS记载的数据，对时差作进一步分选，选出干净的，并定位
delay12_list2 = [];
delay13_list2 = [];
delay14_list2 = [];
% 3-1时差分选
for i = 1:1:length(delay12_GPS)
    idx = i + delay;
    delay12_list_temp = delay12_list(delay12_list(:,1)==idx,2);
    delay13_list_temp = delay13_list(delay13_list(:,1)==idx,2);
    delay14_list_temp = delay14_list(delay14_list(:,1)==idx,2);
    if ~isempty(delay12_list_temp)
        loss = dist(delay12_list_temp,delay12_GPS(i));
        [min_value,min_idx]=min(loss);
        if min_value<100
            delay12_list2 = [delay12_list2;idx,delay12_list_temp(min_idx)];
        end
    end
    if ~isempty(delay13_list_temp)
        loss = dist(delay13_list_temp,delay13_GPS(i));
        [min_value,min_idx]=min(loss);
        if min_value<100
            delay13_list2 = [delay13_list2;idx,delay13_list_temp(min_idx)];
        end
    end
    if ~isempty(delay14_list_temp)
        loss = dist(delay14_list_temp,delay14_GPS(i));
        [min_value,min_idx]=min(loss);
        if min_value<100
            delay14_list2 = [delay14_list2;idx,delay14_list_temp(min_idx)];
        end
    end
end
% 3-2定位
start_idx = delay+1;
end_idx = max([delay12_list2(:,1);delay13_list2(:,1);delay14_list2(:,1)]);
pos_list2 = [];
figure
for i = start_idx:1:end_idx
    delay12_temp = delay12_list2(delay12_list2(:,1)==i,2);
    delay13_temp = delay13_list2(delay13_list2(:,1)==i,2);
    delay14_temp = delay14_list2(delay14_list2(:,1)==i,2);
    if ~isempty(delay12_temp) & ~isempty(delay13_temp) & ~isempty(delay14_temp)
        tdoa3=-[delay12_temp*0.3,delay13_temp*0.3,delay14_temp*0.3];
        height=st2(1,3);
        position1=jiexi4stations1(st3,tdoa3,0,height);
        pos_list2 = [pos_list2;i,position1];
    end
end

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
start_idx = delay+1;
end_idx = max([delay12_list2(:,1);delay13_list2(:,1);delay14_list2(:,1)]);
delay12_err = struct('index',[],'cal_TDOA',[],'real_TDOA',[],'err',[]);
delay13_err = struct('index',[],'cal_TDOA',[],'real_TDOA',[],'err',[]);
delay14_err = struct('index',[],'cal_TDOA',[],'real_TDOA',[],'err',[]);
pos_err = struct('index',[],'cal_pos',[],'real_pos',[],'err',[]);
for i=start_idx:1:end_idx
    delay12_temp = delay12_list2(delay12_list2(:,1)==i,2);
    delay13_temp = delay13_list2(delay13_list2(:,1)==i,2);
    delay14_temp = delay14_list2(delay14_list2(:,1)==i,2);
    pos_temp = pos_list2(pos_list2(:,1)==i,2:4);
    delay12_true = delay12_GPS(i-delay);
    delay13_true = delay13_GPS(i-delay);
    delay14_true = delay14_GPS(i-delay);
    pos_true = pos_GPS(i-delay,:);
    if ~isempty(delay12_temp)
        delay12_err.index = [delay12_err.index;i];
        delay12_err.cal_TDOA = [delay12_err.cal_TDOA;delay12_temp];
        delay12_err.real_TDOA = [delay12_err.real_TDOA;delay12_true];
        delay12_err.err = [delay12_err.err;delay12_temp - delay12_true];
    end
    if ~isempty(delay13_temp)
        delay13_err.index = [delay13_err.index;i];
        delay13_err.cal_TDOA = [delay13_err.cal_TDOA;delay13_temp];
        delay13_err.real_TDOA = [delay13_err.real_TDOA;delay13_true];
        delay13_err.err = [delay13_err.err;delay13_temp - delay13_true];
    end
    if ~isempty(delay14_temp)
        delay14_err.index = [delay14_err.index;i];
        delay14_err.cal_TDOA = [delay14_err.cal_TDOA;delay14_temp];
        delay14_err.real_TDOA = [delay14_err.real_TDOA;delay14_true];
        delay14_err.err = [delay14_err.err;delay14_temp - delay14_true];
    end
    if ~isempty(pos_temp)
        pos_err.index = [pos_err.index;i]; % idx, dx, dy, dz, dr
        pos_err.cal_pos = [pos_err.cal_pos;pos_temp];
        pos_err.real_pos = [pos_err.real_pos;pos_true];
        pos_err.err = [pos_err.err;pos_temp-pos_true,norm(pos_temp-pos_true)];
    end
end
test_data_index = randperm(size(pos_err.real_pos,1),floor(size(pos_err.real_pos,1)*0.2));
train_data_index = setdiff(1:size(pos_err.real_pos,1), test_data_index);
pos_err_train.index = pos_err.index(train_data_index);
pos_err_train.cal_pos = pos_err.cal_pos(train_data_index,:);
pos_err_train.real_pos = pos_err.real_pos(train_data_index,:);
pos_err_train.err = pos_err.err(train_data_index,:);
pos_err_test.index = pos_err.index(test_data_index);
pos_err_test.cal_pos = pos_err.cal_pos(test_data_index,:);
pos_err_test.real_pos = pos_err.real_pos(test_data_index,:);
pos_err_test.err = pos_err.err(test_data_index,:);

save(['test18_data/flight',num2str(flight),'\delay12_err.mat'],'delay12_err');
save(['test18_data/flight',num2str(flight),'\delay13_err.mat'],'delay13_err');
save(['test18_data/flight',num2str(flight),'\delay14_err.mat'],'delay14_err');
save(['test18_data/flight',num2str(flight),'\pos_err.mat'],'pos_err');


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


