function plot_3D_realData_track(real_data,stations,x_span,y_span,z_span)
real_pos_index_list = [];
cal_pos_index_list = [];
station_index_list = [];
for i=1:1:size(real_data.real_pos,1)
    pos_i = real_data.real_pos(i,:);
    x_index = transform_value_to_index(x_span,pos_i(1));
    y_index = transform_value_to_index(y_span,pos_i(2));
    z_index = transform_value_to_index(z_span,pos_i(3));
    real_pos_index_list = [real_pos_index_list;x_index,y_index,z_index];
    pos_i = real_data.cal_pos(i,:);
    x_index = transform_value_to_index(x_span,pos_i(1));
    y_index = transform_value_to_index(y_span,pos_i(2));
    z_index = transform_value_to_index(z_span,pos_i(3));
    cal_pos_index_list = [cal_pos_index_list;x_index,y_index,z_index];
end
for i=1:1:size(stations,1)
    pos_i = stations(i,:);
    x_index = transform_value_to_index(x_span,pos_i(1));
    y_index = transform_value_to_index(y_span,pos_i(2));
    z_index = transform_value_to_index(z_span,pos_i(3));
    station_index_list = [station_index_list;x_index,y_index,z_index];
end
plot3(real_pos_index_list(:,1),real_pos_index_list(:,2),real_pos_index_list(:,3),'r.'); hold on
plot3(cal_pos_index_list(:,1),cal_pos_index_list(:,2),cal_pos_index_list(:,3),'b.'); hold on
plot3(station_index_list(:,1),station_index_list(:,2),station_index_list(:,3),'kp','MarkerSize',10,'MarkerFaceColor','r'); hold on;