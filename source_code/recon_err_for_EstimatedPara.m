function frob_err = recon_err_for_EstimatedPara(cal_para,real_para,scene)
fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;z_span = linspace(10,20,11);
%% 传感器位置
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
x3 = 0;y3 = -L;z3 = 0.3;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];
real_value = real_para;%变量对应的真实值标签，可以看到eta并不是0

if strcmp(scene,'scene1')
    cal_value = [cal_para(1:3),real_value(4),cal_para(4:end)];
elseif strcmp(scene,'scene2')
    cal_value = cal_para;
elseif strcmp(scene,'scene3')
    cal_value = [real_value(1),cal_para];
elseif strcmp(scene,'scene4')
    cal_value = [cal_para(1),cal_para(1),cal_para(1),cal_para(2),real_value(5:end)];
end
for k=1:1:length(z_span)
    for i=1:length(x_span)
        for j=1:length(y_span)
            pos_i = [x_span(i)+0.01,y_span(j)+0.01,z_span(k)+0.001];
            [Gx,Gy,Gz,G] = cal_diff(pos_i,sensor,cal_value);
            [Gx0,Gy0,Gz0,G0] = cal_diff(pos_i,sensor,real_para);
            G3(i,j,k) = G;
            % G3_x(i,j,k) = Gx;
            % G3_y(i,j,k) = Gy;
            % G3_z(i,j,k) = Gz;
            G30(i,j,k) = G0;
            % G30_x(i,j,k) = Gx0;
            % G30_y(i,j,k) = Gy0;
            % G30_z(i,j,k) = Gz0;
        end
    end
end
frob_err = frob(G30-G3);