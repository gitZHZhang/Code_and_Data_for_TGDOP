function [Gx,Gy,Gz,G] = cal_gdop_doa_for_test13(emitter,sensor,value,term)
% value表示测角误差value.sigma_theta1(N个)+value.sigma_theta2(N个)+站址误差value.sigma_s(1个)
% theta1 俯仰角 theta2 方位角

x = emitter(1);y = emitter(2);z = emitter(3);
N = size(sensor,1);
F = zeros(2*N,3);
K = zeros(2*N,1); 
% FF = zeros(2*N,3);
for i=1:1:N
    x_i = sensor(i,1);
    y_i = sensor(i,2);
    z_i = sensor(i,3);
    ri_xy = sqrt((x-x_i)^2+(y-y_i)^2);
    ri_xyz = sqrt((x-x_i)^2+(y-y_i)^2+(z-z_i)^2);
    p_theta1_px = (z-z_i)*(x_i-x)/ri_xyz^2/ri_xy;
    p_theta1_py = (z-z_i)*(y_i-y)/ri_xyz^2/ri_xy;
    p_theta1_pz = ri_xy/ri_xyz^2;
    F(i,:) = [p_theta1_px,p_theta1_py,p_theta1_pz];
    % FF(i,:) = -[p_theta1_px,p_theta1_py,p_theta1_pz];
    p_theta2_px = (y_i-y)/ri_xy^2;
    p_theta2_py = (x-x_i)/ri_xy^2;
    F(i+N,:) = [p_theta2_px,p_theta2_py,0];
    K(i,:) = F(i,:)*diag(ones(1,3).*value.sigma_s^2)*F(i,:)'; % 1/ri_xyz^2为sigma_s的系数
    K(i+N,:) = F(i+N,:)*diag(ones(1,3).*value.sigma_s^2)*F(i+N,:)'; % 1/ri_xy^2为sigma_s的系数
    % FF(i+N,:) = -[p_theta2_px,p_theta2_py,0];
end
term1 = diag([value.sigma_theta1,value.sigma_theta2].^2);
term2 = diag(K);
pinv_F = pinv(F);
P = pinv_F*(term)*pinv_F';
gdop = sqrt(trace(P));
G = trace(P);
Gx = P(1,1);
Gy = P(2,2);
Gz = P(3,3);