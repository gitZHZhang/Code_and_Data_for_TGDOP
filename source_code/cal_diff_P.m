function [P] = cal_diff_P(emitter,sensor,value)
% function [res_x,res_y,res_z,G] = cal_diff(emitter,sensor,value)
%input value=[sigmat1 sigmat2 sigmat3,sigmas,eta12 eta13 eta23];sigmas单位m，sigmat单位us
%output res_x = p_sigmax2_p[sigmat1 sigmat2 sigmat3,sigmas,eta12 eta13 eta23];
%output res_y = p_sigmay2_p[sigmat1 sigmat2 sigmat3,sigmas,eta12 eta13 eta23];
%output res_z = p_sigmaz2_p[sigmat1 sigmat2 sigmat3,sigmas,eta12 eta13 eta23];
% sigmap=3e5*30e-9;
% sigmas = 5e-3;
% eta=0.3;%Ri站距离和测量之间的相关系数
[res_x,res_y,res_z,G] = deal(0);
value = num2cell(value);
[sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23] = deal(value{:});%将参数传入各个变量
eta_matrix = [1,eta12,eta13;eta12,1,eta23;eta13,eta23,1];
sigmat_vec = [sigmat1,sigmat2,sigmat3];
c = 0.3*1e3;%光速，单位m/us
m = emitter(1);n = emitter(2);p = emitter(3);
x_row = num2cell(sensor(:,1));y_col = num2cell(sensor(:,2));z_dim = num2cell(sensor(:,3));
[xt,x1,x2,x3] = deal(x_row{:});
[yt,y1,y2,y3] = deal(y_col{:});
[zt,z1,z2,z3] = deal(z_dim{:});

r1=((m-x1).^2+(n-y1).^2+(p-z1).^2).^(1/2);
r2=((m-x2).^2+(n-y2).^2+(p-z2).^2).^(1/2);
r3=((m-x3).^2+(n-y3).^2+(p-z3).^2).^(1/2);
rt=((m-xt).^2+(n-yt).^2+(p-zt).^2).^(1/2);
c11=(m-x1)/r1;c21=(m-x2)/r2;c31=(m-x3)/r3;ct1=(m-xt)/rt;
c12=(n-y1)/r1;c22=(n-y2)/r2;c32=(n-y3)/r3;ct2=(n-yt)/rt;
c13=(p-z1)/r1;c23=(p-z2)/r2;c33=(p-z3)/r3;ct3=(p-zt)/rt;
F=[(-ct1+c11) (-ct2+c12) (-ct3+c13);(-ct1+c21) (-ct2+c22) (-ct3+c23);(-ct1+c31) (-ct2+c32) (-ct3+c33)];

C=pinv(F);
c1 = C(1,:)';c2 = C(2,:)';c3 = C(3,:)';
% T = sigmap.^2.*[1,eta,eta;eta,1,eta;eta,eta,1];
T = c.^2.*[sigmat1^2,eta12*sigmat1*sigmat2,eta13*sigmat1*sigmat3;
    eta12*sigmat1*sigmat2,sigmat2^2,eta23*sigmat2*sigmat3;
    eta13*sigmat1*sigmat3,eta23*sigmat2*sigmat3,sigmat3^2];
S = sigmas.^2.*[2,1,1;1,2,1;1,1,2];
P = C*(T+S)*C';
G = trace(P);
Gx = P(1,1);
Gy = P(2,2);
Gz = P(3,3);
% anss = 0; %anss=G
% for l=1:1:length(c1)
%     anss = anss + (c1(l)*c1'+c2(l)*c2'+c3(l)*c3')*(T(:,l)+S(:,l));
% end

%求sigma2x、sigma2y和sigma2z关于sigmap、sigmas和eta的偏微分公式（\label{eq40}）
%% 偏微分求解 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [res_x] = cal_diff_func(eta_matrix,sigmat_vec,sigmas,c1);
% [res_y] = cal_diff_func(eta_matrix,sigmat_vec,sigmas,c2);
% [res_z] = cal_diff_func(eta_matrix,sigmat_vec,sigmas,c3);
% for i = 1:1:length(res_x)
%     sol_i = double(subs(diff(G,value(i))-(res_x(i)+res_y(i)+res_z(i)), [sigmas, sigmat1, sigmat2, sigmat3, eta12, eta13, eta23], ...
%     [randn, randn, randn, randn, randn, randn, randn])); %给未知数赋予随机值，结果为0说明真值和计算值相等
%     if sol_i>0.01
%         disp('wrong in cal diff function!')
%     end
% end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if gdop>1e5
%     gdop = 0;
%     gdop_x = 0;
%     gdop_y = 0;
%     gdop_z = 0;
% end

