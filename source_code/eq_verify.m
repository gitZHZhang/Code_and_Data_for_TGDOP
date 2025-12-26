function [sol1,sol2,sol3,sol4] = eq_verify(emitter,sensor)
% sigmap=3e5*30e-9;
% sigmas = 5e-3;
% eta=0.3;%Ri站距离和测量之间的相关系数
syms sigmat1 sigmat2 sigmat3
syms sigmas
syms eta12 eta13 eta23
eta_matrix = [1,eta12,eta13;eta12,1,eta23;eta13,eta23,1];
sigmat_vec = [sigmat1,sigmat2,sigmat3];
c = 3;%光速
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

sigma_error = T+S;
P_error = C*sigma_error*C';
P_error_TDOA = C*T*C';
P_error_S = C*S*C';

gdop=(trace(P_error)).^(1/2);
G = gdop^2;
gdop2_TDOA=(trace(P_error_TDOA));
gdop2_S=(trace(P_error_S));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%验证sigma2x、sigma2y和sigma2z的展开式（\label{eq36}）
sigma2_x = trace(P_error(1,1));%真值
sigma2_y = trace(P_error(1:2,1:2))-sigma2_x;%真值
sigma2_z = trace(P_error(1:3,1:3))-sigma2_x-sigma2_y;%真值
[sigma2_x_hat,sigma2_y_hat,sigma2_z_hat] = deal(0);%由公式计算的值
for i=1:1:length(c1)
    t_i = T(:,i);
    s_i = S(:,i);
    sigma2_x_hat = sigma2_x_hat+c1(i).*(c1'*(t_i+s_i));
    sigma2_y_hat = sigma2_y_hat+c2(i).*(c2'*(t_i+s_i));
    sigma2_z_hat = sigma2_z_hat+c3(i).*(c3'*(t_i+s_i));
end
sol1 = double(subs(sigma2_x_hat-sigma2_x, [sigmas, sigmat1, sigmat2, sigmat3, eta12, eta13, eta23], ...
    [randn, randn, randn, randn, randn, randn, randn])); %给未知数赋予随机值，结果为0说明真值和计算值相等

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%验证sigma2x、sigma2y和sigma2z关于sigmap、sigmas和eta的偏微分公式（\label{eq40}）
%仅需证明simgax2对各个位置参数的导数，其他的把c1换成c2和c3类推即可

%% 1- p_sigmax2/p_sigma_t_i
sigma2_x_hat = 0;
p_sigmax2_p_sigmat_hat = 0;
i=3;%对第i个未知量求偏微分
p_sigmax2_p_sigmat = 0;
for l=1:1:length(c1)
    t_l = T(:,l);
    s_l = S(:,l);
    %给出偏微分真值
    p_tl_p_sigmat = {diff(t_l,sigmat1),diff(t_l,sigmat2),diff(t_l,sigmat3)};
    sigma2_x_hat = sigma2_x_hat+c1(l).*(c1'*(t_l+s_l));%sigmax2真值
    temp = c1(l).*(c1'*(p_tl_p_sigmat{i}));
    p_sigmax2_p_sigmat_hat = p_sigmax2_p_sigmat_hat+temp;%偏微分真值

    %用所提公式计算偏微分:公式的具体推导流程
%     
% %     if l==i
% %         res = 0;
% %         for j=1:length(c1)
% %             if j~=l
% %                 res = res + c^2*(c1(l)*c1(j)*eta_matrix(l,j)*sigmat_vec(j));
% %             end
% % %         ans = c^2*(c1(i)*c1(2)*eta_matrix()*sigmat_vec(2)+c1(i)*c1(3)*eta_matrix()*sigmat_vec(3)+2*c1(i)^2*sigmat_vec(1));
% %         end
% %         res = res + c^2*2*c1(l)^2*sigmat_vec(l);
% %     else
% %         res = c^2*c1(i)*c1(l)*eta_matrix(i,l)*sigmat_vec(l);
% %     end
    %公式的解析解
    res = 2*c^2*c1(i)*c1(l)*eta_matrix(i,l)*sigmat_vec(l);
%     disp(temp-res)
    p_sigmax2_p_sigmat = p_sigmax2_p_sigmat + res;%偏微分计算值
end
p_sigmax2_p_sigmat1 = diff(sigma2_x_hat,sigmat_vec(i));%偏微分真值
sol2 = double(subs(p_sigmax2_p_sigmat-p_sigmax2_p_sigmat1, [sigmas, sigmat1, sigmat2, sigmat3, eta12, eta13, eta23], ...
    [randn, randn, randn, randn, randn, randn, randn])); %给未知数赋予随机值，结果为0说明真值和计算值相等
%% 2- p_sigmax2/p_eta_ij
sigma2_x_hat = 0;
p_sigmax2_p_eta_hat = 0;
i=1;j=2;%eta_ij
p_sigmax2_p_etaij = 0;
for l=1:1:length(c1)
    t_l = T(:,l);
    s_l = S(:,l);
    p_tl_p_etaij = {1,diff(t_l,eta12),diff(t_l,eta13);diff(t_l,eta12),1,diff(t_l,eta23);diff(t_l,eta13),diff(t_l,eta23),1};
    sigma2_x_hat = sigma2_x_hat+c1(l).*(c1'*(t_l+s_l));%sigmax2真值
    temp = c1(l).*(c1'*(p_tl_p_etaij{i,j}));
    p_sigmax2_p_eta_hat = p_sigmax2_p_eta_hat+temp;%偏微分真值

    %用所提公式计算偏微分:公式的具体推导流程
    if l==j || l==i
        res = c^2*c1(i)*c1(j)*sigmat_vec(i)*sigmat_vec(j);
    else
        res = 0;
    end
%     disp(temp-res)
    p_sigmax2_p_etaij = p_sigmax2_p_etaij + res;%偏微分计算值
end

eq_res = 2*c^2*c1(i)*c1(j)*sigmat_vec(i)*sigmat_vec(j);%公式的解析解
p_sigmax2_p_eta = diff(sigma2_x_hat,eta_matrix(i,j));%偏微分真值
sol3 = double(subs(p_sigmax2_p_etaij-p_sigmax2_p_eta, [sigmas, sigmat1, sigmat2, sigmat3, eta12, eta13, eta23], ...
    [randn, randn, randn, randn, randn, randn, randn])); %给未知数赋予随机值，结果为0说明真值和计算值相等
sol3 = double(subs(eq_res-p_sigmax2_p_eta, [sigmas, sigmat1, sigmat2, sigmat3, eta12, eta13, eta23], ...
    [randn, randn, randn, randn, randn, randn, randn])); %给未知数赋予随机值，结果为0说明真值和计算值相等
%% 3- p_sigmax2/p_sigmas
sigma2_x_hat = 0;
p_sigmax2_p_sigmas_hat = 0;
i=3;%对第i个未知量求偏微分
p_sigmax2_p_sigmas = 0;
for l=1:1:length(c1)
    t_l = T(:,l);
    s_l = S(:,l);
    %给出偏微分真值
    p_sl_p_sigmas = diff(s_l,sigmas);
    sigma2_x_hat = sigma2_x_hat+c1(l).*(c1'*(t_l+s_l));%sigmax2真值
    temp = c1(l).*(c1'*(p_sl_p_sigmas));
    p_sigmax2_p_sigmas_hat = p_sigmax2_p_sigmas_hat+temp;%偏微分真值

    %用所提公式计算偏微分:推导过程即解析解
    res = 0;
    for j=1:length(c1)
        if j~=l
            res = res + 2*sigmas*c1(j)*c1(l);
        end
        %         ans = c^2*(c1(i)*c1(2)*eta_matrix()*sigmat_vec(2)+c1(i)*c1(3)*eta_matrix()*sigmat_vec(3)+2*c1(i)^2*sigmat_vec(1));
    end
    res = res + 2*sigmas*2*c1(l)^2;

%     disp(temp-res)
    p_sigmax2_p_sigmas = p_sigmax2_p_sigmas + res;%偏微分计算值
end
p_sigmax2_p_sigmas_real = diff(sigma2_x_hat,sigmas);%偏微分真值
sol4 = double(subs(p_sigmax2_p_sigmas-p_sigmax2_p_sigmas_real, [sigmas, sigmat1, sigmat2, sigmat3, eta12, eta13, eta23], ...
    [randn, randn, randn, randn, randn, randn, randn])); %给未知数赋予随机值，结果为0说明真值和计算值相等



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if gdop>1e5
%     gdop = 0;
%     gdop_x = 0;
%     gdop_y = 0;
%     gdop_z = 0;
% end

