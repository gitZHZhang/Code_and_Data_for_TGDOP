close all
clear all
% 协方差交叉与观测量融合算法
% 输入矩阵 P1 和 P2，假设它们是对称正定矩阵
pho1 = 0.3; %相关系数范围【-1，1】
sigmax1 = 0.5; sigmay1 = 2.5;
pho2 = -0.2;
sigmax2 = 2.3; sigmay2 = 0.3;
P1 = [sigmax1^2, pho1*sigmax1*sigmay1; pho1*sigmax1*sigmay1, sigmay1^2]; 
P2 = [sigmax2^2, pho2*sigmax2*sigmay2; pho2*sigmax2*sigmay2, sigmay2^2]; 

% sigmax1 = 2; sigmay1 = 1;
% theta = 150/180*pi;
% P1 = generate_P(sigmax1,sigmay1,theta);
% sigmax1 = 1.5; sigmay1 = 2.5;
% theta = 140/180*pi;
% P2 = generate_P(sigmax1,sigmay1,theta);

sigmax1 = 2; sigmay1 = 0.1;
theta = 150/180*pi;
P1 = generate_P(sigmax1,sigmay1,theta);
sigmax1 = 2.5; sigmay1 = 0.15;
theta = 140/180*pi;
P2 = generate_P(sigmax1,sigmay1,theta);

% P1 = [2 0;0 1];P2 = [1 0;0 2];
%% EI
% 确保 T 是非奇异的
[V1, D1] = eig(P1);
[V2, D2] = eig(pinv(sqrt(D1))*pinv(V1)*P2*V1*pinv(sqrt(D1)));
T = V1*sqrt(D1)*V2;
D_tau = diag(max(diag(D2), 1));
tau = T*D_tau*T';
EI_P = pinv(pinv(P1)+pinv(P2)-pinv(tau));
% EI_x = EI_P*(pinv(P1)*x1+pinv(P2)*x2-pinv(tau)*gamma);

%% Bar-ShalomandCampo
BC_P = pinv(pinv(P1)+pinv(P2));
% BC_x = BC_P*(pinv(P1)*x1+pinv(P2)*x2);
%% CI算法
w = 0.5; %0到1
CI_P = pinv(w*pinv(P1)+(1-w)*pinv(P2));
% CI_x = CI_P*(w*pinv(P1)*x1+(1-w)*pinv(P2)*x2);
%% 仅约束tr（P）的最优估计
CRLB_W = P2*pinv(P1+P2);
CRLB_P = CRLB_W*P1*CRLB_W'+(eye(size(CRLB_W))-CRLB_W)*P2*(eye(size(CRLB_W))-CRLB_W)';
% CRLB_x = CRLB_W*x1+(eye(size(CRLB_W))-CRLB_W)*x2;

%% Proposed
% 初始化权重矩阵 W 和惩罚项 lambda



% 设置参数
tol = 1e-6; % 收敛容忍度
max_iter = 1e5; % 最大迭代次数

lambda = 0; % 惩罚项
[Propose_W,Propose_P] = Propose_CI(P1,P2,lambda,tol,max_iter);


Pe=0.95;
line_styles = {'-','--','-.',':'};
% Plot_P({P1,P2,CRLB_P,Propose_P},Pe,{'P1','P2','CRLB_P','PROPOSE'},line_styles)
Plot_P({P1,P2,CI_P,Propose_P,EI_P,BC_P},Pe,{'P1','P2','CI_P','Propose','EI_P','BC_P'},line_styles)
% Plot_P({P1,P2,BC_P,Propose_P},Pe,{'P1','P2','BC_P','PROPOSE'},line_styles)
% Plot_P({P1,P2,EI_P,Propose_P},Pe,{'P1','P2','EI_P','PROPOSE'})

P_cell = cell(1,6);
legend_label = cell(1,6);
idx=1;
for lambda = 0:0.2:1
    [Propose_W,Propose_P] = Propose_CI(P1,P2,lambda,tol,max_iter);
    P_cell{idx} = Propose_P;
    legend_label{idx} = ['lambda=',num2str(lambda)];
    idx = idx+1;
end
Plot_P(P_cell,Pe,legend_label,line_styles)

