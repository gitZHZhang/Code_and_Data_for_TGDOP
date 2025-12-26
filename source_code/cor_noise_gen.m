function cor_noise = cor_noise_gen(value,N)
% value = [30e-3, 25e-3, 15e-3,5e-3, 0.1, 0.2, 0.3];
% value = num2cell(value);

[sigmat1,sigmat2,sigmat3,~,eta12,eta13,eta23] = deal(value{:});%将参数传入各个变量
% 设置样本大小

% 定义协方差矩阵 Sigma
% 假设已知三个白噪声的协方差如下：
Sigma = [sigmat1^2, eta12*sigmat1*sigmat2, eta13*sigmat1*sigmat3;
        eta12*sigmat1*sigmat2, sigmat2^2, eta23*sigmat2*sigmat3;
        eta13*sigmat1*sigmat3, eta23*sigmat2*sigmat3, sigmat3^2];

% 进行Cholesky分解
L = chol(Sigma)';

% 生成三个独立的标准正态分布随机向量
Z1 = randn(N, 1);
Z2 = randn(N, 1);
Z3 = randn(N, 1);

% 生成具有协方差矩阵 Sigma 的随机向量
X1 = L(1, 1) * Z1;
X2 = L(2, 1) * Z1 + L(2, 2) * Z2;
X3 = L(3, 1) * Z1 + L(3, 2) * Z2 + L(3, 3) * Z3;
% cor_noise = [X1,X2,X3];
cor_noise = [Z1,Z2,Z3]*L';

% % 检查生成的随机向量的协方差矩阵
% disp('相关系数:');
% disp(corr(X1, X2));
% disp(corr(X1, X3));
% disp(corr(X3, X2));
% disp('方差:');
% disp(var(X1));
% disp(var(X2));
% disp(var(X3));

