function [f,theta,sigma1_2,sigma2_2] = Draw_err_ellipse(P, Pe,center)
    % 输入参数：
    % P: 2x2协方差矩阵 
    % mu: 椭圆中心坐标（默认为原点）
    [V, D] = eig(P);
    lambda = diag(D);
    [lambda_sorted, idx] = sort(lambda, 'descend');
    V_sorted = V(:, idx);
    
    % 计算缩放因子k（基于卡方分布）
    alpha = Pe;
    k = sqrt(chi2inv(alpha, 2));
    
    % 椭圆参数 
    a = k * sqrt(lambda_sorted(1));
    b = k * sqrt(lambda_sorted(2));
    theta = atan2(V_sorted(2,1), V_sorted(1,1));

    f=@(x,y) ((x-center(1))*cos(theta)+(y-center(2))*sin(theta))^2/a^2+(-(x-center(1))*sin(theta)+(y-center(2))*cos(theta))^2/b^2-1;
    sigma1_2 = lambda_sorted(1);
    sigma2_2 = lambda_sorted(2);
