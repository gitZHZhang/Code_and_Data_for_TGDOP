function [W,P] = Propose_CI(P1,P2,lambda,tol,max_iter)
W = eye(size(P1)); % 初始权重矩阵，单位矩阵
learning_rate_W = 0.01; % 学习率
learning_rate_lambda = 0.01; % 学习率

% 优化循环
for iter = 1:max_iter
    % 计算 P
    P = W * P1 * W' + (eye(size(P1)) - W) * P2 * (eye(size(P1)) - W)';
    
    % 计算目标函数 L
    L = (1-lambda)*trace(P) + lambda * log(det(P));  %log(det(P))=log(det((eye-W)*P2*(eye-W)'+W*P1*W'))
    
    % 计算 P 的梯度 grad_P_W
    grad_P = 2 * W * P1 - 2 * (eye(size(P1)) - W)* P2;
    
    T0 = eye(size(P1)) - W;
    T1 = eye(size(P1)) + (- W)';
    T2 = pinv(T0*P2'*T1+W*P1'*W');
    T3 = pinv(T0*P2*T1+W*P1*W');
    grad_log_det = T2*W*P1'-(T2*T0*P2'+T3*T0*P2)+T3*W*P1;
    % 计算 det(P) 的梯度
    if det(P) == 0
        grad_det = 0;
    else
        invP = inv(P);
        grad_P_term = P1 * W' + W * P1 - P2 * (eye(size(P1)) - W)' - (eye(size(P1)) - W) * P2;
        grad_det = det(P) * trace(invP * grad_P_term);
    end
    
    % 计算 L 关于 W 的梯度
    grad_L_W = (1-lambda)*grad_P + lambda * grad_log_det;
    % grad_L_W = grad_P;
    
    % 更新 W 和 lambda
    W = W - learning_rate_W * grad_L_W;
    % 投影 W 到半正定矩阵空间
    % [Q, Lambda] = eig(W);
    % Lambda = max(diag(Lambda), 0); % 将负特征值置零
    % W = Q * diag(Lambda) * Q';

    % lambda = lambda - learning_rate_lambda * det(P);
    % if lambda<0
    %     lambda=0;
    % end
    
    % 检查收敛条件
    if iter > 1
        grad_norm = norm(grad_L_prev - grad_L_W);
        if grad_norm < tol
            break;
        end
    end
    grad_L_prev = grad_L_W;
end