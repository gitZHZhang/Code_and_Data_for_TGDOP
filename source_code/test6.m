close all
%% 论文模4乘积公式验证
N1 = 2;N2=3;N3=4;N4=5;
X = randn([N1,N2,N3,N4]);
X_3d = sum(X,4);
X_1 = tens2mat(X,1);X_2 = tens2mat(X,2);X_3 = tens2mat(X,3);
u1 = kron(ones(N4,1),eye(N2*N3));u2 = kron(ones(N4,1),eye(N1*N3));u3 = kron(ones(N4,1),eye(N2*N1));
u1_inv = 1/N4*kron(ones(1,N4),eye(N2*N3));u2_inv = kron(ones(1,N4),eye(N1*N3));u3_inv = kron(ones(1,N4),eye(N2*N1));
Y_1 = X_1*u1;%论文公式18
Y_2 = X_2*u2;%论文公式19
Y_3 = X_3*u3;%论文公式20
% X_3d_hat = mat2tens(X_1*b,1);
Y_1_HAT = tens2mat(X_3d,1);
Y_2_HAT = tens2mat(X_3d,2);
Y_3_HAT = tens2mat(X_3d,3);
disp(Y_1_HAT-Y_1)
disp(Y_2_HAT-Y_2)
disp(Y_3_HAT-Y_3)
disp(u1_inv*u1)
%% 验证优化算法中四个子问题公式

N1 = 5;N2 = 6;N3 = 7;N4 = 3;
M1 = 2;M2 = 3;M3 = 4;

%% 1- L.DeLathauwer BTD论文公式
% U = randn(N1,M1);
% V = randn(N2,M2);
% W = randn(N3,M3);
% S = randn(M1,M2,M3);
% X = tmprod(S,{U,V,W},1:3);
% % X_(1)^T的展开式(3.2)
% X_1 = tens2mat(X,1)';
% res1= kron(W,V)*tens2mat(S,1)'*U';
% frob(X_1-res1)
% % X_(2)^T的展开式(3.3)
% X_2 = tens2mat(X,2)';
% res2= kron(W,U)*tens2mat(S,2)'*V';
% frob(X_2-res2)
% % X_(3)^T的展开式(3.4)
% X_3 = tens2mat(X,3)';
% res3= kron(V,U)*tens2mat(S,3)'*W';
% frob(X_3-res3)
% % Vec(X)的展开式(3.5)
% vec_X = vec(X);
% res4 = kron(W,V,U)*vec(S);
% frob(vec_X-res4)


%% 2- 本论文中公式
S1 = randn(M1,M2,M3); S2 = randn(M1,M2,M3); S3 = randn(M1,M2,M3);
V11 = randn(N1,M1);   V12 = randn(N1,M1);   V13 = randn(N1,M1);   e1 = [1 0 0]';    V1 = [V11,V12,V13];
V21 = randn(N2,M2);   V22 = randn(N2,M2);   V23 = randn(N2,M2);   e2 = [0 1 0]';    V2 = [V21,V22,V23];
V31 = randn(N3,M3);   V32 = randn(N3,M3);   V33 = randn(N3,M3);   e3 = [0 0 1]';    V3 = [V31,V32,V33];
X = outprod(tmprod(S1,{V11,V21,V31},1:3),e1)+outprod(tmprod(S2,{V12,V22,V32},1:3),e2)+outprod(tmprod(S3,{V13,V23,V33},1:3),e3);
G = sum(X,4);

% G_(1)的展开式
E_K_V3_K_V2 = [kron(e1,V31,V21),kron(e2,V32,V22),kron(e3,V33,V23)];
G_1 = tens2mat(G,1);
res1 = V1*blkdiag(tens2mat(S1,1),tens2mat(S2,1),tens2mat(S3,1))*E_K_V3_K_V2'*kron(ones(3,1),eye(N2*N3));
frob(G_1-res1)
% G_(2)的展开式
E_K_V3_K_V1 = [kron(e1,V31,V11),kron(e2,V32,V12),kron(e3,V33,V13)];
G_2 = tens2mat(G,2);
res2 = V2*blkdiag(tens2mat(S1,2),tens2mat(S2,2),tens2mat(S3,2))*E_K_V3_K_V1'*kron(ones(3,1),eye(N1*N3));
frob(G_2-res2)
% G_(3)的展开式
E_K_V2_K_V1 = [kron(e1,V21,V11),kron(e2,V22,V12),kron(e3,V23,V13)];
G_3 = tens2mat(G,3);
res3 = V3*blkdiag(tens2mat(S1,3),tens2mat(S2,3),tens2mat(S3,3))*E_K_V2_K_V1'*kron(ones(3,1),eye(N2*N1));
frob(G_3-res3)
% VEC(G)的展开式
vec_G = vec(G);
E_K_V3_K_V2_K_V1 = [kron(e1,V31,V21,V11),kron(e2,V32,V22,V12),kron(e3,V33,V23,V13)];
res4 = kron(ones(3,1),eye(N3*N2*N1))'*E_K_V3_K_V2_K_V1*[vec(S1);vec(S2);vec(S3)];
frob(vec_G-res4)
%% 画模4转置积的图
figure
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
set(gca,'ztick',[],'zticklabel',[])
origin_cube_pos = [0  0  0;10  0  0;20  0  0]; %立方体左下角点的位置
% origin_cube_pos = [0  0  0;5 0 0;10 0 0]; %用来搞flatten
cube_size = [5 5 5;5 5 5;5 5 5];%立方体大小
bias = [2 2 2]; %平面切片的偏移量
N_f = 5;%切片分割为N_f个纤维
n=2;
for i =1:1:size(origin_cube_pos,1)
    % 绘画方块
    plotcube(cube_size(i,:),origin_cube_pos(i,:),.8,[1 1 1]);
    hold on
    % 绘画切片
    bias_i = bias(i);
    origin_cube_pos_i = origin_cube_pos(i,:);
    if n==1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%切片
        corner1 = origin_cube_pos_i + [0 0 bias_i];
        corner2 = origin_cube_pos_i + [0 cube_size(i,2) bias_i];
        corner3= origin_cube_pos_i + [cube_size(i,1) cube_size(i,2) bias_i];
        corner4 = origin_cube_pos_i + [cube_size(i,1) 0 bias_i];
        fill3([corner1(1), corner2(1), corner3(1), corner4(1)], ...
            [corner1(2), corner2(2), corner3(2), corner4(2)], ...
            [corner1(3), corner2(3), corner3(3), corner4(3)],'g');
        hold on
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%纤维
        for j =1:1:N_f
            a1 = cube_size(i,1)/N_f*(j-1);
            a11 = cube_size(i,1)/N_f*j;
            corner1 = origin_cube_pos_i + [a1 0 bias_i];
            corner2 = origin_cube_pos_i + [a1 cube_size(i,2) bias_i];
            corner3= origin_cube_pos_i + [a11 cube_size(i,2) bias_i];
            corner4 = origin_cube_pos_i + [a11 0 bias_i];
            if j==1
                col = [0.22,0.37,0.06];
            else
                col = 'g';
            end
            fill3([corner1(1), corner2(1), corner3(1), corner4(1)], ...
                [corner1(2), corner2(2), corner3(2), corner4(2)], ...
                [corner1(3), corner2(3), corner3(3), corner4(3)], col);
            hold on
        end
    elseif n==2
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%切片
        corner1 = origin_cube_pos_i + [0 bias_i 0];
        corner2 = origin_cube_pos_i + [0 bias_i cube_size(i,3)];
        corner3= origin_cube_pos_i + [cube_size(i,1) bias_i cube_size(i,3)];
        corner4 = origin_cube_pos_i + [cube_size(i,1) bias_i 0];
        fill3([corner1(1), corner2(1), corner3(1), corner4(1)], ...
            [corner1(2), corner2(2), corner3(2), corner4(2)], ...
            [corner1(3), corner2(3), corner3(3), corner4(3)], 'r');
        hold on
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%纤维
        for j =1:1:N_f
            a1 = cube_size(i,2)/N_f*(j-1);
            a11 = cube_size(i,2)/N_f*j;
            corner1 = origin_cube_pos_i + [0 bias_i a1];
            corner2 = origin_cube_pos_i + [cube_size(i,3) bias_i a1];
            corner3= origin_cube_pos_i + [cube_size(i,3) bias_i a11];
            corner4 = origin_cube_pos_i + [0 bias_i a11];
            if j==N_f
                col = [0.37,0.15,0.07];
            else
                col = 'r';
            end
            fill3([corner1(1), corner2(1), corner3(1), corner4(1)], ...
                [corner1(2), corner2(2), corner3(2), corner4(2)], ...
                [corner1(3), corner2(3), corner3(3), corner4(3)], col);
            hold on
        end
    elseif n==3
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%切片
        corner1 = origin_cube_pos_i + [0 bias_i 0];
        corner2 = origin_cube_pos_i + [0 bias_i cube_size(i,3)];
        corner3= origin_cube_pos_i + [cube_size(i,1) bias_i cube_size(i,3)];
        corner4 = origin_cube_pos_i + [cube_size(i,1) bias_i 0];
        fill3([corner1(1), corner2(1), corner3(1), corner4(1)], ...
            [corner1(2), corner2(2), corner3(2), corner4(2)], ...
            [corner1(3), corner2(3), corner3(3), corner4(3)], 'y');
        hold on
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%纤维
        for j =1:1:N_f
            a1 = cube_size(i,1)/N_f*(j-1);
            a11 = cube_size(i,1)/N_f*j;
            corner1 = origin_cube_pos_i + [a1 bias_i 0];
            corner2 = origin_cube_pos_i + [a1 bias_i cube_size(i,3)];
            corner3= origin_cube_pos_i + [a11 bias_i cube_size(i,3)];
            corner4 = origin_cube_pos_i + [a11 bias_i 0];
            if j==1
                col = [1,0.38,0];
            else
                col = 'y';
            end
            fill3([corner1(1), corner2(1), corner3(1), corner4(1)], ...
                [corner1(2), corner2(2), corner3(2), corner4(2)], ...
                [corner1(3), corner2(3), corner3(3), corner4(3)], col);
            hold on
        end
    end
end
axis equal;




