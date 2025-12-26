close all
clear all




%% 加载原始数据（和test4的数据相同）
load G.mat
load sigma2_x.mat
load sigma2_y.mat
load sigma2_z.mat

fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;z_span = linspace(10,20,11);
for i=1:1:size(G,1)/fs
    for j=1:1:size(G,2)/fs
        for k=1:1:size(G,3)
            G2(j,i,k) = G(fs*i,fs*j,k);
            Sigma2_x(j,i,k) = sigma2_x(fs*i,fs*j,k);
            Sigma2_y(j,i,k) = sigma2_y(fs*i,fs*j,k);
            Sigma2_z(j,i,k) = sigma2_z(fs*i,fs*j,k);
        end
    end
end
% figure
% mlrankest(Sigma2_y)
% size_core = mlrankest(Sigma2_x);%计算sizecore
% size_core = [3,3,2];
% [UXhat,SXhat,output]=lmlra(Sigma2_x,size_core);
% [UYhat,SYhat,output]=lmlra(Sigma2_y,size_core);
% size_core = [4,4,2];
% [UZhat,SZhat,output]=lmlra(Sigma2_z,size_core);


%% 传感器位置
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
x3 = 0;y3 = -L;z3 = 0.3;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];
%% 按照比例抽取部分G作为观测量
incomplete_T = G2+0.01*randn(size(G2));%0.01km标准差的噪声
ALL_ELE = 1:numel(incomplete_T);
NAN_ratio = 0.99;
CHOSEN_IDX = randperm(numel(incomplete_T),round(NAN_ratio*numel(incomplete_T)));
load test8_data/CHOSEN_IDX_fs_10_ratio_0.99.mat
unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
%画出观测数据在全部区域的分布
a = zeros(1,numel(incomplete_T));
a(unselected_elements)=1;
a_3d = reshape(a,size(G2,1),size(G2,2),size(G2,3));
plot_title = 'The 3-D distribution of the selected grid.';
[~] = plot_3D_tensor(a_3d,x_span,y_span,z_span,plot_title);
incomplete_T(CHOSEN_IDX) = NaN; %将G2中96%的值设置为未知

%% 开始和别的算法作比较:Proposed/AI/kriging/SVT/NNM
frob_list = [];
recon_tensor = zeros(size(G2));
for i = 1:1:size(G2,3)
    A = incomplete_T(:,:,i);
    algorithm = 1;
    
    if algorithm==1
        %% 方法1-1：找出矩阵A中某半径内的全部连通域
        % %直接对A全域插值
        % % A矩阵中的非NaN值的位置
        % [rows, cols] = find(~isnan(A));
        % values = A(~isnan(A));
        % % 创建插值对象
        % F = scatteredInterpolant(rows, cols, values, 'natural');
        % % 定义插值网格
        % [gridRows, gridCols] = find(isnan(A));
        % % 执行插值
        % interpValues = F(gridRows, gridCols);
        % % 将插值结果放入原矩阵的对应位置
        % A(isnan(A)) = interpValues;
        Meas_matrix = a_3d(:,:,i);
        % 定义邻域大小
        neighborhoodSize = 10;%局部线性拟合的半径
        % 创建一个与邻域大小相对应的全1矩阵作为卷积核
        kernel = ones(neighborhoodSize);
        % 计算卷积，'same'选项表示输出与原始矩阵A大小相同
        % 'valid'选项将不包括边缘的计算，如果你需要包括边缘，可以使用'full'选项
        InstructionMatrix = conv2(Meas_matrix, kernel, 'same');
        % 使用 bwlabel 函数标记连通区域
        [L, num] = bwlabel(InstructionMatrix);
        stats = regionprops(L, 'Area', 'PixelIdxList');
        % %区域可视化
        % figure;
        % imshow(L, 'InitialMagnification', 'fit');
        % title('Connected Components');
        % colormap(hot); % 使用热图颜色映射
        % colorbar;
        %% 方法1-2：对每一个连通域所在的子矩阵单独进行局部线性拟合插值
        for j=1:1:num
            pixelIdx = stats(j).PixelIdxList;
            %当前连通区域的两个角对应的矩阵中的索引
            area_begin = pixelIdx(1);
            [row_begin, col_begin] = ind2sub(size(InstructionMatrix), area_begin);
            area_end = pixelIdx(end);
            [row_end, col_end] = ind2sub(size(InstructionMatrix), area_end);
            area_A = A(min(row_begin,row_end):max(row_begin,row_end),min(col_begin,col_end):max(col_begin,col_end));
            % 矩阵中的非NaN值的位置
            [rows, cols] = find(~isnan(area_A));
            if length(rows)>3
                values = area_A(~isnan(area_A));
                % 创建插值对象
                F = scatteredInterpolant(rows, cols, values, 'natural');
                % 定义插值网格
                [gridRows, gridCols] = find(isnan(area_A));
                % 执行插值
                interpValues = F(gridRows, gridCols);
                % [X_axis, Y_axis, interpValues]=griddata(rows, cols, values, gridRows, gridCols);
                % 将插值结果放入原矩阵的对应位置
                area_A(isnan(area_A)) = interpValues;
                A(min(row_begin,row_end):max(row_begin,row_end),min(col_begin,col_end):max(col_begin,col_end))=area_A;
            end
        end
    end
    if algorithm==2
        %% 方法2：Griddata插值(效果不如1)
        [rows, cols] = find(~isnan(A));
        values = A(~isnan(A));
        % 定义插值网格
        [gridRows, gridCols] = find(isnan(A));
        [X_axis, Y_axis, interpValues]=griddata(rows, cols, values, gridRows, gridCols);
        A(isnan(A)) = interpValues;
    end
    %% 对局部插值完的矩阵进行SVT重建
    [m,n]=size(A);
    % 开始CVX模型
    cvx_begin
        variable A2(m,n);
        minimize( norm_nuc(A2) );
        subject to
            % norm(A2(~isnan(A)) - A(~isnan(A)))<=100;
            A2(~isnan(A)) == A(~isnan(A));
    cvx_end

    % [n1,n2] = size(A);
    % Omega = find(~isnan(A));
    % p = length(Omega)/(n1*n2);%矩阵中观测内容比例
    % data = A(Omega);
    % tau = 5*sqrt(n1*n2);
    % delta = 0.1/p;
    % maxiter = 5000;
    % tol = 1e-5;
    % [U,S,V,numiter] = SVT([n1 n2],Omega,data,tau,delta,maxiter,tol);
    % toc
    % A2 = U*S*V';

    recon_tensor(:,:,i) = A2;
    frob_list = [frob_list,frob(A2-G2(:,:,i))];

    % figure();
    % [c,handle]=contour(x_span,y_span,A2,20);
    % clabel(c,handle);
    % title('The reconstructed distribution of $G$ using NNM,','interpreter','latex');
    % xlabel('x(km)');
    % ylabel('y(km)');
    % figure();
    % [c,handle]=contour(x_span,y_span,G2(:,:,i),20);
    % clabel(c,handle);
    % title('The real distribution of $G$','interpreter','latex');
    % xlabel('x(km)');
    % ylabel('y(km)');

end