clear
close all
Proposed = load('test17_data/tdoa1/res_cell1.mat').res_cell1;
BTD = load('test17_data/tdoa1/res_cell2.mat').res_cell2;
Proposed_means = cell2mat(cellfun(@(x) mean(x(:)), Proposed,'UniformOutput',false));
BTD_means = cell2mat(cellfun(@(x) mean(x(:)), BTD,'UniformOutput',false));

Proposed_means= log(Proposed_means);
BTD_means = log(BTD_means);

down_samples_list = round(800./[2,4,6,8,10,12,14,16,18,20]);
NAN_ratio_list = 1-[0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.99];
% figure; % 创建一个新的图形窗口
% heatmap(NAN_ratio_list,down_samples_list,Proposed_means);
% figure; % 创建一个新的图形窗口
% heatmap(NAN_ratio_list,down_samples_list,BTD_means);
global_min = min([Proposed_means(:);BTD_means(:)]);
global_max = max([Proposed_means(:);BTD_means(:)]);
max_val = max(max(max(Proposed_means)),max(max(BTD_means)));


figure(1);
imagesc(NAN_ratio_list,down_samples_list,Proposed_means);
colorbar;
clim([global_min, global_max]);
xlabel(' Fraction of entries observed ')
ylabel('Size of tensor (N)')
title('Tensor completion error with proposed method');
% hold on;
% [M, c] = contour(NAN_ratio_list,down_samples_list,Proposed_means, [6.5 6.5], ...
%     'LineWidth', 2, 'Color', 'r', 'LineStyle', '--');


figure(2);
imagesc(NAN_ratio_list,down_samples_list,BTD_means);
colorbar;
clim([global_min, global_max]);
xlabel(' Fraction of entries observed ')
ylabel('Size of tensor (N)')
title('Tensor completion error with traditional BTD');
% hold on;
% [M, c] = contour(NAN_ratio_list,down_samples_list,BTD_means, [5.6 5.6], ...
%     'LineWidth', 2, 'Color', 'r', 'LineStyle', '--');





% Proposed
L1 = [5 4 4];
L2 = [4 5 5];
L3 = [5 5 4];
L = [L1;L2;L3];
m = down_samples_list;
lbound_for_Tsize= [];

for i = 1:1:length(m)
    Tensor_size = [m(i)+m(i)*11,m(i)+m(i)*11,11+m(i)^2]; % size m_i,m_i,11  -> unfold: m_i*(m(i)*11), m_i*(m(i)*11), 11*m(i)^2
    LBound = [];
    poly = 8 + (length(m)-i)*2;
    for r = 1:1:3
        for k = 1:1:3
            lb = (L(r,k)+floor((length(m)-i)))*(poly+poly^2);
            LBound = [LBound,lb];
        end
    end
    LBound = LBound/(m(i)*m(i)*11); % 转化成百分比
    lbound_for_Tsize = [lbound_for_Tsize,max(LBound)];
end
figure(1)
hold on
% plot(lbound_for_Tsize,down_samples_list, 'LineWidth', 2, 'Color', 'r', 'LineStyle', '--')
plot([lbound_for_Tsize,lbound_for_Tsize(end)],[down_samples_list,0], 'r--','LineWidth', 2)

% BTD
L1 = [5 5 3];
L2 = [6 6 3];
L3 = [7 7 3];
L = [L1;L2;L3];
m = down_samples_list;
lbound_for_Tsize= [];
for i = 1:1:length(m)
    Tensor_size = [m(i)+m(i)*11,m(i)+m(i)*11,11+m(i)^2]; % size m_i,m_i,11  -> unfold: m_i*(m(i)*11), m_i*(m(i)*11), 11*m(i)^2
    LBound = [];
    for r = 1:1:3
        for k = 1:1:3
            lb = (L(r,k)+floor((length(m)-i)/2))*Tensor_size(k);
            LBound = [LBound,lb];
        end
    end
    LBound = LBound/(m(i)*m(i)*11); % 转化成百分比
    lbound_for_Tsize = [lbound_for_Tsize,max(LBound)];
end
figure(2)
hold on
% plot(lbound_for_Tsize,down_samples_list, 'LineWidth', 2, 'Color', 'r', 'LineStyle', '--')
plot([lbound_for_Tsize,lbound_for_Tsize(end)],[down_samples_list,0], 'r--','LineWidth', 2)