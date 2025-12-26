clear 
close all
Proposed_LOS = load ('test19_data\test11_MER0.99_Proposed.mat').output.fval;
GeneralBTD_LOS = load ('test19_data\test11_MER0.99_GeneralBTD.mat').output.fval;
Proposed_NLOS = load ('test19_data\test15_MER0.99_Proposed.mat').output.fval;
GeneralBTD_NLOS = load ('test19_data\test15_MER0.99_GeneralBTD.mat').output.fval;

figure;
% 检查是否存在 fval 字段
% 绘制对数坐标下的收敛曲线
semilogy(Proposed_LOS, 'LineWidth', 2); hold on
semilogy(GeneralBTD_LOS, 'LineWidth', 2); hold on
semilogy(Proposed_NLOS, 'LineWidth', 2); hold on
semilogy(GeneralBTD_NLOS, 'LineWidth', 2); hold on
legend('Proposed LOS','BTD LOS','Proposed NLOS','BTD NLOS')
% 添加图表标注
xlim([0,500])
xlabel('Iteration');
ylabel('Objective Function Value ');
title('Convergence Curve');
grid on;


