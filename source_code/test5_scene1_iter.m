close all
clear all
syms k1 k2 k3 %常数，但有多种
syms x1 x2 x3 %变量
y = k1*x1^2+k2*x2^2+k3;
%% 先给出变量的真值
x1_real = 0.3;
x2_real = 0.25;
%% 仿真6组常数并生成对应的label(y)
y_real = [];
k_real = [];
for i=1:1:6
    k_i = randn(3,1);
    k_real = [k_real,k_i];
    y_real = [y_real,subs(y, [k1,k2,k3,x1,x2],[k_i;x1_real;x2_real])];
end
%% 