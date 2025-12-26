function [f_i,theta,sigma1_2,sigma2_2] = GetGaussFunction(P,center)
% 2024-08-22 给定定位误差的协方差矩阵，给出对应的Pe概率下的误差椭圆
% P：定位误差协方差矩阵
% Pe: 样本点落入椭圆的概率，椭圆越大落入到里面的概率越大
% center: 椭圆圆心的坐标
% f是误差椭圆对应的隐函数句柄，可以用于绘图
theta0 = 1/2*atan(2*P(1,2)/(P(1,1)-P(2,2)));
%1-先求出在（-2pi，2pi）里的4个解
if(theta0<0)
    theta_all = [theta0-pi/2,theta0,theta0+pi/2,theta0+pi];
else
    theta_all = [theta0-pi/2,theta0,theta0+pi/2,theta0-pi];
end
%根据象限决定用哪一个解
if center(1)>0 && center(2)>0 
    theta = theta_all(theta_all<=pi/2&theta_all>=0);
elseif center(1)>0 && center(2)<0
    theta = theta_all(theta_all<=pi&theta_all>=pi/2);
elseif center(1)<0 && center(2)<0
    theta = theta_all(theta_all<=-pi/2&theta_all>=-pi);
else
    theta = theta_all(theta_all<=0/2&theta_all>=-pi/2);
end

sigma1_2 = (2*P(1,1)*P(2,2)-2*P(1,2)^2) / (P(1,1)+P(2,2)-sqrt((P(1,1)-P(2,2))^2+4*P(1,2)^2));
sigma2_2 = (2*P(1,1)*P(2,2)-2*P(1,2)^2) / (P(1,1)+P(2,2)+sqrt((P(1,1)-P(2,2))^2+4*P(1,2)^2));
% k=sqrt(-2*log(1-Pe));
%% 这里是令g（x,y）=1,再得到f(x,y)=0
term1 = 1/(2*pi*sqrt(sigma1_2)*sqrt(sigma2_2));
% term2 = @(x,y) ((x-center(1))*cos(theta)+(y-center(2))*sin(theta))^2/sigma1_2 + (-(x-center(1))*sin(theta)+(y-center(2))*cos(theta))^2/sigma2_2;
f_i=@(x,y) term1.*exp(-1/2.*(((x-center(1)).*cos(theta)+(y-center(2)).*sin(theta)).^2/sigma1_2 + (-(x-center(1)).*sin(theta)+(y-center(2)).*cos(theta)).^2/sigma2_2));
% f=@(x,y) ((x-center(1))*cos(theta)+(y-center(2))*sin(theta))^2/sigma1_2/k^2+(-(x-center(1))*sin(theta)+(y-center(2))*cos(theta))^2/sigma2_2/k^2;

