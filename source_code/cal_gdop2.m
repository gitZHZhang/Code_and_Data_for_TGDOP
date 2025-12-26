function [gdop,gdop2_TDOA,gdop2_S,sigma2_x,sigma2_y,sigma2_z,G] = cal_gdop2(emitter,sensor)
m = emitter(1);n = emitter(2);p = emitter(3);
x_row = num2cell(sensor(:,1));y_col = num2cell(sensor(:,2));z_dim = num2cell(sensor(:,3));
[xt,x1,x2,x3] = deal(x_row{:});
[yt,y1,y2,y3] = deal(y_col{:});
[zt,z1,z2,z3] = deal(z_dim{:});

r1=((m-x1).^2+(n-y1).^2+(p-z1).^2).^(1/2);
r2=((m-x2).^2+(n-y2).^2+(p-z2).^2).^(1/2);
r3=((m-x3).^2+(n-y3).^2+(p-z3).^2).^(1/2);
rt=((m-xt).^2+(n-yt).^2+(p-zt).^2).^(1/2);
c11=(m-x1)/r1;c21=(m-x2)/r2;c31=(m-x3)/r3;ct1=(m-xt)/rt;
c12=(n-y1)/r1;c22=(n-y2)/r2;c32=(n-y3)/r3;ct2=(n-yt)/rt;
c13=(p-z1)/r1;c23=(p-z2)/r2;c33=(p-z3)/r3;ct3=(p-zt)/rt;
c=[(-ct1+c11) (-ct2+c12) (-ct3+c13);(-ct1+c21) (-ct2+c22) (-ct3+c23);(-ct1+c31) (-ct2+c32) (-ct3+c33)];

b=pinv(c);
sigmap=3e5*30e-9;
% sigmap = 1;
sigmas = 5e-3;
% sigmas = 1;
eta=0.3;%Ri站距离和测量之间的相关系数
% sigma11=sigmap.^2+sigmas.^2*2;
% sigma12=eta*sigmap.^2+sigmas.^2;
% sigma13=eta*sigmap.^2+sigmas.^2;
% sigma21=eta*sigmap.^2+sigmas.^2;
% sigma22=sigmap.^2+sigmas.^2*2;
% sigma23=eta*sigmap.^2+sigmas.^2;
% sigma31=eta*sigmap.^2+sigmas.^2;
% sigma32=eta*sigmap.^2+sigmas.^2;
% sigma33=sigmap.^2+sigmas.^2*2;
% sigma_error = [sigma11,sigma12,sigma13;sigma21,sigma22,sigma23;sigma31,sigma32,sigma33];
sigma_error_TDOA = sigmap.^2.*[1,eta,eta;eta,1,eta;eta,eta,1];
sigma_error_S = sigmas.^2.*[2,1,1;1,2,1;1,1,2];
sigma_error = sigma_error_TDOA+sigma_error_S;
P_error = b*sigma_error*b';
P_error_TDOA = b*sigma_error_TDOA*b';
P_error_S = b*sigma_error_S*b';

gdop=(trace(P_error)).^(1/2);
sigma2_x = trace(P_error(1,1));
sigma2_y = trace(P_error(1:2,1:2))-sigma2_x;
sigma2_z = trace(P_error(1:3,1:3))-sigma2_x-sigma2_y;
% if gdop>1e3
%     gdop = 0;
%     sigma2_x = 0;
%     sigma2_y = 0;
%     sigma2_z = 0;
% end
G = gdop^2;
gdop2_TDOA=(trace(P_error_TDOA));
gdop2_S=(trace(P_error_S));
%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%sigmax2
dim = 'x';
[sigma2_TDOA,sigma2_S] = cal_sigma2_xyz(b,sigma_error_TDOA,sigma_error_S,dim);
gdop_x_hat = sqrt(sigma2_TDOA+sigma2_S);
%%%%%%%%%%%%%%%%%%%%%%%sigmay2
dim = 'y';
[sigma2_TDOA,sigma2_S] = cal_sigma2_xyz(b,sigma_error_TDOA,sigma_error_S,dim);
gdop_y_hat = sqrt(sigma2_TDOA+sigma2_S);
%%%%%%%%%%%%%%%%%%%%%sigmaz2
dim = 'z';
[sigma2_TDOA,sigma2_S] = cal_sigma2_xyz(b,sigma_error_TDOA,sigma_error_S,dim);
gdop_z_hat = sqrt(sigma2_TDOA+sigma2_S);
