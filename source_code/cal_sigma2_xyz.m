function [sigma2_TDOA,sigma2_S] = cal_sigma2_xyz(b,sigma_error_TDOA,sigma_error_S,dim)
b1 = b(1,:);
b2 = b(2,:);
b3 = b(3,:);
if dim=='x'
    bm=b1;
    a1 = sigma_error_TDOA(:,1);
    a2 = sigma_error_TDOA(:,2);
    a3 = sigma_error_TDOA(:,3);
    sigmax2_TDOA = bm(1)*bm*a1+bm(2)*bm*a2+bm(3)*bm*a3;
    a1 = sigma_error_S(:,1);
    a2 = sigma_error_S(:,2);
    a3 = sigma_error_S(:,3);
    sigmax2_S = bm(1)*bm*a1+bm(2)*bm*a2+bm(3)*bm*a3;
    sigma2_TDOA = sigmax2_TDOA;
    sigma2_S = sigmax2_S;
elseif dim=='y'
    bm=b2;
    a1 = sigma_error_TDOA(:,1);
    a2 = sigma_error_TDOA(:,2);
    a3 = sigma_error_TDOA(:,3);
    sigmay2_TDOA = bm(1)*bm*a1+bm(2)*bm*a2+bm(3)*bm*a3;
    a1 = sigma_error_S(:,1);
    a2 = sigma_error_S(:,2);
    a3 = sigma_error_S(:,3);
    sigmay2_S = bm(1)*bm*a1+bm(2)*bm*a2+bm(3)*bm*a3;
    sigma2_TDOA = sigmay2_TDOA;
    sigma2_S = sigmay2_S;
else
    bm=b3;
    a1 = sigma_error_TDOA(:,1);
    a2 = sigma_error_TDOA(:,2);
    a3 = sigma_error_TDOA(:,3);
    sigmaz2_TDOA = bm(1)*bm*a1+bm(2)*bm*a2+bm(3)*bm*a3;
    a1 = sigma_error_S(:,1);
    a2 = sigma_error_S(:,2);
    a3 = sigma_error_S(:,3);
    sigmaz2_S = bm(1)*bm*a1+bm(2)*bm*a2+bm(3)*bm*a3;
    sigma2_TDOA = sigmaz2_TDOA;
    sigma2_S = sigmaz2_S;
end
