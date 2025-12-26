function [res_out] = cal_diff_func(eta_matrix,sigmat_vec,sigmas,c_vec)
c = 0.3;%光速，单位km/us
res_out=[];
%% 1- p_sigmax2/p_sigma_t_i
for i =1:1:3
    p_sigmax2_p_sigmat = 0;
    for l=1:1:length(c_vec)
        %公式的解析解
        res = 2*c^2*c_vec(i)*c_vec(l)*eta_matrix(i,l)*sigmat_vec(l);
        %     disp(temp-res)
        p_sigmax2_p_sigmat = p_sigmax2_p_sigmat + res;%偏微分计算值
    end
    res_out = [res_out,p_sigmax2_p_sigmat];
end
%% 2- p_sigmax2/p_sigmas
p_sigmax2_p_sigmas = 0;
for l=1:1:length(c_vec)
    %用所提公式计算偏微分:推导过程即解析解
    res = 0;
    for j=1:length(c_vec)
        if j~=l
            res = res + 2*sigmas*c_vec(j)*c_vec(l);
        end
    end
    res = res + 2*sigmas*2*c_vec(l)^2;
    p_sigmax2_p_sigmas = p_sigmax2_p_sigmas + res;%偏微分计算值
end
res_out = [res_out,p_sigmax2_p_sigmas];
%% 3- p_sigmax2/p_eta_ij
for i=1:1:3
    for j=1:1:3
        if j>i
            eq_res = 2*c^2*c_vec(i)*c_vec(j)*sigmat_vec(i)*sigmat_vec(j);%公式的解析解
            res_out = [res_out,eq_res];
        end
    end
end