function [estimated_pos,loss] = PerformanceEnhancementWithMAP(measured_pos,real_pos,sensor,vars_real,Pe)
loss = [];
n = size(measured_pos,1);
for grid_x = real_pos(1)-150:1:real_pos(1)+150
    for grid_y = real_pos(2)-150:1:real_pos(2)+150
        P_i = cal_diff_P([grid_x,grid_y,500],sensor,vars_real);
        P_i = P_i(1:2,1:2);
        % sum_term = 0;
        % for i=1:1:n
        %     u_i = measured_pos(i,:);
        %     p_u_i = 1/2/pi/sqrt(det(P_i))*exp(-1/2*(u_i-[grid_x,grid_y])*pinv(P_i)*(u_i-[grid_x,grid_y])');
        %     sum_term = sum_term + log(p_u_i);
        % end
        % Map = exp(-1/2*sum_term);
        % loss_i = -n*log(2*pi)-n/2*log(det(P_i))+log(Map);
        % [gdop,~,~,P_i] = cal_gdop([grid_x,grid_y],sensor);
        [~,theta,sigma1_2,sigma2_2]=Draw_err_ellipse(P_i,Pe,[grid_x,grid_y]);
        sum_term = 0;
        for i=1:1:n
            pos_i = measured_pos(i,:);
            sum_term = sum_term + ((pos_i(1)-grid_x)*cos(theta)+(pos_i(2)-grid_y)*sin(theta))^2/sigma1_2 + ...
                (-(pos_i(1)-grid_x)*sin(theta)+(pos_i(2)-grid_y)*cos(theta))^2/sigma2_2;
        end
        loss_i = -n*log(2*pi*sqrt(sigma1_2)*sqrt(sigma2_2))-1/2*sum_term;
        loss = [loss;grid_x,grid_y,loss_i];
    end
end
loss(:,3) = exp(loss(:,3));
[a,idx]=max(loss(:,3));
estimated_pos = loss(idx,1:2);