function [plot_3d_matrix] = plot_3D_tensor(tensor,x_span,y_span,z_span,plot_title)
plot_3d_matrix = zeros(size(tensor,1)*size(tensor,2)*size(tensor,3),3);
count=0; 
for i=1:1:size(tensor,1)
    for j=1:1:size(tensor,2)
        for k=1:1:size(tensor,3)
            count = count + 1;
            plot_3d_matrix(count,1:4)=[x_span(i),y_span(j),z_span(k),tensor(i,j,k)];
        end
    end
end   
figure
scatter3(plot_3d_matrix(:,1),plot_3d_matrix(:,2),plot_3d_matrix(:,3),10,plot_3d_matrix(:,4),'filled');%此时散点大小不能缺省；使用filled属性填充
colorbar
xlabel('x(km)')
ylabel('y(km)')
zlabel('z(km)')
title(plot_title,'interpreter','latex')