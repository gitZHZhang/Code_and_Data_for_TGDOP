function Plot_P(P_list,Pe,Legend,line_styles)
figure
for i=1:1:length(P_list)
    P = P_list{i};
    [f1,~,~,~]=Draw_err_ellipse(P,Pe,[0,0]);
    fimplicit(f1,[-10,10,-10,10],'LineStyle',line_styles{4-mod(i,4)});hold on
end
legend(Legend)