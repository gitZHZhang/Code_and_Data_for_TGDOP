clear; 
clc; 
close all;
%% 纵轴截断后图像
% 数据
data=[2.3   2.1 1.9 1.8 1.7 1.7 1.7 1.7 1.7 1
3   2.5 2.1 2   1.9 1.9 1.9 1.9 1.9 1.2
3.4 3.3 3.2 3.1 3   3   3   3   3   2.5
10.8    10.6    10.5    10.2    10.1    10.1    10.1    10.1    10.1    9.1
];
%% 参数设置
x_min=0.1; %横坐标刻度最小值
x_interval=0.1; %横坐标刻度间隔距离
x_max=1; %横坐标刻度最大值
y_interval=1;  %纵坐标两个刻度间隔距离
y_max=11; %纵轴刻度最大值
y_break_start=4; % 截断的开始值
y_break_end=9; % 截断的结束值
 
X=x_min:x_interval:x_max;
adjust_value=0.4*y_interval; %微调截断处y坐标
uptate_num=y_break_end-y_break_start-y_interval; %最高处曲线向下平移大小
 
% 超过截断结束位置的那些曲线统统向下平移uptate_num个长度
for i=1:length(data(:, 1))
    if data(i, :)>y_break_end
        data(i, :)=data(i, :)-uptate_num;
    end
end
 
%% 绘图
% 根据曲线的个数进行修改，这里曲线是4条
h=plot(X, data(1, :), 'k*-', X, data(2, :), 'g^-', X, data(3, :), 'r-s', X, data(4, :), 'b-x', 'MarkerFaceColor','y', 'MarkerSize',7,'linewidth',1.5);
 
set(gcf,'color','w') %后面背景变白
xlim([x_min x_max]); %横坐标范围
xlabel('X');
ylabel('Y');
hl = legend('Line-1', 'Line-2', 'Line-3', 'Line-4', 'Location', 'east');  %图例
set(hl,'Box','off','location','NorthOutside','NumColumns',4);
 
% 纵坐标截断设置
ylimit=get(gca,'ylim');
location_Y=(y_break_start+adjust_value-ylimit(1))/diff(ylimit);
t1=text(0, location_Y,'//','sc','BackgroundColor','w','margin',eps, 'fontsize',13);
set(t1,'rotation',90);
t2=text(1, location_Y,'//','sc','BackgroundColor','w','margin',eps, 'fontsize',13);
set(t2,'rotation',90);
 
% 重新定义纵坐标刻度
ytick=0:y_interval:y_max;
set(gca,'ytick',ytick);
ytick(ytick>y_break_start+eps)=ytick(ytick>y_break_start+eps)+uptate_num;
for i=1:length(ytick)
   yticklabel{i}=sprintf('%d',ytick(i));
end
set(gca,'yTickLabel', yticklabel, 'FontSize', 12, 'FontName', 'Times New Roman'); %修改坐标名称、字体
text( 'string',"(a) ", 'Units','normalized','position',[0.05,0.95],  'FontSize',14,'FontWeight','Bold','FontName','Times New Roman');   
set(gca,'Layer','top','FontSize',12,'Fontname', 'Times New Roman');
saveas(gcf,sprintf('Break_Y_Axis.jpg'),'bmp'); %保存图片