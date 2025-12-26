function [ax1,c] = colorbarn(tick,color,mode)
%---------------------------------
%带尖端的colorbar
%[ax1,c] = COLORBARN(tick,color,mode)
%坐标轴句柄是c,绘图窗句柄是ax1
%详情请参考函数 [ax1,c,h]=contf_line(lon,lat,X,tick,color,mode)
%---------------------------------
[m,~]=size(color);
len_tick=length(tick);
if len_tick-m ~= 1
    return
end
figure;
set(gcf,'color','w');
if isequal(mode,'h')
    c=axes('position',[0.1 0.1 0.8 0.03]);
    cdata=zeros(1,m);
    for i=1:m
        cdata(i)=(tick(i)+tick(i+1))/2;
    end
    axis([0 m 0 1])
    line([0 1],[0 0],'linewidth',2,'color','w','parent',c)
    line([m-1 m],[0 0],'linewidth',2,'color','w','parent',c)
    colormap(color);
    %------------------------
    pat_v1=[1 0;1 1;2 1;2 0];pat_f1=reshape(1:(m-2)*4,4,m-2)';
    for j=2:m-2
        pat_v1=[pat_v1;[j 0;j 1;j+1 1;j+1 0]];
    end
    pat_col1=[cdata(2:end-1)]';
    patch('Faces',pat_f1,'Vertices',pat_v1,'FaceVertexCData',pat_col1,'FaceColor','flat');
    %--------------------------------------------------   
    pat_v2=[0 0.5;1 0;1 1;m-1 1;m-1 0;m 0.5];
    pat_f2=[1 2 3;4 5 6];
    pat_col2=[cdata(1);cdata(end)];
    patch('Faces',pat_f2,'Vertices',pat_v2,'FaceVertexCData',pat_col2,'FaceColor','flat');
    %---------------------------------------------------------
    set(c,'color','none','xcolor','k','ycolor','none');
    box off
%     line([1 m-1],[0 0],'color','k')
%     line([1 m-1],[1 1],'color','k')
%     for i=2:m-2
%         line([i i],[0 1],'color','k')
%     end
    set(c,'xtick',1:m-1,'xticklabel',num2cell(tick(2:end-1)),'ytick',[])
    ax1=axes('position',[0.1 0.2 0.8 0.7]);
elseif isequal(mode,'v')
    c=axes('position',[0.87 0.11 0.02 0.81]);
    set(c,'yaxislocation','right');
    cdata=zeros(1,m);
    for i=1:m
        cdata(i)=(tick(i)+tick(i+1))/2;
    end
    axis([0 1 0 m])
    line([1 1],[0 1],'linewidth',2,'color','w','parent',c)
    line([1 1],[m-1 m],'linewidth',2,'color','w','parent',c)
    colormap(color);
    %--------------------------------------------
    pat_v1=[0 1;1 1;1 2;0 2];pat_f1=reshape(1:(m-2)*4,4,m-2)';
    for j=2:m-2
        pat_v1=[pat_v1;[0 j;1 j;1 j+1;0 j+1]];
    end
    pat_col1=[cdata(2:end-1)]';
    patch('Faces',pat_f1,'Vertices',pat_v1,'FaceVertexCData',pat_col1,'FaceColor','flat');
    %--------------------------------------------------   
    pat_v2=[0.5 0;1 1;0 1;1 m-1;0.5 m;0 m-1];
    pat_f2=[1 2 3;4 5 6];
    pat_col2=[cdata(1);cdata(end)];
    patch('Faces',pat_f2,'Vertices',pat_v2,'FaceVertexCData',pat_col2,'FaceColor','flat');
    %------------------------------------------------------
    set(c,'color','none','xcolor','none','ycolor','k');
    box off
    set(c,'ytick',1:m-1,'yticklabel',num2cell(tick(2:end-1)),'xtick',[])
    ax1=axes('position',[0.11 0.11 0.74 0.81]);
else
    disp('colorbarn格式出错')
    return
end

end


