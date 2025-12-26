function result = LBH_xyz(L0B0H0,X0Y0Z0,LBH)
% 地球固定参数
 a=6378137.0;
 b=6356752.3142;
 e1=sqrt((a^2-b^2)/a^2);
 e2=sqrt((a^2-b^2)/b^2);
 
%将经纬坐标系转换到地心坐标系 
 L=LBH(1); %经
 B=LBH(2); %纬
 H=LBH(3); %高
 N=a/sqrt(1-e1^2*sind(B)^2);
 X=(N+H)*cosd(B)*cosd(L);
 Y=(N+H)*cosd(B)*sind(L);
 Z=(N*(1-e1^2)+H)*sind(B);
 XYZ=[X,Y,Z];
 
 
  %测量直角坐标系
 X0=X0Y0Z0(1);  %中央主楼作为中心站
 Y0=X0Y0Z0(2);
 Z0=X0Y0Z0(3);
 L0=L0B0H0(1);
 B0=L0B0H0(2);
 H0=L0B0H0(3);
 
 %将地心坐标系转换到以主楼为中心的xyz直角坐标系
 m1=-sind(B0)*cosd(L0);
 n1=-sind(B0)*sind(L0);
 l1=cosd(B0);
 m2=cosd(B0)*cosd(L0);
 n2=cosd(B0)*sind(L0);
 l2=sind(B0);
 m3=-sind(L0);
 n3=cosd(L0);
 l3=0;

 X=XYZ(1);Y=XYZ(2);Z=XYZ(3);
 x=[m1 n1 l1]*[X-X0 Y-Y0 Z-Z0]';
 y=[m2 n2 l2]*[X-X0 Y-Y0 Z-Z0]';
 z=[m3 n3 l3]*[X-X0 Y-Y0 Z-Z0]';  
 result=[z,x,y+H0];

 
end

