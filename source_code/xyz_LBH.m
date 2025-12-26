function result = xyz_LBH(L0B0H0,X0Y0Z0,xyz)
 % 地球固定参数
 a=6378137.0;
 b=6356752.3142;
 e1=sqrt((a^2-b^2)/a^2);
 e2=sqrt((a^2-b^2)/b^2);
 
 
 
  % 测量直角坐标系转大地直角坐标系

  L0=L0B0H0(1);   
  B0=L0B0H0(2);%中央主楼的大地坐标系
  H0=L0B0H0(3);
  
  X0=X0Y0Z0(1);
  Y0=X0Y0Z0(2);
  Z0=X0Y0Z0(3);    %中央主楼的大地直角坐标系
  
  m1=-sind(B0)*cosd(L0);
  n1=cosd(B0)*cosd(L0);
  l1=-sind(L0);
  m2=-sind(B0)*sind(L0);
  n2=cosd(B0)*sind(L0);
  l2=cosd(L0);
  m3=cosd(B0);
  n3=sind(B0);
  l3=0;
      x=xyz(2);y=xyz(3);z=xyz(1);
      X=[m1 n1 l1]*[x y z]'+X0;
      Y=[m2 n2 l2]*[x y z]'+Y0;
      Z=[m3 n3 l3]*[x y z]'+Z0;
      XYZ=[X Y Z];  %中央主楼的大地直角坐标系
  % 大地直角坐标系转大地坐标系

      X=XYZ(1);Y=XYZ(2);Z=XYZ(3);
      theta=atand(a*Z/(b*sqrt(X^2+Y^2)));
      L=180+atand(Y/X);
      B=atand((Z+b*e2^2*sind(theta)^3)/(sqrt(X^2+Y^2)-a*e1^2*cosd(theta)^3));
      N=a/sqrt(1-e1^2*sind(B)^2);
      H=sqrt(X^2+Y^2)/cosd(B)-N;
      result=[L,B,H];

end

