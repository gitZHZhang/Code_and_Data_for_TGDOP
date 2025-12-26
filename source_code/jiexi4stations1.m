function position1= jiexi4stations1(st,tdoa3,plot_en,height)
%四站定出三维位置

ylim([-5000,3000]);
xlim([-3000,3000]);
zlim([-1000 1000]);grid on;


x1=st(1,1);y1=st(1,2);z1=st(1,3);
x2=st(2,1);y2=st(2,2);z2=st(2,3);
x3=st(3,1);y3=st(3,2);z3=st(3,3);
x4=st(4,1);y4=st(4,2);z4=st(4,3);

K1=x1^2+y1^2+z1^2;K2=x2^2+y2^2+z2^2;K3=x3^2+y3^2+z3^2;K4=x4^2+y4^2+z4^2;
x21=x2-x1;x31=x3-x1;x41=x4-x1;
y21=y2-y1;y31=y3-y1;y41=y4-y1;
z21=z2-z1;z31=z3-z1;z41=z4-z1;

r21=tdoa3(1);
r31=tdoa3(2);
r41=tdoa3(3);

A=[x21,y21,z21;x31,y31,z31;x41,y41,z41];
det_A=x21*(y31*z41-z31*y41)-y21*(x31*z41-z31*x41)+z21*(x31*y41-y31*x41);
B11=y31*z41-y41*z31;   B21=-(x31*z41-x41*z31);B31=x31*y41-x41*y31;
B12=-(y21*z41-y41*z21);B22=x21*z41-x41*z21;   B32=-(x21*y41-x41*y21);
B13=y21*z31-y31*z21;   B23=-(x21*z31-x31*z21);B33=x21*y31-x31*y21;
B=-1/det_A*[B11,B12,B13;B21,B22,B23;B31,B32,B33];
a1=B(1,:)*[r21,r31,r41]';
b1=1/2*B(1,:)*[r21^2-K2+K1,r31^2-K3+K1,r41^2-K4+K1]';
a2=B(2,:)*[r21,r31,r41]';
b2=1/2*B(2,:)*[r21^2-K2+K1,r31^2-K3+K1,r41^2-K4+K1]';
a3=B(3,:)*[r21,r31,r41]';
b3=1/2*B(3,:)*[r21^2-K2+K1,r31^2-K3+K1,r41^2-K4+K1]';

A1=a1^2+a2^2+a3^2-1;
B1=2*a1*b1+2*a2*b2+2*a3*b3-2*a1*x1-2*a2*y1-2*a3*z1;
C1=b1^2+b2^2+b3^2-2*x1*b1-2*b2*y1-2*b3*z1+K1;

% position1=[0,0,0];   
%% 如果delta大于零
if B1^2-4*A1*C1>0
        r1_1=(-B1+sqrt(B1^2-4*A1*C1))/(2*A1);
        r1_2=(-B1-sqrt(B1^2-4*A1*C1))/(2*A1);


        xx1=a1*r1_1+b1;
        yy1=a2*r1_1+b2;
        zz1=a3*r1_1+b3+height;

        xx2=a1*r1_2+b1;
        yy2=a2*r1_2+b2;
        zz2=a3*r1_2+b3+height;

    %% 如果两个解都大于零
    if r1_1>0 && r1_2>0
%       plot3(xx1,yy1,zz1,'r.');hold on;
%       plot3(xx2,yy2,zz2,'m.');hold on;
         if plot_en==1
                  if zz1>height
                  plot3(xx1,yy1,zz1,'b.');hold on;
                  end
                  if zz2>height
                  plot3(xx2,yy2,zz2,'m.');hold on;
                  end
         end
          position1=[xx1,yy1,zz1];
%           if -1000<position1(1)&&position1(1)<1000 && -500<position1(2)&&position1(2)<2000 && -1000<position1(3)&&position1(3)<1000
%               11
%               -tdoa3/0.3
%           else
%               10
%               -tdoa3/0.3
%           end
%           position1=[xx1,yy1,zz1;xx2,yy2,zz2];
    end


 
    %% 如果有一个大于零的解
    if r1_1>0 && r1_2<0     
%         plot3(xx1,yy1,zz1,'g.');hold on;
            if plot_en==1
               plot3(xx1,yy1,zz1,'b.');hold on;
            end
            position1=[xx1,yy1,zz1];
%           if -1000<position1(1)&&position1(1)<1000 && -500<position1(2)&&position1(2)<2000 && -1000<position1(3)&&position1(3)<1000
%               21
%               -tdoa3/0.3
%           else
%               20
%               -tdoa3/0.3
%           end
    end
    %% 如果有一个大于零的解
    if r1_1<0 && r1_2>0
%         plot3(xx2,yy2,zz2,'g.');hold on;
                if plot_en==1
                   plot3(xx2,yy2,zz2,'m.');hold on;
                end
            position1=[xx2,yy2,zz2];
%           if -1000<position1(1)&&position1(1)<1000 && -500<position1(2)&&position1(2)<2000 && -1000<position1(3)&&position1(3)<1000
%               21
%               -tdoa3/0.3
%           else
%               20
%               -tdoa3/0.3
%           end
    end
    %% 如果没有大于零的解，函数返回值为（0，0，0）
    if r1_1<0 && r1_2<0
%        plot3(b1,b2,b3+height,'y.');hold on;
%        position1=[0,0,0];
       position1=[b1,b2,b3+height];
%           if -1000<position1(1)&&position1(1)<1000 && -500<position1(2)&&position1(2)<2000 && -1000<position1(3)&&position1(3)<1000
%               31
%               -tdoa3/0.3
%           else
%               30
%               -tdoa3/0.3
%           end
    end
end


 %% 如果delta小于零，函数返回值为（0，0，0）
if B1^2-4*A1*C1<=0
    if B1<=0
        r1_1=-B1/(2*A1);
        xx1=a1*r1_1+b1;
        yy1=a2*r1_1+b2;
        zz1=a3*r1_1+b3+height;
        plot3(xx1,yy1,zz1,'k.');hold on;
        position1=[xx1,yy1,zz1];
%           if -1000<position1(1)&&position1(1)<1000 && -500<position1(2)&&position1(2)<2000 && -1000<position1(3)&&position1(3)<1000
%               41
%               -tdoa3/0.3
%           else
%               40
%               -tdoa3/0.3
%           end
    else
        position1=[b1,b2,b3+height];
    end
    
end

end

