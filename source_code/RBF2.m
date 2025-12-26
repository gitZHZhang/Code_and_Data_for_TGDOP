function z=RBF2(x0,y0,z0,x,y,Method,Rs,Npoly,ExtrapMethod,Error)
%二维RBF插值，输入x0，y0散点，z0值。x，y是输出插值得到的点。
%Method方法，默认'linear'。
%'linear',|R|
%'gaussian',exp(-(R/Rs)^2)
%'thin_plate',R^2*log(R)
%'linearEpsR',|R|。分段线性插值，要求s更大一些
%'cubic',|R^3|

%Rs，插值核作用半径。Rs对于'linear'，'cubic'，'thin_plate'无影响。Rs大概和点和点之间的距离差不多就行。
%Npoly多项式拟合。默认是1，只拟合1次项。
%ExtrapMethod,外插方法。某个数，全部赋值为这个数。'nearest'，按照临近值外插。'ploy'，多项式外插。'rbf'，默认用RBF插值得到的值。
%Error误差。在(-∞,∞)区间。默认是0，无误差。表示可以在部分误差范围内去插值。
%Error一般在0~1之内。如果只是为了收敛，可以比较小，在0.1以内就可以有很好效果；如果想实现平滑，可适当增大，大于1。

%示例：

N=numel(x0);%点的数量，也是RBF核的数量
%整理为列向量
x0=x0(:);
y0=y0(:);
z0=z0(:);
x=x(:);
y=y(:);
%计算包线
[k_ch,V]=convhull(x0,y0);
if V<=0
    warning('输入散点过于集中');
end
%函数默认信息
narginchk(5,10)
if nargin<7 || isempty(Method)
    Method='linear';%默认线性核函数
elseif nargin<8 || isempty(Rs)
    Rs=1.1*(max(x0)-min(x0))/N;%假设空间均匀分布
elseif nargin<9 || isempty(Npoly)
    Npoly=1;%默认拟合1次项
elseif nargin<10 || isempty(Npoly)
    ExtrapMethod='rbf';%默认不外插
elseif nargin<11 || isempty(Error)
    Error=0;%默认输入数据没有误差
end

%选择核函数
switch Method
    case 'linear'
        fun=@(RMat) Kernel_Linear(RMat,Rs);
    case 'gaussian'
        fun=@(RMat) Kernel_Gaussian(RMat,Rs);
        Error=-Error;%因为零点不是零，远点是0，所以这里误差为负
    case 'thin_plate'
        fun=@(RMat) Kernel_Thin_plate(RMat,Rs);
    case 'linearEpsR'
        fun=@(RMat) Kernel_LinearEpsR(RMat,Rs);
        Error=-Error;%因为零点不是零，远点是0，所以这里误差为负
end

%将原始数据分离出多项式项Npoly
if Npoly==0
    C=ones(N,1);%常数项
    wC=mean(z0);
elseif Npoly==1
    if N<2;warning('点数太少，建议Npoly等于0');end
    C=[ones(N,1),x0,y0];%常数项+一次项
    wC=C\z0;
elseif Npoly==2
    if N<5;warning('点数太少，建议Npoly等于1');end
    C=[ones(N,1),x0,y0,x0.^2,y0.^2,x0.*y0];%常数项+一次项+二次项
    wC=C\z0;
else
    error('只支持Npoly=0,1,2')
end
zC=C*wC;%多项式项
z1=z0-zC;

%计算距离矩阵
DisMat=squareform(pdist([x0,y0]));
K3=fun(DisMat);%每一个核函数的中心点在节点上
K3=K3-Error*eye(N);%把误差函数加入

w=K3\z1;%计算权重

%根据权重计算插值
z=zeros(size(x));
for k=1:N %计算每个核的贡献，然后叠加
    R_k=sqrt((x-x0(k)).^2+(y-y0(k)).^2);
    zt=w(k)*fun(R_k );
    z=z+zt;
end

NOut=length(z);
%再还原回多项式项
if Npoly==0
    C=ones(NOut,1);%常数项
elseif Npoly==1
    C=[ones(NOut,1),x,y];%常数项+一次项
elseif Npoly==2
    C=[ones(NOut,1),x,y,x.^2,y.^2,x.*y];%常数项+一次项+二次项
end
zC2=C*wC;%再加上多项式项

%判断外插方法
Inpoly=inpolygon(x,y,x0(k_ch),y0(k_ch));%多边形内
indxInpoly=find(Inpoly);
if isnumeric(ExtrapMethod)
    z(Inpoly)=z(Inpoly)+zC1(Inpoly);%内部的正常
    z(~Inpoly)=ExtrapMethod;%外部的固定值
elseif ischar(ExtrapMethod)
    switch ExtrapMethod
        case 'rbf'
            z=z+zC2;%再加上多项式项
        case 'ploy'
            z(Inpoly)=z(Inpoly)+zC2(Inpoly);%内部的正常
            z(~Inpoly)=zC2(~Inpoly);%外部的直接用多项式值
        case 'nearest'
            z(Inpoly)=z(Inpoly)+zC2(Inpoly);%内部的正常
            %找到最近的点
            F = scatteredInterpolant(x(Inpoly),y(Inpoly),z(Inpoly),'nearest','nearest');
            %外部的直接插值
            z(~Inpoly)=F(x(~Inpoly),y(~Inpoly));
    end
else
    error('未识别ExtrapMethod')
end

%检查结果
if max(abs(w))/max(abs(z1))>1e3
    warning('结果未收敛，建议调整误差Error');
elseif (max(z)-min(z))>5*(max(z0)-min(z0))
    warning('结果未收敛，建议增大区间Eps，或者调整误差Error');
end

end

function z=Kernel_Linear(R,s)
%R距离
z=abs(R);%线性函数
end

function z=Kernel_Gaussian(R,s)
%R距离
%s大概和采样点间距差不多就行，可以略大（更胖）
z=exp(-R.^2/2/s^2);%正态函数
end

function z=Kernel_Thin_plate(R,s)
%R距离
z=R.^2.*log(R);%thin_plate
end

function z=Kernel_LinearEpsR(R,s)
%R距离
%s大概和采样点间距差不多就行，可以略大（更胖）
s=2*s;%这里要更大
z=s-R;%线性函数
z(z<0)=0;
end
