function [sol,output] = runBTD(tensor,MLrank_mat,poly,type)
size_tens = tensor.size;
L1 = MLrank_mat(1,:);
L2 = MLrank_mat(2,:);
L3 = MLrank_mat(3,:);
model= struct;
% if(NAN_ratio<0)
if strcmp(type,'typicalBTD_nonneg')
    %% 随机初值
    model.variables.A1=randn(size_tens(1),L1(1));
    model.variables.B1=randn(size_tens(2),L1(2));
    model.variables.C1=randn(size_tens(3),L1(3));
    model.variables.S1=randn(L1(1),L1(2),L1(3));
    model.variables.A2=randn(size_tens(1),L2(1));
    model.variables.B2=randn(size_tens(2),L2(2));
    model.variables.C2=randn(size_tens(3),L2(3));
    model.variables.S2=randn(L2(1),L2(2),L2(3));
    model.variables.A3=randn(size_tens(1),L3(1));
    model.variables.B3=randn(size_tens(2),L3(2));
    model.variables.C3=randn(size_tens(3),L3(3));
    model.variables.S3=randn(L3(1),L3(2),L3(3));
    model.factors={ {'A1',@struct_nonneg},...
        {'B1', @struct_nonneg},...
        {'C1', @struct_nonneg},'S1',...
        {'A2', @struct_nonneg},...
        {'B2', @struct_nonneg},...
        {'C2', @struct_nonneg},'S2',...
        {'A3', @struct_nonneg},...
        {'B3', @struct_nonneg},...
        {'C3', @struct_nonneg},'S3' };
elseif strcmp(type,'typicalBTD')
    model.variables.A1=randn(size_tens(1),L1(1));
    model.variables.B1=randn(size_tens(2),L1(2));
    model.variables.C1=randn(size_tens(3),L1(3));
    model.variables.S1=randn(L1(1),L1(2),L1(3));
    model.variables.A2=randn(size_tens(1),L2(1));
    model.variables.B2=randn(size_tens(2),L2(2));
    model.variables.C2=randn(size_tens(3),L2(3));
    model.variables.S2=randn(L2(1),L2(2),L2(3));
    model.variables.A3=randn(size_tens(1),L3(1));
    model.variables.B3=randn(size_tens(2),L3(2));
    model.variables.C3=randn(size_tens(3),L3(3));
    model.variables.S3=randn(L3(1),L3(2),L3(3));
    model.factors={'A1','B1','C1','S1','A2','B2','C2','S2','A3','B3','C3','S3'};
elseif strcmp(type,'polyBTD_nonneg')
    m=poly;
    model.variables.A1=randn(L1(1),m);
    model.variables.B1=randn(L1(2),m);
    model.variables.C1=randn(L1(3),m);
    model.variables.S1=randn(L1(1),L1(2),L1(3));
    model.variables.A2=randn(L2(1),m);
    model.variables.B2=randn(L2(2),m);
    model.variables.C2=randn(L2(3),m);
    model.variables.S2=randn(L2(1),L2(2),L2(3));
    model.variables.A3=randn(L3(1),m);
    model.variables.B3=randn(L3(2),m);
    model.variables.C3=randn(L3(3),m);
    model.variables.S3=randn(L3(1),L3(2),L3(3));
    t1=1:1:size_tens(1);
    t2=1:1:size_tens(3);
    model.factors={ {'A1',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
        {'B1',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
        {'C1',  @(z,task) struct_poly(z,task,t2),@struct_nonneg},...
        {'S1',@struct_nonneg},...
        {'A2',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
        {'B2',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
        {'C2',  @(z,task) struct_poly(z,task,t2),@struct_nonneg},...
        {'S2',@struct_nonneg},...
        {'A3',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
        {'B3',  @(z,task) struct_poly(z,task,t1),@struct_nonneg},...
        {'C3',  @(z,task) struct_poly(z,task,t2),@struct_nonneg},...
        {'S3',@struct_nonneg} };
elseif strcmp(type,'polyBTD')
    m=poly;
    model.variables.A1=randn(L1(1),m);
    model.variables.B1=randn(L1(2),m);
    model.variables.C1=randn(L1(3),m);
    model.variables.S1=randn(L1(1),L1(2),L1(3));
    model.variables.A2=randn(L2(1),m);
    model.variables.B2=randn(L2(2),m);
    model.variables.C2=randn(L2(3),m);
    model.variables.S2=randn(L2(1),L2(2),L2(3));
    model.variables.A3=randn(L3(1),m);
    model.variables.B3=randn(L3(2),m);
    model.variables.C3=randn(L3(3),m);
    model.variables.S3=randn(L3(1),L3(2),L3(3));
    t1=1:1:size_tens(1);
    t2=1:1:size_tens(3);
    model.factors={ {'A1',  @(z,task) struct_poly(z,task,t1)},...
        {'B1',  @(z,task) struct_poly(z,task,t1)},...
        {'C1',  @(z,task) struct_poly(z,task,t2)},...
        {'S1'},...
        {'A2',  @(z,task) struct_poly(z,task,t1),},...
        {'B2',  @(z,task) struct_poly(z,task,t1)},...
        {'C2',  @(z,task) struct_poly(z,task,t2)},...
        {'S2'},...
        {'A3',  @(z,task) struct_poly(z,task,t1)},...
        {'B3',  @(z,task) struct_poly(z,task,t1)},...
        {'C3',  @(z,task) struct_poly(z,task,t2)},...
        {'S3'} };
end
model.factorizations.mybtd.data=tensor;
model.factorizations.mybtd.btd={{1,2,3,4},{5,6,7,8},{9,10,11,12}};
sdf_check(model,'print');
[sol,output] = sdf_nls(model);