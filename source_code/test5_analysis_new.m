close all
clear 

fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;z_span = linspace(10,20,11);
G2 = zeros(length(x_span),length(y_span),length(z_span));
%% 传感器位置
L=30;
xt = 0;yt = 0;zt = 0;
x1 = L*cos(30*pi/180);y1 = L*sin(30*pi/180);z1=0.1;
x2 = L*cos(150*pi/180);y2 = L*sin(150*pi/180);z2 = 0.2;
x3 = 0;y3 = -L;z3 = 0.3;
sensor = [xt,yt,zt;x1,y1,z1;x2,y2,z2;x3,y3,z3];
% var_value = [sigmat1,sigmat2,sigmat3,sigmas,eta12,eta13,eta23];%变量列表，eta初始化为0，eta必须取常数而不是变量
real_value = [30e-3, 25e-3, 15e-3,5e-3, 0.2, 0.15, 0.1];%变量对应的真实值标签，可以看到eta并不是0
% real_value = [30e-3, 30e-3, 30e-3,5e-3, 0.2, 0.15, 0.1];%变量对应的真实值标签，可以看到eta并不是0
% real_valuecell = num2cell(real_value);
% [sigmat1_real,sigmat2_real,sigmat3_real,sigmas_real,eta12_real,eta13_real,eta23_real] = deal(real_valuecell{:});

err_list = [0.01,0.5,1,1.5,2,2.5,3,5,8,12,16];
SNR_LIST = [20,24,28,32,36,40,1e3];
fs = 10; %降采样倍数
x_span = -399:fs:400;y_span = -399:fs:400;z_span = linspace(10,20,11);



Mont_times = 50;
NAN_ratio = 0.99;
scene_list = {'scene1','scene2','scene3','scene4'};



%% 总数据分析3：四种场景and观测误差err对比(重构误差)
% scene_list = {'scene4'};
frob_list_scenes = cell(1,length(scene_list));
estimated_err_all_scenes = [];
for scene_i = 1:1:length(scene_list)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%遍历全部对比算法start
    scene = scene_list{scene_i};
    frob_list_all_Mont = []; %存储每次蒙特卡洛实验的重构误差
    load_path = ['test5_data/',scene,'/'];
    G3_real = load([load_path,'G3.mat']).G3;
    G3x_real = load([load_path,'G3_x.mat']).G3_x;
    G3y_real = load([load_path,'G3_y.mat']).G3_y;
    G3z_real = load([load_path,'G3_z.mat']).G3_z;
    estimated_err_all = [];
    for err_i = 1:1:length(err_list)
        err = err_list(err_i);
        SNR = SNR_LIST(end);
        load_file = [load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/'];
        res_all = load([load_file,'res_all.mat']).res_all;
        G3_cal = load([load_file,'G3.mat']).G3;
        G3x_cal = load([load_file,'G3_x.mat']).G3_x;
        G3y_cal = load([load_file,'G3_y.mat']).G3_y;
        G3z_cal = load([load_file,'G3_z.mat']).G3_z;

        incomplete_T = G3_real+err*randn(size(G3_real));%0.01km标准差的噪声
        incomplete_Tx = NaN(size(incomplete_T));%0.01km标准差的噪声
        incomplete_Ty = NaN(size(incomplete_T));%0.01km标准差的噪声
        incomplete_Tz = NaN(size(incomplete_T));%0.01km标准差的噪声

        ALL_ELE = 1:numel(incomplete_T);
        CHOSEN_IDX = randperm(numel(G2),round(NAN_ratio*numel(G2)));
        unselected_elements = setdiff(ALL_ELE, CHOSEN_IDX);
        incomplete_Tx(unselected_elements) = incomplete_T(unselected_elements).*G3x_cal(unselected_elements)./G3_cal(unselected_elements);
        incomplete_Ty(unselected_elements) = incomplete_T(unselected_elements).*G3y_cal(unselected_elements)./G3_cal(unselected_elements);
        incomplete_Tz(unselected_elements) = incomplete_T(unselected_elements).*G3z_cal(unselected_elements)./G3_cal(unselected_elements);
        % % estimated_err_all = [estimated_err_all,frob(G3_cal-G3_real),frob(G3x_cal-G3x_real),frob(G3y_cal-G3y_real),frob(G3z_cal-G3z_real)];
        % estimated_err_all = [estimated_err_all,frob(G3_cal-G3_real)];
        % BTD
        incomplete_T2x = fmt(incomplete_Tx);
        incomplete_T2y = fmt(incomplete_Ty);
        incomplete_T2z = fmt(incomplete_Tz);
        size_tens = incomplete_T2x.size;
        % size_tens = size(incomplete_T);
        L1 = [10 10 3];
        L2 = [10 10 3];
        L3 = [10 10 3];
        L1 = [5 5 3];
        L2 = [6 6 3];
        L3 = [7 7 3];
        model= struct;
        
        %% 列为m-1阶多项式初值的因子矩阵
        m=3;
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
        
        model.factorizations.mybtd1.data=incomplete_T2x;
        model.factorizations.mybtd1.btd={{1,2,3,4}};
        model.factorizations.mybtd2.data=incomplete_T2y;
        model.factorizations.mybtd2.btd={{5,6,7,8}};
        model.factorizations.mybtd3.data=incomplete_T2z;
        model.factorizations.mybtd3.btd={{9,10,11,12}};
        
        sdf_check(model,'print');
        [sol,output] = sdf_nls(model);
        [A1_res,B1_res,C1_res,S1_res,A2_res,B2_res,C2_res,S2_res,A3_res,B3_res,C3_res,S3_res] = deal(sol.factors{:});
        Sigma2_x_hat = tmprod(S1_res,{A1_res,B1_res,C1_res},1:3);
        Sigma2_y_hat = tmprod(S2_res,{A2_res,B2_res,C2_res},1:3);
        Sigma2_z_hat = tmprod(S3_res,{A3_res,B3_res,C3_res},1:3);
        recon_tensor = Sigma2_x_hat + Sigma2_y_hat + Sigma2_z_hat;
        frob_list1 = [];
        frob_listx = [];
        frob_listy = [];
        frob_listz = [];
        for i = 1:1:size(recon_tensor,3)
            frob_list1 = [frob_list1,frob(recon_tensor(:,:,i)-G3_real(:,:,i))];
            frob_listx = [frob_listx,frob(Sigma2_x_hat(:,:,i)-G3x_cal(:,:,i))];
            frob_listy = [frob_listy,frob(Sigma2_y_hat(:,:,i)-G3y_cal(:,:,i))];
            frob_listz = [frob_listz,frob(Sigma2_z_hat(:,:,i)-G3z_cal(:,:,i))];
        end
        frob_list.errG = frob_list1;
        frob_list.errGx = frob_listx;
        frob_list.errGy = frob_listy;
        frob_list.errGz = frob_listz;
        recon_T.G = recon_tensor;
        recon_T.Gx = Sigma2_x_hat;
        recon_T.Gy = Sigma2_y_hat;
        recon_T.Gz = Sigma2_z_hat;
        save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/frob_list.mat'],'frob_list')
        save([load_path,'err_',num2str(err),'_SNR_',num2str(SNR),'/recon_T.mat'],'recon_T')
    end
    estimated_err_all_scenes = [estimated_err_all_scenes;frob_list];
end
figure
plot(err_list,estimated_err_all_scenes,'*-')
grid on
xlabel('Standard deviation of observation error \underline{\textbf{N}}.','interpreter','latex')
ylabel('Reconstruction error.')
legend('scene1','scene2','scene3','scene4')



