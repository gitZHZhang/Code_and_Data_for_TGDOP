function tdoa = cal_tdoa(emitter,sensor,real_value,N)
%N表示时差的样本数
c = 0.3;%光速，单位km/us
m = emitter(1);n = emitter(2);p = emitter(3);
x_row = num2cell(sensor(:,1));y_col = num2cell(sensor(:,2));z_dim = num2cell(sensor(:,3));
[xt,x1,x2,x3] = deal(x_row{:});
[yt,y1,y2,y3] = deal(y_col{:});
[zt,z1,z2,z3] = deal(z_dim{:});
real_value = num2cell(real_value);
cor_noise = cor_noise_gen(real_value,N);
tdoa_t1 = sqrt((xt-m)^2+(yt-n)^2+(zt-p)^2)/c-sqrt((x1-m)^2+(y1-n)^2+(z1-p)^2)/c;
tdoa_t1 = tdoa_t1 + cor_noise(:,1);
tdoa_t2 = sqrt((xt-m)^2+(yt-n)^2+(zt-p)^2)/c-sqrt((x2-m)^2+(y2-n)^2+(z2-p)^2)/c;
tdoa_t2 = tdoa_t2 + cor_noise(:,2);
tdoa_t3 = sqrt((xt-m)^2+(yt-n)^2+(zt-p)^2)/c-sqrt((x3-m)^2+(y3-n)^2+(z3-p)^2)/c;
tdoa_t3 = tdoa_t3 + cor_noise(:,3);
tdoa = [tdoa_t1,tdoa_t2,tdoa_t3];

