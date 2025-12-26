经典GDOP可以假设误差源稳定，然后通过变换基站构型来寻找最优基站部署方式
但GDOP方法对类似于多径的空间特性难以建模，比如即使引入了多径误差作为误差源，基站部署发生改变时，不仅仅是G变了，误差源中的多径误差也会发生改变，这是所提方法的优势.
test4：张量重构（初步尝试，用处不大）
test5：特定场景下定位误差求解（误差模型已知时求解析解的四种场景，现已删除这部分内容）
test6：文章所提公式的验证（验证论文中转置模n积及其展开式）
*test7：所提算法的应用（根据协方差矩阵得到误差椭圆，并定位增强和评估定位置信度）
test8：所提算法与对比算法的对比（所提算法在S_tdoa1的性能对比，后续扩展到三种场景）
test9：实际数据测试所提算法（用实际数据检验所提算法）
*test10:实际数据不同算法对比（test9的扩展，比较了所提算法和对比算法在不同训练集比例情况下的性能）
*test11:比较了不同定位体制和场景下所提算法的性能（test8扩展到三种场景的对比：DOA TDOA1 TDOA2）
*test12：比较了不同定位体制和场景下所提算法方向分量重建的性能（将test11中G的重建转变成对方向分量GxGyGz的重建，Proposed和test11不同，其余的基本类似）
*test13：分析了张量的低秩特性
test14：绘画场景的文件
*test15：扩充实验场景，TDOA的观测值引入nlos误差
*test16：在test15场景的基础上，将重建改为针对方向分量的重建
注意：test16和test12只跑了proposed的结果，其余的方法代码还没更改，计算Frob时还没有取根号，要先取根号再对比
*test17：在TDOA的不含nlos误差的场景下，探究采样策略、张量大小对于重建精度的影响
*test18：在test10的基础上进一步扩展，探究四轨实测数据的重建效果
*test19：探究所提算法和GeneralBTD算法的收敛特性

注意我们用的性能指标是MFNE，它的M是关于MC实验的mean，而不是张量元素的mean
当mc=1时，MFNE = frob(G-G')，这里的frob是F范数。假设N*N*N的张量每一个元素都为x，
那么frob(X)=sqrt(N^3 * x^2)=N^(3/2)*x，显然它是N和x的函数，不能用它表征和理解定位误差的绝对值。

主要用到的函数：

test11_doa的NaNratio对比和SNR对比：test11_doa.m test11_doa_snr.m
test11_tdoa的NaNratio对比和SNR对比：test11_tdoa1.m test11_tdoa1_snr.m
性能分析：test11_analysis.m

存在多方向观测量时的重建：test12_doa.m test12_tdoa1.m
性能分析：test12_analysis.m


test15_tdoa的NaNratio对比和SNR对比：test15_tdoa1.m test15_tdoa1_snr.m
性能分析：test15_analysis.m

NLOS下存在多方向观测量时的重建：test16_tdoa1.m 
性能分析：test16_analysis.m

实际数据分析：
test10.m 所提算法和对比算法在实际数据中的表现
test10_only_proposed.m 从上面的文件中提取出proposed，仅研究所提算法
性能分析：test10_analysis.m


