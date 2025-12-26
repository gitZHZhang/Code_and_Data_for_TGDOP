function lowerIndex = transform_value_to_index(sortedVector,targetValue )
% sortedVector由小到大排序
% 初始化索引

upperIndex = 0;
% 遍历向量找到目标值所在的范围
for i = 1:length(sortedVector) - 1
    if sortedVector(i) <= targetValue && targetValue < sortedVector(i + 1)
        lowerIndex = i;
        upperIndex = i + 1;
        break;
    end
    if upperIndex == 0
        upperIndex = length(sortedVector);
        lowerIndex = upperIndex - 1;
    end
end

end