function position= position_process(position_set,function_type,i)
% origin是原始代码 average是对十个分片求平均 all是将十个分片结果全部考虑
position=0;
if strcmp(function_type,'origin')
    position = [i,position_set(end,:)];
    
elseif strcmp(function_type,'average')
    if ~isempty(position_set)
        if size(position_set,1)==1
            position=(position_set);
        else
            position=mean(position_set);
        end
    end
elseif strcmp(function_type,'all')
    position = position_set;
elseif strcmp(function_type,'all2')
    position = [i*ones(length(position_set(:,1)),1),position_set];
end