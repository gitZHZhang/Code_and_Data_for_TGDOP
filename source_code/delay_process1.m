function delay_second = delay_process1(delay_slices,piece_num)

frequency_num = length( delay_slices)/piece_num;
delay_second =  cell(1,frequency_num);
for i=1:1:frequency_num
    slices_len=[];
    slice_i = [];
    for j=i:frequency_num:frequency_num*piece_num
        slices_len = [slices_len,length(delay_slices{j})];
    end
    slices_len((slices_len==0))=[];
    if isempty(slices_len)
        delay_second{i}=[];
    else
        slices_min_len = min(slices_len);
        for k=i:frequency_num:frequency_num*piece_num
            if ~isempty(delay_slices{k})
                slice_i = [slice_i;delay_slices{k}(1:slices_min_len)];
            end
        end
        slice_i_mode = mode(slice_i,1);
        delay_second{i} = slice_i_mode;
    end
end