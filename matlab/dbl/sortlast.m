function [sorted, I] = sortlast(strs)
%SORTLAST Sort strings based on last number
%   SORTED = SORTLAST(STRINGS) sorts the cell array STRINGS according
%   to the last number in it.
%   [SORTED, I] = SORTLAST(STRINGS) additionally returns the index array,
%   such that SORTED = STRINGS(I).
%
%   Author:
%      Wouter Caarls <w.caarls@tudelft.nl>

    n = length(strs);
    nums = zeros(n, 1);
    
    for ii = 1:n
        num = '';
        for jj=numel(strs{ii}):-1:1
            if (strs{ii}(jj) >= '0' && strs{ii}(jj) <= '9') || strs{ii}(jj) == '.'
                num = [strs{ii}(jj) num];
            elseif ~isempty(num)
                break
            end
        end
        
        if ~isempty(num)
            nums(ii) = str2double(num);
        end
    end
    
    [~, I] = sort(nums);
    sorted = strs(I);

end

