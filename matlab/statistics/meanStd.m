function [Y, E] = meanStd(data, dim)

    if nargin < 2
        dim = 1;
        if size(data, 1) < size(data, 2)
            data = data';
        end
    end
    
    Y = mean(data, dim);
    E = std(data, [], dim);
end