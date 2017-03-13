function e = conf(data, dim)
    
    if nargin < 2
        dim = 1;
        if size(data, 1) < size(data, 2)
            data = data';
        end
    end

    n = size(data, dim);
    s = std(data, 0, dim);
    cutoff = abs(tinv(0.05/2, n-1)); % tinv is a one-sided function
    e = cutoff * s / sqrt(n);

end