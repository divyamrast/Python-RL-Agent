function [y, e] = mMeanStd(data, dim)
    Y = mean(data, dim);
    E = conf(data, dim);
    
    y = mean(Y);
    e = mean(E);
end