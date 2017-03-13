function [y, e] = estimateEndPerformance(data, dim)
    E = conf(data, dim);
    M = mean(data, dim);
    
    lowerBound = min(M - E);
    upperBound = max(M + E);
    
    y = mean(M);
    e = max([upperBound-y y-lowerBound]);
end