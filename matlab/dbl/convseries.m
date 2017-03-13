function [t, e, te, ee] = convseries(val, varargin)
%CONVSERIES Series convergence
%   [T, E] = CONVSERIES(VAL, PATH, ...) returns the time point at which
%   the average of the series read from PATH crosses VAL. Extra arguments
%   are passed to READSERIES.
%
%   EXAMPLE:
%      t = convseries(100, 'TLMLearn_SEQ_LLR_*.txt');
%
%   SEE ALSO:
%      readseries
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    data = readseries(varargin{:});
    
    c = zeros(1, length(data));
    p = zeros(1, length(data));
    
    for ii = 1:length(data)
        ii;
        d = data{ii};
        c(ii) = d(find(cumsumr(d(:, 2)>val) == 3, 1, 'first')-2, 1);
        
        if size(d,1) >= 20
            p(ii) = mean(d(end-10:end, 2));
        else
            p(ii) = mean(d(end-5:end, 2));
        end
    end
    
    t = mean(c);
    te = 1.96*std(c)/sqrt(length(c));
    e = mean(p);
    ee = 1.96*std(p)/sqrt(length(p));
    
    
    %[x, y] = avgseries(data, 5);
    
    
    
%    t = fzerodata(x, y'-val);
    %[~, avg] = avgseries(data);
    %e = mean(avg(end-10:end));

end
