function [m, e] = perfseries(varargin)
%PERFSERIES Series end performance
%   [M, E] = PERFSERIES(PATH, ...) returns the performance over the last
%   10% of the series read from PATH. Extra arguments are passed to
%   READSERIES.
%
%   EXAMPLE:
%      t = perfseries('TLMLearn_SEQ_LLR_*.txt');
%
%   SEE ALSO:
%      readseries
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    data = readseries(varargin{:});
    [x, y, d, e] = avgseries(data);
    
    s = round(length(y)/10);
    
    m = mean(y(end-s:end));
    e = mean(e(end-s:end));

end
