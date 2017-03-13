function y = movavg(x, wdw)
%MOVAVG Moving average filter.
%   Y = MOVAVG(X, WDW) filters the (1 or 2-dimensional) matrix X with a
%   moving average filter. The default window size is 3.
% 
%   NOTE:
%      For extrapolation purposes, the first and last values are
%      repeated.
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    if nargin < 2
        wdw = 3;
    end

    if size(x, 1) > 1
        x = [repmat(x(1, :), wdw, 1);x;repmat(x(end,:), wdw, 1)];
    end
    
    if size(x, 2) > 1
        x = [repmat(x(:, 1), 1, wdw),x,repmat(x(:,end), 1, wdw)];
    end

    y = conv2(x, ones(wdw, wdw)/wdw^sum(size(x)>1), 'same');
    
    if size(x, 1) > 1
        y = y(wdw+1:end-wdw,:);
    end
    if size(x, 2) > 1
        y = y(:,wdw+1:end-wdw);
    end
        
    