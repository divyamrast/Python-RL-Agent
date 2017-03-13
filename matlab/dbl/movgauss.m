function y = movgauss(x, wdw, sigma)
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
    if nargin < 3
        sigma = 1;
    end

    if size(x, 1) > 1
        x = [repmat(x(1, :), wdw, 1);x;repmat(x(end,:), wdw, 1)];
    end
    
    if size(x, 2) > 1
        x = [repmat(x(:, 1), 1, wdw),x,repmat(x(:,end), 1, wdw)];
    end
    
    wdw1 = wdw-1;
    [xx, yy] = meshgrid(-1:(2/wdw1):1,-1:(2/wdw1):1);
    kernel = exp(-xx.^2./(2*sigma.^2)-yy.^2./(2*sigma.^2));
    kernel = kernel ./ sum(sum(kernel));
    
    if size(x, 1) == 1
        kernel = sum(kernel, 1);
    end
    
    if size(x, 2) == 1
        kernel = sum(kernel, 2);
    end
    

    y = conv2(x, kernel, 'same');
    
    if size(x, 1) > 1
        y = y(wdw+1:end-wdw,:);
    end
    if size(x, 2) > 1
        y = y(:,wdw+1:end-wdw);
    end
        
    