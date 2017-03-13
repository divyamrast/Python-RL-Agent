function p = fzerodata(x, y)
%FZERODATA Find zero crossing of noisy learning curves
%   P = FZERODATA(X, Y) finds the value of X at which Y makes a zero
%   crossing. The data is fitted with a sigmoid to get an interpolated X.
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    if any(size(x) ~= size(y))
        error('X and Y must be same size');
    end

    opt = statset('RobustWgtFun', 'cauchy');
    %opt = statset;
    
    if y(end) > y(1)
        par = nlinfit(x, y, @sigmoid, [-5 10*1./max(x) min(y) max(y)-min(y)], opt);
    else
        par = nlinfit(x, y, @sigmoid, [-5 10*1./max(x) max(y) min(y)-max(y)], opt);
    end
    
    m = sigmoid(par, x);
    [~, i] = min(abs(m));
    p = fzero(@(x) sigmoid(par, x), x(i));
    
      plot(x, y, x, m);
      line([p p], [min(y), max(y)], 'color', 'k');
      line([min(x) max(x)], [0, 0], 'color', 'k');
      drawnow;

    if isnan(p)
        error('Couldn''t find zero crossing');
    end
    
    function [ y ] = sigmoid( par, x )

        xoffset = par(1);
        xscale  = par(2);
        yoffset = par(3);
        yscale  = par(4);

        y = yoffset + yscale .* (1./(1+exp(-(xoffset + xscale.*x))));

    end

end
