function [oX, oY, oE] = getStdHist(X, Y, filterWindowSize)

    if (filterWindowSize == 0)
        filterWindowSize = min(X(2:end)-X(1:end-1));
    end
    
    xLength = max(X) - min(X);
    numBars = ceil(xLength / filterWindowSize);

    m = mean(Y, 2);
    s = std(Y, 0, 2);
    n = size(Y, 2);

    oX = zeros([numBars 1]);
    oY = zeros([numBars 1]);
    oE = zeros([numBars 1]);
%    oS = zeros([numBars 1]);
    for i = 1:numBars
       oX(i) = filterWindowSize/2 + filterWindowSize*(i-1);
       
       idx = ((filterWindowSize*(i-1) < X) & (X <= filterWindowSize*i));
       
       oY(i) = mean( m(idx) );
       oE(i) = mean( s(idx) );
%       oS(i) = sum(idx);
    end

    % Cut-off
    cutoff = abs(tinv(0.05/2, n-1)); % tinv is a one-sided function
    
    % does not account for samples falling into same bin
    oE = cutoff * oE / sqrt(n);

    % does account
%    oE = 1.96 * oE ./ sqrt(size(Y, 2) * oS(i));
end