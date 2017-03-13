function [lc, meanYY, stdYY] = calcLC(v, c)

    % delete incomplete cycles
    idx0 = find(c, 1, 'first');
    idx1 = find(c, 1, 'last');
    v = v(idx0:idx1);
    c = c(idx0:idx1);

    i0 = 1;
    NPT = 10;
    stepNum = sum(c)-1;
    YY = zeros(stepNum, NPT);
	skip = 0;
    for j = 1:stepNum-skip
        i1 = i0 + find(c(i0+1:end), 1, 'first');
        l = i1 - i0;
        if (l < 2)
           YY(end, :) = [];
           skip = skip + 1;
           i0 = i1; % transit to the next step
           continue;
        end
        xx = 1:(l-1)/(NPT-1):l;
        yy = spline(1:l, v(i0:i1-1), xx);
        YY(j-skip,:) = yy;
        i0 = i1; % transit to the next step
    end
    
    stdYY = std(YY);
    meanYY = mean(YY);
    lc = max(stdYY);
end