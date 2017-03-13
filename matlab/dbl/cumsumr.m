function y = cumsumr(x)
%CUMSUMR Cumulative sum with reset
%   Y = CUMSUMR(X) returns the cumulative sum of X, with resets
%   when X is 0.

    y = double(x);

    for ii = 2:length(x)
        if x(ii) ~= 0
            y(ii) = y(ii-1) + x(ii);
        else
            y(ii) = 0;
        end
    end

end

