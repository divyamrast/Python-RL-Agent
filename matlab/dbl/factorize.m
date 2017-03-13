function y = factorize(n, k)
%FACTORIZE Calculates minimal sum factorization
%   Y = FACTORIZE(N, K) calculates an approximation to the factorization
%   of N into K factors with minimal sum.
%
%   EXAMPLE:
%      >> factorize(12, 3)
%
%      ans =
%           3     2     2
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    if k == 1
        y = n;
        return;
    end

    t = ceil(n^(1/k));
    d = 1;
    y = [];
    
    while t <= n
        if t > 0
            r = n/t;
            if r == fix(r)
                y = factorize(r, k-1);
                if ~isempty(y)
                    y = [t y];
                    break;
                end
            end
        end
        
        t = t + d;
        d = -sign(d)*(abs(d)+1);
    end

    y = sort(y, 'descend');
    
end
