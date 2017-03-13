function s = discretize(v, minv, maxv, numstates)
%DISCRETIZE Discretize and clip a continuous value.
%   S = DISCRETIZE(V, MINV, MAXV, NUMSTATES) discretizes V,
%   from domain [MINV, MAXV] into a discrete state S in range
%   [1, NUMSTATES].
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    if any(size(v) ~= size(maxv))
        minv = repmat(minv, size(v)./size(minv));
        maxv = repmat(maxv, size(v)./size(maxv));
        numstates = repmat(numstates, size(v)./size(numstates));
    end

    myx = max(min(v, maxv), minv);
    myx = (myx - minv)./(maxv-minv+eps(maxv-minv));
    s = fix(myx .* numstates)+1;
end
