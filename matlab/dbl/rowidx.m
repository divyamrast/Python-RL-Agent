function y = rowidx(A, x)
%ROWIDX Row-wise multiple indexing
%   Y = ROWIDX(A, X) returns a vector containing, for each row of A, the
%   value of the column specified by X. Y(II) = A(II, X(II))
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    y = zeros(size(x));
    for ii=1:numel(x)
        y(ii) = A(ii, x(ii));
    end
end
