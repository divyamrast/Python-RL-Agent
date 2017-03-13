function ndx = sub2indv(sz, s)
%SUB2INDV Version of SUB2IND which takes a vector
%   NDX = SUB2INDV(SZ, S) returns the linear index NDX for the subscripts
%   S in a matrix of size SZ.
%
%   Unlike SUB2IND, SUB2INDV does not apply error checking.
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    k = [1 cumprod(sz(1:end-1))];
    ndx = k*(s-1)'+1;
