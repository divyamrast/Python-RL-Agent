function s = ind2subv(sz, ndx)
%IND2SUBV Version of IND2SUB which returns a vector
%   S = IND2SUBV(SZ, NDX) returns the vector of subscripts for the linear
%   index NDX in a matrix of size SZ.
%
%   Unlike IND2SUB, IND2SUBV does not apply error checking.
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    s = zeros(size(sz));
    k = [1 cumprod(sz(1:end-1))];
    n = numel(sz);
    
    for i = n:-1:1,
      vi = rem(ndx-1, k(i)) + 1;         
      vj = (ndx - vi)/k(i) + 1; 
      s(i) = vj; 
      ndx = vi;     
    end
