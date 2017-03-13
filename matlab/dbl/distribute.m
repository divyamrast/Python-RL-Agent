function [dist, sz] = distribute(rmi, rma, N)
%DISTRIBUTE Partition a state space between processors
%   [DIST, SZ] = DISTRIBUTE(RMI, RMI, N) distributes the state space
%   spanned by the range RMI-RMA among N processors. RMI and RMA are 1xD
%   vectors that contain the minimum and maximum along the D dimensions. If
%   N is a 1xD vector, it is used as a predefined partition.
%
%   SZ is the partition, specifying the number of processors per dimension.
%   PROD(SZ) = N.
%
%   DIST is a DxM matrix containing the partition borders, where M =
%   MAX(SZ)+1. Irrelevant values (for SZ ~= MAX(SZ)) are set to Inf.
%   ROWIDX(DIST, ONES(D, 1)) = RMI and ROWIDX(DIST, SZ'+1) = RMA.
%
%   EXAMPLE:
%      >> [dist, sz] = distribute([0 0 0], [1 2 3], 12)
%      dist =
%               0    0.3333    0.6667    1.0000
%               0    1.0000    2.0000       Inf
%               0    1.5000    3.0000       Inf
%      
%      sz =
%           3     2     2
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    if numel(N) == numel(rmi)
        sz = N;
        N = prod(sz);
    else
        sz = factorize(N, length(rmi));
    end

    dist = repmat(rmi', 1, N+1) + repmat((0:N), length(rmi), 1).*repmat(((rma'-rmi')./sz'), 1, N+1);
    dist(dist>repmat(rma', 1, N+1)) = Inf;
    dist = dist(:,~all(isinf(dist)));

end
