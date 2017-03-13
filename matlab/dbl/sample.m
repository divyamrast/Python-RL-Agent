function a = sample(p)
%SAMPLE Sample from a discrete distribution
%   A = SAMPLE(P) samples an action according to distribution P.
%   P is a vector of probabilities for each action. P does not
%   have to sum to 1, in which case it is normalized.
    pc = cumsum(p);
    a = find(pc > pc(end)*rand, 1);
end
