function C = calcWalkerCost(timesteps, vref, theta, dtheta, u, gamma, contact, slides)
    
    if (nargin < 6)
        gamma = 1;
    end
    
    if (nargin < 7)
        ids = 1;
        slides = 1;
    else
        ids = find(contact, slides, 'first');
    end
    
    if isempty(ids)
        C = NaN(1, 2);
        return;
    end
    
    c = zeros(slides, 1);
    j = 1;
    for i = ids'
        c(j) = walkerCost(timesteps, vref, theta(i:end), dtheta(i:end), u(i:end), gamma);
        j = j+1;
    end
    C = [min(c) max(c)];
end