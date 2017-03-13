function [C, hipvx, len] = walkerCost(time, tsx, tsu, vref, theta, dtheta, u, gamma)

    % coast calculation shall be started when contact = 1

    if (nargin < 7)
        u = zeros(size(theta));
    end
    
    if (nargin < 8)
        gamma = 1;
    end
    
    % calculate average velocity and number of times it enters the
    % objective function
    [hipax, hipvx] = avgv(theta, dtheta);
    n = (time(end) - time(1)) / tsx;
    
    % resample u to match the shooting nodes interval / sampling period
    if (tsu ~= 0)
        ru = interp1(time, u, time(1):tsu:time(end), 'nearest');
    else
        ru = u;
    end

    % calculate target cost function
    C = 10*n*((hipax-vref)^2) + 0.01*sum(ru.^2);
    
    len = length(hipvx);
end

function [hipax, hipvx] = avgv(theta, dtheta)
    hipvx = - dtheta.*cos(theta);
    hipax = mean(hipvx);
end