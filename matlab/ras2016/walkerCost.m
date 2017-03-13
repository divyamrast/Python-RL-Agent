function [C, hipvx, len] = walkerCost(time, tsx, vref, theta, dtheta, u)
    
    % calculate average velocity and number of times it enters the
    % objective function
    [hipax, hipvx] = avgv(theta, dtheta);
    n = floor(max([1, (time(end) - time(1)) / tsx]));
    cv = 10*n*((hipax-vref)^2);

    % integral reward for controls
    tc = diff(time);
    cu = 0.01*sum(tc.*u(1:end-1).^2) / (time(end) - time(1));

    % calculate target cost function
    C = cv + cu;
    
    len = length(hipvx);
end

function [hipax, hipvx] = avgv(theta, dtheta)
    hipvx = - dtheta.*cos(theta);
    hipax = mean(hipvx);
end