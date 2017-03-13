function c = cartpoleRewards(timesteps, pos, phi, posDot, phiDot)
    
    rewardShaping = 0; % == 0 or 1
    
    if (rewardShaping)
        failureReward = -100;
        sucessReward = 1;
        shapingGamma = 1.0;
    else
        failureReward = -1000;
        sucessReward = 0;
    end
        
    f = failed(pos, phi, posDot, phiDot);
    cf = failureReward * f; %reward for failure
    
    s = succeeded(pos, phi, posDot, phiDot);
    cs = sucessReward * s; %reward for success
    
    if (rewardShaping)
        % shift previous
        p_pos = [pos(1); pos];
        p_pos(end, :) = [];
        p_posDot = [posDot(1); posDot];
        p_posDot(end, :) = [];
        p_phi = [phi(1); phi];
        p_phi(end, :) = [];
        p_phiDot = [phiDot(1); phiDot];
        p_phiDot(end, :) = [];

        reward = shapingGamma * getPotential(pos, phi, posDot, phiDot) ...
            - getPotential(p_pos, p_phi, p_posDot, p_phiDot);
    else
        reward = getPotential(pos, phi, posDot, phiDot);
    end
    
    % take non failure cases of reward
    reward = reward .* (1 - f);
    
    R = reward + cf + cs;
    %R = reward + cs;
    %R = getPotential(pos, phi, posDot, phiDot);
    R = R .* timesteps;
    c = sum(R);
end

% The function wraps the angle a from RBDL representation to MPRL (RVIZ).
% Input: any angle where pi means bottom position and 0 or pi mean upswing position (RBDL)
% Output: any angle in range [0; 2pi] where pi means upswing position and 0 or 2pi mean botton position (MPRL and RVIZ)
function b = wrap_angle(a)
  b = zeros(size(a));
  a = mod(a, 2*pi);
  b(a <  pi) = a(a < pi) + pi; % [  0; pi) -> [pi; 2pi)
  b(a >= pi) = a(a > pi) - pi; % [2pi; pi] -> [pi; 0]
end

function p = getPotential(pos, phi, posDot, phiDot)
  phi = wrap_angle(phi) - pi;

  % original reward
  p = -2    *  pos.^2 ...
      -1    *  phi.^2 ...
      -0.1  *  posDot.^2 ...
      -0.5  *  phiDot.^2;
end

function s = succeeded(pos, phi, posDot, phiDot)
    phi = wrap_angle(phi) - pi;
    pos    = pos    < 0.1           & pos    > -0.1;
    posDot = posDot < 0.5           & posDot > -0.5;
    phi    = phi    < 5*pi/180      & phi    > -5*pi/180;
    phiDot = phiDot < 50*pi/180     & phiDot > -50*pi/180;
    
    s = pos .* posDot .* phi .* phiDot;
end

function f = failed(pos, phi, posDot, phiDot)
    f = pos < -2.4 | pos > 2.4;
end