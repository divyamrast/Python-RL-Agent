function C = cartPendulumCost(timesteps, wrap, pos, phi, posDot, phiDot, u, gamma)
    
    if (nargin < 7)
        u = zeros(size(pos));
    end
    
    if (nargin < 8)
        gamma = 1;
    end

    phi = wrapping(phi, wrap);
    
    c = cost(pos, phi, posDot, phiDot, u);
    c = c .* timesteps;
    g = gamma.^(0:numel(pos)-1);
    c = c.* g';
    C = sum(c);
end

function c = cost(pos, phi, posDot, phiDot, u)
  c = +2      *  pos.^2 ...
      +1      *  phi.^2 ...
      +0.2    *  posDot.^2 ...
      +0.5    *  phiDot.^2 ...
      +0.0005 * u.^2;
end

function phi = wrapping(phi, wrap)
    if ~sum(phi > pi) && wrap
        warning('Wrapping might not be requited!');
    end
    
    if (wrap)
        phi(phi > pi) = phi(phi > pi) - 2*pi;
    end
    
    if sum(phi > pi) || sum(phi < -pi)
        %warning('Wrapping might be requited!');
    end
end