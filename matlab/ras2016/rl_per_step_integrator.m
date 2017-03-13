function [x, u, t, stepLength, velocity, Xq, Uq] = rl_per_step_integrator(rl_data, interpolStep)

    rl_tc   = rl_data(:, 1);        % time
    rl_x    = rl_data(:, 2:5);      % state
    rl_cs   = rl_data(:, 6);        % contacts
    rl_u    = rl_data(:, 7);        % controls
    
    stateNum = size(rl_x, 2);
    stepNum = sum(rl_cs)-1;
    Xq = zeros(interpolStep, stateNum, stepNum);
    Uq = zeros(interpolStep, stepNum);
    tq = linspace(0, 1, interpolStep);

    cs = find(rl_cs);
    for step = 1:stepNum
        from = cs(step);
        to   = cs(step+1)-1;
        
        % interpolate each step states
        %t = (rl_tc(from:to)-rl_tc(from)) / (rl_tc(to)-rl_tc(from));
        %t = linspace(0, rl_tc(to)-rl_tc(from), numel(rl_tc(from:to)));
        t = linspace(0, 1, to-from+1);
        for state = 1:stateNum
            try
                Xq(:, state, step) = interp1(t, rl_x(from:to, state), tq, 'spline');
            catch
                warning('Problem using function.  Assigning a value of 0.');
            end
        end
        
        % ... and controls
        Uq(:, step) = interp1(t, rl_u(from:to), tq, 'nearest');
    end
    
    x = mean(Xq, 3);
    u = mean(Uq, 2);
    
    % time
    %c = find(rl_cs);
    tc = rl_tc(rl_cs == 1);
    dt = diff(tc);
    [m, v] = meanStd(dt);
    t = [m, 2*v];
    
    % step length
    thetas = squeeze(Xq([1, end], 1, :));
    s = sin(thetas);
    ss = (s(1, :) - s(2, :))';
    [m, v] = meanStd(ss);
    stepLength = [m, 2*v];
    
    % velocity
    vel = ss ./ dt;
    [m, v] = meanStd(vel);
    velocity = [m, 2*v];
    
end