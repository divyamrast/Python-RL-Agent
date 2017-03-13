function rl_cost = walkerCostRL(rl_data)
    
    if strcmp(type, 'mayer')
        rl_ts_x = rl_data(end,1)-rl_data(1,1);
    else
        rl_ts_x = 0.2;
    end
    
    % Lagrange term x cost
%     
%     oc_ts_x = rl_ts_x;
    
    % Mayer term x cost
    
    oc_ts_x = oc_data(end,1)-oc_data(1,1);
    
    %RL
    addpath('/home/ivan/work/scripts/matlab/ras2016/');
    rl_t      = rl_data(:, 1);
    rl_theta  = rl_data(:, 2);
    rl_dtheta = rl_data(:, 3);
    rl_u      = rl_data(:, 4);
    rl_cost   = walkerCost(rl_t, rl_ts_x, 0.2, vref, rl_theta, rl_dtheta, rl_u);
    
    % OC
    % obtain shooting nodes interval
    oc_u      = oc_data(:, 4);
    [~,ia,ic]  = unique(oc_u);
    oc_uc = oc_u(sort(ia));
    m = numel(ic);
    for i = 1:max(ic)
        if (sum(ic==i)) < m
            m = sum(ic==i);
        end
    end
    oc_shooting_intervals = round(numel(ic) / m);
    oc_t      = oc_data(:, 1);
    oc_theta  = oc_data(:, 2);
    oc_dtheta = oc_data(:, 3);
    %oc_u      = oc_data(:, 4);
    ts_u   = 0; % use all
    
    if (numel(oc_uc) == oc_shooting_intervals)
        oc_cost   = walkerCost(oc_t, oc_ts_x, ts_u, vref, oc_theta, oc_dtheta, oc_uc);
    else
        error('take care several same values of control');
    end   
end