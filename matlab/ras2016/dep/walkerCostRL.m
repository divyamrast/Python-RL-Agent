function rl_cost = walkerCostRL(rl_data, vref, type)
    
    if strcmp(type, 'mayer')
        rl_ts_x = rl_data(end,1)-rl_data(1,1);
    else
        rl_ts_x = 0.2;
    end
       
    %RL
    addpath('/home/ivan/work/scripts/matlab/ras2016/');
    rl_t      = rl_data(:, 1);
    rl_theta  = rl_data(:, 2);
    rl_dtheta = rl_data(:, 3);
    rl_u      = rl_data(:, 4);
    rl_cost   = walkerCost(rl_t, rl_ts_x, 0.2, vref, rl_theta, rl_dtheta, rl_u);
end