function oc_cost = walkerCostOC(oc_data, vref, type)
    
    if strcmp(type, 'mayer')
        oc_ts_x = oc_data(end,1)-oc_data(1,1);
    else
        oc_ts_x = 0.2;
    end
    
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
    ts_u      = 0; % use all
    
    if (numel(oc_uc) == oc_shooting_intervals)
        oc_cost   = walkerCost(oc_t, oc_ts_x, ts_u, vref, oc_theta, oc_dtheta, oc_uc);
    else
        error('take care several same values of control');
    end   
end