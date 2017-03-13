function oc_cost = walkerCostOC(oc_data, vref, type)
    
    if strcmp(type, 'mayer')
        oc_ts_x = oc_data(end,1)-oc_data(1,1);
    else
        oc_ts_x = 0.2;
    end
    
    oc_t      = oc_data(:, 1);
    oc_theta  = oc_data(:, 2);
    oc_dtheta = oc_data(:, 3);
    oc_u      = oc_data(:, 4);
    oc_cost   = walkerCost(oc_t, oc_ts_x, vref, oc_theta, oc_dtheta, oc_u);
end