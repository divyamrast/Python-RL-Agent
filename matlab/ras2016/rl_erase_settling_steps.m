function data = rl_erase_settling_steps(rl_data, t_cut)
    rl_data(:, 1) = rl_data(:, 1) - rl_data(1, 1);
    rl_tc = rl_data(:, 1);      % time
    rl_cs = rl_data(:, 6);      % contacts
    
    idx = find(rl_tc > t_cut, 1, 'first');
    contactBegin = idx + find(rl_cs(idx:end), 1, 'first') - 1;
    contactEnd   = idx + find(rl_cs(idx:end), 1, 'last')  - 1;
    
    data  = rl_data(contactBegin:contactEnd, :);
end