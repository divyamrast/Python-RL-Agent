function compare()

    addpath('/home/ivan/work/scripts/matlab/plotters');
    par = initPlotSettings();
    par.time_indexed = 1;
    par.export = 1;
    par.showTitle = 0;
    par.graphsPerPlot = 10;
    par.legendPrecise = 0;
    par.size = [400 300];
    %par.size = [800 600];
    par.x_lim = [0 34000];
    par.xtick = par.x_lim(1):10000:par.x_lim(2);
%    par.legendLocation = 'northoutside';
    
    path = 'data/';

%         'greedy', [path 'leo_ou_test_leosim_sarsa_walk_greedy-mp*.txt'], ...
%         'egreedy', [path 'leo_ou_test_leosim_sarsa_walk_egreedy-mp*.txt'], ...
%         'pada', [path 'leo_ou_test_leosim_sarsa_walk_pada-mp*.txt'], ...
%         'acou', [path 'leo_ou_test_leosim_sarsa_walk_ou-mp*.txt'], ...
%         'ou', ['50-2-ou2/leo_ou_test_leosim_sarsa_walk_ou2-mp*.txt'], ...
%         'oupada_____', [path 'leo_ou_test_leosim_sarsa_walk_oupada-mp*.txt'], ...
%         'epada', [path 'leo_ou_test_leosim_sarsa_walk_epada-mp*.txt'], ...
%         'eou', [path 'leo_ou_test_leosim_sarsa_walk_eou-mp*.txt'], ...
%         'padatwo', [path 'leo_ou_test_leosim_sarsa_walk_pada_2-mp*.txt'], ...
    
% PAPER
    series = {...
        'greedy', [path 'leo_ou_test_leosim_sarsa_walk_greedy-mp*.txt'], ...
        'pada', [path 'leo_ou_test_leosim_sarsa_walk_pada-mp*.txt'], ...
        'ou', [path 'leo_ou_test_leosim_sarsa_walk_ou-mp*.txt'], ...
        'oupada____', [path 'leo_ou_test_leosim_sarsa_walk_oupada-mp*.txt'], ...
        'acou', [path 'leo_ou_test_leosim_sarsa_walk_acou-mp*.txt'], ...
        };
    
% PRESENTATION
    series = {...
        'greedy', [path 'leo_ou_test_leosim_sarsa_walk_greedy-mp*.txt'], ...
        'egreedy', [path 'leo_ou_test_leosim_sarsa_walk_egreedy-mp*.txt'], ...
        'pada', [path 'leo_ou_test_leosim_sarsa_walk_pada-mp*.txt'], ...
        'ou', [path 'leo_ou_test_leosim_sarsa_walk_ou-mp*.txt'], ...
        'eou', [path 'leo_ou_test_leosim_sarsa_walk_eou-mp*.txt'], ...
        };
    
    for i = 1:numel(series)
        jseries{i} = strrep(series{i}, '.txt', '.j');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%
    
%     files = dir(jseries{2});
%     d = zeros(length(files), 2);
%     for ii=1:length(files)
%         f = files(ii).name;
%         a = strfind(f, '-mp')+3;
%         b = strfind(f, '-learn')-1;
%         c = strfind(f, '-0-l')+4;
%         d(ii, 1) = str2double(f(a:b));
%         d(ii, 2) = str2double(f(c:c+1));
%     end
%     
%     B = readseries(jseries{2}, 2, 1, 1, 1);
%     en = zeros(length(B), 1);
%     for ii=1:length(B)
%         en(ii) = B{ii}(end, 2);
%     end
%     
%     n = numel(en);
%     sn = sqrt(n);
%     cutoff = abs(tinv(0.05/2, n-1)); % ivan: tinv is a one-sided function
% 
%     m = mean(en);
%     s = std(en);
%     ci95 = s*cutoff/sn;

    
    %%%%%%%%%%%%%%%%%%%%%%

    par.time_indexed_time = 1;
    par.time_indexed_meas = 2;
    par.time_indexed_timediv = 1;
    par.downsample = 1200;
    c = horzcat({'Fatigue', 'Time', 'Fatigue'}, jseries);
    plotRewards(c, par);
     
    par.time_indexed_time = 2;
    par.time_indexed_meas = 3;
    par.time_indexed_timediv = 30;
    par.downsample = 2;
    c = horzcat({'Leo learning curve', 'Time', 'Reward'}, series);
    plotRewards(c, par);
    
    par.time_indexed_meas = 5;
    c = horzcat({'Leo fall rate', 'Time', 'Cumulative number of falls'}, series);
    par.y_lim = [100 20000];
    par.logy = 1;
    %par.legendLocation = [0.6, 0.15, .25, .25];
    plotRewards(c, par);
    
    % falls per hour
    %plotRewards(c, par, 'derivative', 'multiplier', 120000*10/11, 'denominator', 2);
    
end
