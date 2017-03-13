function plotCumulativeRegret()
    % config:
    % 'modelChanges': variations of model
    % 'timeStep': time between state transitions (used in integral CR calculations)
    %             0 means 'equal spread': (T1 - T0)/N. .csv files ignore this
    %             option and use 'equal spread'
    % 'dispStep': which states do we display (used by RL to display several solutions 
    %             with stochasticity)
    plotCumulative = 1;
    
    modelChanges = [0.10:0.20:0.90 0.95:0.05:1.05 1.1:0.3:2.0 3:10];
    %modelChanges = [0.30 1.00];
    addpath('/media/ivan/ivan/scripts/cr/');
    addpath('/media/ivan/ivan/scripts/csv/');
    addpath('/media/ivan/ivan/scripts/dbl/')
    addpath('/media/ivan/ivan/scripts/export_fig/')
        
    lineWidth = 2;
    baseline = calculateBaseline('baseline', modelChanges, 3);
    config = struct('modelChanges', modelChanges, 'baseline', baseline, 'timeStep', 0, 'dispStep', 0, 'dispOffset', 0, 'episodeNum', 1, 'playTrails', 0, 'plotPlays', 0);
    if plotCumulative
        config.episodeNum = 6000;
    end
    
    hold on; grid on;
    [H{1, 1} H{1, 2}]= plotCumulativeRegretSingle('manuel_cp_ff',  config, '-', 'r', lineWidth, 'OC\_FF');
%    [H{end+1, 1} H{end+1, 2}] = plotCumulativeRegretSingle('manuel_cp_pid_robust', config, '-', 'c', lineWidth, 'OC\_TT');

%    if plotCumulative
        config.dispStep = 1;
        [H{end+1, 1} H{end+1, 2}] = plotCumulativeRegretSingle('manuel_cp_rl_play',  config, '-', 'b', lineWidth, 'RL\_play');
%    end
    
    [H{end+1, 1} H{end+1, 2}] = plotCumulativeRegretSingle('manuel_cp_nmpc-th',  config, '-', 'k', lineWidth, 'NMPC-TH');
    [H{end+1, 1} H{end+1, 2}] = plotCumulativeRegretSingle('manuel_cp_nmpc-4',  config, ':', 'k', lineWidth, 'NMPC');
    
    if ~plotCumulative
        config.plotPlays = 1;
    end
    config.timeStep = 0.005;
    config.dispStep = 6001;
    config.playTrails = 1;
%    [H{end+1, 1} H{end+1, 2}] = plotCumulativeRegretSingle('manuel_cp_rl_noinit-6001',   config,  ':', 'b', lineWidth, 'RL\_no-init');
    [H{end+1, 1} H{end+1, 2}] = plotCumulativeRegretSingle('manuel_cp_rl_rlinit-6001',   config, '--', 'b', lineWidth, 'RL\_RL-init');
    %[H{end+1, 1} H{end+1, 2}] = plotCumulativeRegretSingle('manuel_cp_rl_pidinit-6001',  config, '--', 'm', lineWidth, 'RL\_TT-init');
    
    % RL + NMPC
    config.timeStep = 0.005;
    config.dispStep = 1001;
    config.playTrails = 1;
%    [H{end+1, 1} H{end+1, 2}] = plotCumulativeRegretSingle('manuel_cp_rl_nmpc-1001',  config, '--', 'g', lineWidth, 'RL+NMPC');
    hold off;

    legend(cell2mat(H(:, 1)), H(:, 2), ...
        'Location','northoutside','Orientation','horizontal');
    
    if plotCumulative
        ylabel('Cumulative regret'); ylim([-100 16000]);
    else
        ylabel('Simple regret'); ylim([0 2.2]);
    end
    
    xlabel('Pendulum mass, log_{10}(m)'); xlim([-1 1]);
    set(gcf, 'PaperPositionMode', 'auto');
   %set(gcf, 'Position', [100 100 500 200]);
    set(gcf, 'Position', [100 100 1000 350]);
    set(gcf, 'Color', 'w');

    %export_fig regret.eps -painters
    %export_fig simple-regret-rl+nmpc.eps
    if plotCumulative
        export_fig cumulative-regret.pdf
    else
        export_fig simple-regret.pdf
    end
    
end
