function finalizePlot(par, labels_names, size, lim)

    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'Position', [100 100 size(1) size(2)]);
    set(gcf, 'Color', 'w');

    if strcmp(par.log, 'log')
        set(gca,'XScale','log');
    end

    if (numel(labels_names) > 0)
        xlabel(labels_names{1}, 'Interpreter', 'none');
    end
    if (numel(labels_names) > 1)
        ylabel(labels_names{2}, 'Interpreter', 'none');
    end
       
    if nargin >= 4
        xlim(lim(1:2));
        if numel(lim) == 4
            ylim(lim(3:4));
        end
    end
    
    fixFonts(par.fontSize, par.fontFamily);
     
    % Floating point labels
    xtick = get(gca, 'xtick');
    xticklabel = textscan(sprintf(strcat(par.xlabelformat, ';'), xtick), '%s', 'delimiter', ';');
    set(gca,'xticklabel', xticklabel{1});
    
    ytick = get(gca, 'ytick');
    yticklabel = textscan(sprintf(strcat(par.ylabelformat, ';'), ytick), '%s', 'delimiter', ';');
    set(gca,'yticklabel', yticklabel{1});
    
    % Legend
    lh = legend;
    set(lh, 'Location', par.legendLocation, 'Orientation', par.legendOrientation);

    % Align y labels if we use subplots
    align_Ylabels(gcf);
end
