function print_eps(fileName)
    addpath('/home/ivan/work/scripts/export_fig/')
    set(gcf, 'PaperPositionMode', 'auto');
    set(gcf, 'Color', 'w');
    export_fig(fileName, '-eps');
    
    % convet to an eps which supports psfrag
%    system(['rm ',filenamebase,'.svg']);
%    inkscape phase_plot_rl012_oc012.eps  --export-eps=phase_plot_rl012_oc012-3.eps
    
    system(['inkscape ',fileName,'.eps --export-eps=',fileName,'.eps']);
end
