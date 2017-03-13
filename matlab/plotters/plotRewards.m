function Data = plotRewards(c, par, varargin)
    close all;
    addpath('/home/ivan/work/scripts/matlab/color');
    lineColours = {'b', 'r', 'm', 'c', 'k', 'g'};
    %lineColours = {hex2rgb('51843b'), hex2rgb('ffb171'), 'k'};
    lineStyles  = {'-', ':', '-.', '--'}; 
    
    figPar.dataColumn = 3;
    figPar.title    = c{1};
    figPar.xLabel   = c{2};
    figPar.yLabel   = c{3};
    figPar.lineWidth   = 1;
    figPar.type     = 'errorbar-alpha';
    dataSourse      = c(4:end);
    
    if mod(length(dataSourse),2)
        error('Data plots is incorrectly defined');
    end
    plotsNum = length(dataSourse)/2;
    plotsPerSubplot = plotsNum;
    
    if plotsNum > par.graphsPerPlot
        if (mod(plotsNum, 2) == 0 && plotsNum <= 4)
            if plotsNum <= 2*par.graphsPerPlot
                plotsPerSubplot = 2;
            end
        else
%            plotsPerSubplot = ceil(plotsNum / par.graphsPerPlot);
            plotsPerSubplot = par.graphsPerPlot;
        end
    end
    
    figPar.subplotCount = 1;
    figPar.subplotNum = ceil(plotsNum / plotsPerSubplot);

    plotsCountAtSubplot = 0;
    lineColourCounter = 1;
    lineStyleCounter = 1;
    legendInfo = cell(plotsPerSubplot, 2);
    Data = cell(plotsNum, 1);
    subplotInfo = subplot(figPar.subplotNum,1,figPar.subplotCount);
    for i = 1:plotsNum
        plotsCountAtSubplot = 1 + plotsCountAtSubplot;
        figPar.name = dataSourse{2*(i-1)+1};
        figPar.path = dataSourse{2*(i-1)+2};
        
        % Select line specs
        if par.bw == 0
            % colour
            figPar.lineColour = lineColours{lineColourCounter};
            figPar.lineStyle = lineStyles{lineStyleCounter};
        else
            % bw
            figPar.lineColour = 'k';
            figPar.lineStyle = lineStyles{lineStyleCounter};
            lineStyleCounter = lineStyleCounter + 1;
        end
            
        if plotsCountAtSubplot > plotsPerSubplot
            
            % Plot main title
            if (figPar.subplotCount)
                title(figPar.title);
            end
            
            % Plot subplot legend
            %legend(cell2mat(legendInfo(:, 1)), legendInfo(:, 2), 'Location', par.legendLocation, 'Orientation', par.legendOrientation);
            legend([legendInfo{:, 1}], legendInfo(:, 2), 'Location', par.legendLocation, 'Orientation', par.legendOrientation);
            legendInfo = {}; % empty for new legend
            
            plotsCountAtSubplot = 1;
            figPar.subplotCount = 1 + figPar.subplotCount;
            subplotInfo = [subplotInfo subplot(figPar.subplotNum,1,figPar.subplotCount)];
        end
        
        numberOfPlots = (figPar.subplotCount-1)*plotsPerSubplot + plotsCountAtSubplot;
        if par.bw == 0
            % colour
            if mod(numberOfPlots, length(lineColours)) ~= 0
                lineColourCounter = lineColourCounter + 1;
            else
                lineColourCounter = 1;
            end
            
            if lineStyleCounter == numel(lineStyles)
                lineStyleCounter = 1;
            else
                lineStyleCounter = lineStyleCounter + 1;
            end
        end
        
        if (par.time_indexed)
            [legendInfo{plotsCountAtSubplot, 1}, legendInfo{plotsCountAtSubplot, 2}, Data{i}] = displayLearningCurveVsTime(par, figPar, varargin);
        else
            [legendInfo{plotsCountAtSubplot, 1}, legendInfo{plotsCountAtSubplot, 2}, Data{i}] = displayLearningCurve(par, figPar);
        end
    end

    % apply limits if required
    if ~isnan(par.y_lim)
        for i = 1: numel(subplotInfo)
            ylim(subplotInfo(i), [par.y_lim(1) par.y_lim(2)]);
        end
    else
        mmlim = ylim(subplotInfo(1));
        for i = 2: numel(subplotInfo)
            lim = ylim(subplotInfo(i));
            mmlim = [min(mmlim(1), lim(1)) max(mmlim(2), lim(2))];
        end
        for i = 1: numel(subplotInfo)
            ylim(subplotInfo(i), mmlim);
        end
    end
    if ~isnan(par.x_lim)
        for i = 1: numel(subplotInfo)
            xlim(subplotInfo(i), [par.x_lim(1) par.x_lim(2)]);
        end
    end
    if ~isnan(par.xtick)
        for i = 1: numel(subplotInfo)
            set(subplotInfo(i), 'xtick', par.xtick);
        end
    end
    
    if par.plot == 1
        % In case no subplots are used, print title in the end
        if (figPar.subplotNum == 1 && par.showTitle)
            title(figPar.title);
        end
        
        legendInfo(plotsCountAtSubplot+1:end,:) = [];
        
        % Legend was not printed for the last subplot or plot
        legend([legendInfo{:, 1}], legendInfo(:, 2), 'Location', par.legendLocation, 'Orientation', par.legendOrientation);
    end
       
    set(gcf, 'PaperPositionMode', 'auto');
    if isfield( par, 'size' )
        set(gcf, 'Position', [100 100 par.size(1) par.size(2)]);
    end
    set(gcf, 'Color', 'w');
    
    % Scale
    if (par.logy)
        set(gca,'YScale','log');
    end
        
    % Legend
    if ischar(par.legendLocation)
        set(legend, 'Location', par.legendLocation, 'Orientation', par.legendOrientation);
    else
        set(legend, 'Position', par.legendLocation, 'Orientation', par.legendOrientation);
    end
    
    if (par.showLegend == 0)
        delete(legend);
    end
    
    set(gca,'XTickLabel',num2str(get(gca,'XTick').'));
    
    % saving
    if par.export == 1
        addpath('/home/ivan/work/scripts/matlab/exporters');
        addpath('/home/ivan/work/scripts/matlab/exporters/export_fig');
        save_fname = strcat(lower(figPar.title), '_', lower(figPar.xLabel), '_', lower(figPar.yLabel));
        save_fname(ismember(save_fname,' ,.:;!%()')) = '_';
        %print(save_fname, '-dpng');
        if strcmp(par.exportFormat, 'pdf')
            painter = '-painters';
            if isfield( par, 'exportPainter' )
                painter = par.exportPainter;
            end
            %print([save_fname '-l.pdf'], '-dpdf', painter);
            print_pdf_alpha(save_fname, 1, 1);
            print_pdf_alpha([save_fname '-l'], 1, 0);
            
            % MISC:
            %print_pdf_alpha(save_fname, 1, 0);
            %print([save_fname '.svg'], '-dsvg');
            %print('-depsc2', '-painters', '-r864', [save_fname '.eps']);
        end
    end
end