function [h, name, data] = displayLearningCurve(par, figPar)
    
    if ~any(strcmp('type',fieldnames(figPar)))
        figPar.type = 'errorbar';
    end
    if ~any(strcmp('dataColumn',fieldnames(figPar)))
        figPar.dataColumn = getColumnID('rewardPerEpisode');
    end
    if ~any(strcmp('line',fieldnames(figPar)))
        figPar.line = 'b';
    end
    if ~any(strcmp('path',fieldnames(figPar)))
        figPar.path = 'fqi-ertrees-learning-curve/*.txt';
    end
    
    % precalculate length of data to read and number of files to read
    rows = rowsNum(figPar.path);

    % read measurement column from each file
    [dataBank, rows] = readColunms(figPar.path, figPar.dataColumn, rows, par.maxEpisodes);
    s = sprintf('%s averages over %d curves', figPar.path, size(dataBank,2));
    disp(s);

    % downsample
    dataBank = downsample(dataBank, par.downsample);
    rows = size(dataBank, 1);
    
    % read "episode number" column from each file (always 1st column)
    episodeBank = readColunms(figPar.path, 1, rows);
    episodes = episodeBank(:, 1); % use first column for episodes

    if par.plot
        subplot(figPar.subplotNum, 1, figPar.subplotCount);
        hold on;
    end
    
    % Calculate last 10 values mean and 95% conf value
    mfilepath=fileparts(which('displayLearningCurve'));
    mfilepath = fullfile(mfilepath,'../statistics');
    addpath(mfilepath);
    data = cell(1, 3);
    h = 0;
    if strcmp(figPar.type, 'errorbar-alpha')
        [X, Y, E] = getStdHist(episodes, dataBank, par.filterWindowSize);
        data = {X, Y, E};
        if (par.plot)
            h = errorbaralpha(X, Y, E, ...
                'Color', figPar.lineColour, 'LineWidth', 1, ...
                'LineStyle', figPar.lineStyle, 'rendering', 'alpha');
            xlabel(figPar.xLabel);
            ylabel(figPar.yLabel);
            if par.grid
                grid on;
            end
            if par.box
                box on;
            end
        end
    elseif strcmp(figPar.type, 'errorbar')
        [X, Y, E] = getStdHist(episodes, dataBank, par.filterWindowSize);
        data = {X, Y, E};
        if (par.plot)
            h = errorbar(X, Y, E, figPar.line);
            xlabel(figPar.xLabel);
            ylabel(figPar.yLabel);
            if par.grid
                grid on;
            end
            if par.box
                box on;
            end
        end
    end

    bi = floor(length(E)*0.99); % take last 1%
    index = find(E(bi:end) == max(E(bi:end)), 1, 'last')-1;
%    precise = sprintf('%s, R = %.1f±%.1f', figPar.name, Y(end), E(end));
    precise = sprintf('%s, R = %.1f±%.1f', figPar.name, Y(bi+index), E(bi+index));
    disp(precise);
    if par.legendPrecise == 0
        name = figPar.name;
    else
        name = precise;
    end
end


