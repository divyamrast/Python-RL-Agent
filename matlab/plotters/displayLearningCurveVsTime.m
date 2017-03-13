function [h, name, data] = displayLearningCurveVsTime(par, figPar, varargin)
%PLOTSERIES Plot tabular data from time-indexed file series.
%   PLOTSERIES(PATH, MEAS, TIME, TIMEDIV, ...) averages and plots the MEAS
%   column of the files that match PATH using TIME (divided by TIMEDIVE)
%   as the time index. By default, MEAS is 2 and TIME is 1. If TIME is 0,
%   the series is plotted agains the line number instead.
%   PLOTSERIES(..., 'integral') plots the integral of the MEAS column.
%   PLOTSERIES(..., 'derivative') plots the finite difference of the MEAS
%   column.
%   PLOTSERIES(..., 'extrap') extrapolates the data of shorter series.
%   Extra arguments are passed to ERRORBARALPHA. 
%   [H, HE] = PLOTSERIES(...) returns the handles to the average and error
%   data plots.
%
%   EXAMPLE:
%      plotseries('TLMLearn_SEQ_LLR_*.txt', 2, 1, 'Color', 'r');
%
%   SEE ALSO:
%      readseries, avgseries, errorbaralpha.
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>
    
    addpath('/home/ivan/work/scripts/matlab/dbl');

    if nargin < 2
        error('Insufficient number of parameters!');
    end
    
    path = figPar.path;
    meas = par.time_indexed_meas; 
    time = par.time_indexed_time;
    timediv = par.time_indexed_timediv;
    
    varargin = varargin{1};
    process = 0;
    extrap = {[]};
    dolog = 0;
    multiplier = 1;
    plot_error = 1;
    denominator = [];
    
    for ii = 1:5
        if ~isempty(varargin)
            if strcmp(varargin{1}, 'integral')
                process = 1;
                varargin = varargin(2:end);
            elseif strcmp(varargin{1}, 'derivative')
                process = 2;
                varargin = varargin(2:end);
            elseif strcmp(varargin{1}, 'median')
                process = 3;
                varargin = varargin(2:end);
            elseif strcmp(varargin{1}, 'filter')
                process = 4;
                varargin = varargin(2:end);
            elseif any(strcmp(varargin{1}, 'extrap') + strcmp(varargin{1}, 'maxinterval') + strcmp(varargin{1}, 'extrap10'))
                extrap = {varargin{1}};
                varargin = varargin(2:end);
            elseif strcmp(varargin{1}, 'log')
                dolog = 1;
                varargin = varargin(2:end);
            elseif strcmp(varargin{1}, 'multiplier')
                multiplier = varargin{2};
                varargin = varargin(3:end);
            elseif strcmp(varargin{1}, 'denominator')
                denominator = varargin{2};
                varargin = varargin(3:end);
            elseif strcmp(varargin{1}, 'no-error')
                plot_error = 0;
                varargin = varargin(2:end);
            end
        end
    end
    
    if ~isempty(denominator)
        meas = [meas denominator];
    end
    data = readseries(path, meas, time, timediv, par.downsample); 
    
    if abs(process)
        for ii = 1:length(data)
            d = data{ii};
            if process == 1
                data{ii} = [d(:, 1) cumsum(d(:, 2:end))];
            elseif process == 2
                data{ii} = [d(2:end, 1) [d(2:end, 2:end) - d(1:end-1, 2:end)]];
            elseif process == 3
                data{ii} = [d(:, 1) movmedian(d(:, 2:end))];
            elseif process == 4
                data{ii} = filteroutliers(d);
            end
        end
    end
    
    
    for ii=1:length(data)
        C = data{ii};
        if ~isempty(denominator)
            C = horzcat(C(:,1), C(:,2) ./ C(:,3));
        end
        if (multiplier ~= 1)
            C(:,2) = multiplier * C(:,2);
        end
        data{ii} = C;
    end
    
    [X, Y, ~, E] = avgseries(data, extrap{:}, 5);  
    
    if dolog
        X = log2(X);
    end
    
    data = {X, Y, E};
    
    if (plot_error == 0)
        E = zeros(size(X));
    end
     
    % plot
    if par.plot
        hold on;
    end
    
    if (par.plot)
        h = errorbaralpha(X, Y, E, ...
            'Color', figPar.lineColour, 'LineWidth', figPar.lineWidth, ...
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
    
    
    len = length(X);
    bi = floor(len*0.99); % take last 1%
    m = mean(Y(bi:len));
    ci95 = mean(E(bi:len));
%     s = std(Y(bi:len));
%     n = len-bi+1;
%     
%     sn = sqrt(n);
%     cutoff = abs(tinv(0.05/2, n-1)); % ivan: tinv is a one-sided function
%     cutoff(n == 1) = 0; % if there is only one data point then ci95 = 0
%     ci95 = s*cutoff/sn;
    
    precise = sprintf('%s, R = %.2f±%.2f', figPar.name, m, ci95);
    disp(precise);
    if par.legendPrecise == 0
        name = figPar.name;
    else
        name = precise;
    end

end

%     % calculate end point performance
%     bi = floor(length(E)*0.99); % take last 1%
%     index = find(E(bi:end) == max(E(bi:end)), 1, 'last')-1;
%     precise = sprintf('%s, R = %.2f±%.2f', figPar.name, Y(bi+index), E(bi+index));
%     disp(precise);
%     if par.legendPrecise == 0
%         name = figPar.name;
%     else
%         name = precise;
%     end
% end
