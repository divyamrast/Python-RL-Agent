function [h, he] = plotseries(path, meas, time, timediv, varargin)
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

    if nargin < 4
        error('Insufficient number of parameters!');
    end
    
    process = 0;
    extrap = {[]};
    dolog = 0;
    
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
            elseif strcmp(varargin{1}, 'extrap') || strcmp(varargin{1}, 'maxinterval') || strcmp(varargin{1}, 'extrap10')
                extrap = {varargin{1}};
                varargin = varargin(2:end);
            elseif strcmp(varargin{1}, 'log')
                dolog = 1;
                varargin = varargin(2:end);
            end
        end
    end
            
    data = readseries(path, meas, time, timediv);
    
    if abs(process)
        for ii = 1:length(data)
            d = data{ii};
            if process == 1
                data{ii} = [d(:, 1) cumsum(d(:, 2))];
            elseif process == 2
                data{ii} = [d(2:end, 1) [d(2:end, 2) - d(1:end-1, 2)]];
            elseif process == 3
                data{ii} = [d(:, 1) movmedian(d(:, 2))];
            elseif process == 4
                data{ii} = filteroutliers(d);
            end
        end
    end
    
    [t, m, ~, ci95] = avgseries(data, extrap{:}, 5);
    
    if dolog
        t = log2(t);
    end
    
    [h, he] = errorbaralpha(t, m, ci95, varargin{:});
end
