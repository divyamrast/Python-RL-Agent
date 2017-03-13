function [h, he] = plotmanyseries(path, meas, time, timediv, varargin)
%PLOTMANYSERIES Plot tabular data from multiple file series
%   PLOTMANYSERIES(PATH, MEAS, TIME, TIMEDIV) calls PLOTSERIES for all
%   separate series discovered in PATH. Series are identifed by the
%   presence of a .yaml file.
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl> 

    colors = ['rgbcmykrgbcmykrgbcmykrgbcmyk'];
    styles = ['-------:::::::-------:::::::'];
    
    if nargin < 4
        error('Insufficient number of parameters!');
    end
    
    files = dir(path);

    if length(files) == 0
        error(['No files matching ''' path '''']);
    end
    
    folder = fileparts(path);
    
    series = {};
    for ii=1:length(files)
        if strcmp(files(ii).name(end-4:end), '.yaml')
            series = [series; files(ii).name(1:end-5)];
        end
    end
    
    series = sortlast(series);
     
    n = length(series);
    l = cell(n, 1);
    h = zeros(n, 1);
    he = zeros(n, 1);
    
    for ii=1:n
        [h(ii), he(ii)] = plotseries(fullfile(folder, [series{ii} '-*.txt']), ...
                                     meas, time, timediv, 'maxinterval', varargin{:}, 'color', colors(ii), ...
                                     'linestyle', styles(ii));
        hold on
    end
    hold off
    
    legend(h, series, 'Location', 'Best');
end
