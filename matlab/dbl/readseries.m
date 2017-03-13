function data = readseries(path, meas, time, timediv, down_sample)
%READSERIES Read tabular data from time-indexed file series.
%   DATA = READSERIES(PATH, MEAS, TIME) reads the files that match PATH
%   into a cell array, with each cell containing the TIME and MEAS
%   columns of its respective data file. By default, MEAS is 2 and TIME
%   is 1. If TIME is 0, the line number is recorded instead.
%
%   EXAMPLE:
%      data = readseries('TLMLearn_SEQ_LLR_*.txt');
%      [t, m, ~, e] = avgseries(data);
%      errorbaralpha(t, m, icdf('norm', 0.975, 0, 1)*e);
%      
%   SEE ALSO:
%      avgseries
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

if nargin < 2
    time = 1;
end
if nargin < 3
    meas = 2;
end
if nargin < 4
    timediv = 1;
end
if nargin < 5
    down_sample = 1;
end

files = dir(path);

if length(files) == 0
    error(['No files matching ''' path '''']);
end

folder = fileparts(path);

data = cell(1, length(files));

for ii=1:length(files)
    try
        d = dlmread(fullfile(folder, files(ii).name));
    catch
        error(['dlmread failed for ' fullfile(folder, files(ii).name)]);
    end
    
    d = downsample(d, down_sample);

    if time > 0
        d = d(d(:, time)>=0, :);
        if timediv > 0
            data{ii} = [d(:,time)/timediv d(:, meas)];
        else
            data{ii} = [log(d(:,time))/log(-timediv) d(:, meas)];
        end
    else
        data{ii} = [(1:size(d, 1))'/timediv d(:, meas)];
    end
end
