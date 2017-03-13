function [t, m, s, ci95, n] = avgseries(B, extrapolate, subsampling)
%AVGSERIES Average a set of time series with different samplings.
%   T = AVGSERIES(SERIES) returns a new equidistant sampling of the
%   N time series in SERIES. SERIES is a cell array of ?x(M+1) matrixes,
%   with the first column of each matrix being the sampling times.
%   [T, M, S, CI95] = AVGSERIES(SERIES) additionally returns the mean,
%   standard deviation and 95% confidence interval of the mean of the series
%   evaluated at T.
%   ... = AVGSERIES(..., 'maxinterval') uses the union of the series
%   instead of the intersection to calculate the new interval. Standard
%   errors are calculated over the support of each timepoint.
%   ... = AVGSERIES(..., 'extrap') uses a zero-order hold over the extreme
%   10% of the data as the support for nonexisting timepoints.
%   ... = AVGSERIES(..., N) supersamples the output by a factor of N.
%
%   EXAMPLE:
%      x1 = [-pi:.1:pi] + 0.2*randn(1, 63); y1 = sin(x1)+0.2*randn(1,63);
%      x2 = [-pi:.2:pi] + 0.2*randn(1, 32); y2 = sin(x2)+0.2*randn(1,32);
%      [t, m, e] = avgseries({[x1' y1'], [x2' y2']});
%      errorbaralpha(t, m, e);
% 
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

if nargin < 2
    extrapolate = '';
end
if nargin < 3
    subsampling = 1;
end

times = [];

be = zeros(1, length(B));
en = zeros(1, length(B));
for ii=1:length(B)
    times = [times; B{ii}(:, 1)];
    be(ii) = B{ii}(1, 1);
    en(ii) = B{ii}(end, 1);
end
    
if strcmp(extrapolate, 'extrap') || strcmp(extrapolate, 'maxinterval')
    be = min(be);
    en = max(en);
elseif strcmp(extrapolate, 'extrap10')
    tenpct = floor(length(B)/10);
    be = sort(be); be = be(end-tenpct);
    en = sort(en); en = en(1+tenpct);
else
    be = max(be);
    en = min(en);
end

if strcmp(extrapolate, 'extrap') || strcmp(extrapolate, 'extrap10')
    for ii=1:length(B)
        C = B{ii};
 
        expsz = floor(max(3, size(C, 1)/10));
        
        if C(1,1) > be
            expval = mean(C(1:expsz, :));
            C = [zeros(2, size(C, 2)); C];
            C(1,:) = [be expval(2:end)];
            C(2,:) = [C(3,1)-eps(C(3,1)) expval(2:end)];
        end
        if C(end,1) < en
            expval = mean(C(end-expsz+1:end, :));
            C = [C; zeros(2, size(C, 2))];
            C(end-1,:) = [C(end-2,1)+eps(C(end-2, 1)) expval(2:end)];
            C(end,:) = [en expval(2:end)];
        end
        B{ii} = C;
    end
end

% Setup resampling strategy; we resample on average once per timestep.
%step = (max(times)-min(times))/(length(times)/length(B));
step = (en-be)/(length(times)/length(B));
%t = [min(times):step:max(times)]';
t = be:step/subsampling:en;

if nargout > 1
    D = zeros(length(t), size(B{1},2)-1, length(B));

    % Do resampling.
    for ii = 1:length(B)
        C = B{ii};
        
        % Handle duplicate values by averaging
        C2 = zeros(size(C));
        jj2 = 1;
        c = C(1,:);
        n = 1;
        
        for jj = 2:size(C, 1)
            if n*C(jj,1) ~= c(1)
                C2(jj2, :) = c/n;                
                jj2 = jj2 + 1;
                
                c = C(jj,:);
                n = 0;
            else
                c = c + C(jj,:);
            end
            
            n = n + 1;
        end
        C2(jj2, :) = c/n;
        C2 = C2(1:jj2,:);

        % Resample
        for jj = 2:size(C2, 2)
            yi = interp1(C2(:, 1), C2(:, jj), t, 'linear');
            D(:, jj-1, ii) = yi';
        end
    end
end

m = zeros(size(D, 1), size(D, 2));
s = zeros(size(D, 1), size(D, 2));
ci95 = zeros(size(D, 1), size(D, 2));
ND = squeeze(~isnan(D(:,1,:)));
n = sum(ND, 2);
sn = sqrt(n);
cutoff = abs(tinv(0.05/2, n-1)); % ivan: tinv is a one-sided function
cutoff(n == 1) = 0; % if there is only one data point then ci95 = 0

% Gather statistics
for ii = 1:size(D, 1)
    m(ii, :) = mean(D(ii, :, ND(ii,:)), 3);
    s(ii, :) = std(D(ii, :, ND(ii,:)), 0, 3);
    ci95(ii, :) = s(ii, :)*cutoff(ii)/sn(ii);
end
