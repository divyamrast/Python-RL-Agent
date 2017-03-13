function h = loglog2(x, y, varargin)
%LOGLOG2 Base 2 log-log plot
%   H = LOGLOG2(X, Y, ...) plots X against Y in a base 2 log-log plot.
%
%   AUTHOR:
%      Wouter Caarls <w.caarl@tudelft.nl>

    h = plot(log2(x), log2(y), varargin{:});
    
    xmin = floor(min([get(gca, 'xtick') log2(x)]));
    xmax = ceil(max([get(gca, 'xtick') log2(x)]));
    xticks = xmin:xmax;
    
    set(gca,'xtick',xticks);
    set(gca,'xticklabel',[repmat('2^',length(xticks),1) num2str(xticks.')]);
    
    ymin = floor(min([get(gca, 'ytick') log2(y)]));
    ymax = ceil(max([get(gca, 'ytick') log2(y)]));
    yticks = ymin:ymax;
    
    set(gca,'ytick',yticks);
    set(gca,'yticklabel',[repmat('2^',length(yticks),1) num2str(yticks.')]);

end
