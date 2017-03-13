function [h, he] = ebp(x, varargin)
%ERRORBARALPHA Plot data with alpha-shaded errorbars.
%   ERRORBARALPHA(Y) is the same as PLOT(1:LENGTH(Y), Y).
%   ERRORBARALPHA(Y, E) additionally plots the errors E around Y.
%   ERRORBARALPHA(X, Y, E) plots Y against X.
%   ERRORBARALPHA(..., 'Color', C) sets the line and error color to C.
%   ERRORBARALPHA(..., 'ErrorColor', C) sets error color to C.
%   ERRORBARALPHA(..., 'Alpha', A) sets the transparency to A.
%   ERRORBARALPHA(..., 'Rendering', MODE) sets the rendering mode.
%      MODE is one of 'alpha' : use OpenGL alpha channel.
%                     'opaque': draw light-shaded opaque area.
%                     'edge'  : draw only edge of area.
%   ERRORBARALPHA(..., 'LineWidth', W) sets the line width to W.
%   ERRORBARALPHA(..., 'LineStyle', S) sets the line style to S.
%
%   H = ERRORBARALPHA(...) returns a handle to the main line.
%   [H, HE] = ERRORBARALPHA(...) also returns a handle to the error area.
%
%   EXAMPLES:
%      errorbaralpha(sin(-pi:0.1:pi), rand(1, 63));
%      errorbaralpha(cos(-pi:0.1:pi), rand(1, 63), 'color', 'g', ...
%                    'linestyle', '--');
%      errorbaralpha(sin(2*[-pi:0.1:pi]), rand(1, 63), 'color', 'r',
%                    'errorcolor', [.5 .5 .5], 'rendering', 'edge');
%      
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl

% Always need at least data (y)
if nargin < 1
    error('Missing data');
end

% Check first argument
if numel(x) ~= length(x)
    error('Input must be one-dimensional');
end

% Setup
len = length(x);
ih = ishold;

% Argument parsing
p = inputParser;
p.addOptional('y', [], @(x)(isnumeric(x)&&numel(x)==len));
p.addOptional('eu', [], @(x)(isnumeric(x)&&numel(x)==len));
p.addOptional('el', [], @(x)(isnumeric(x)&&numel(x)==len));
p.addParamValue('alpha', 0.25, @(x)(isnumeric(x)&&numel(x)==1));
p.addParamValue('rendering', 'alpha', @(x)strcmpi(x, 'alpha')||strcmpi(x, 'opaque')||strcmpi(x, 'edge')||strcmpi(x, 'errorbar'));
p.addParamValue('mode', 'lin', @ischar);
p.addParamValue('lineProp', struct('lineStyle', '-', 'lineWidth', 1, 'color', 'k'));
p.parse(varargin{:});
r = p.Results;
r.x = x;

% Handle optional X
if isempty(r.y)
    r.y = r.x;
    r.x = 1:len;
end
if isempty(r.eu)
    % NOTE: if there are 2 arguments, we assume that the first is X
    % if it is monotonically rising. Otherwise, we assume it's Y.
    if ~ismonotonic(r.x)
        r.eu = r.y;
        r.y = r.x;
        r.x = 1:len;
    end
end

% Handle row-vectors and column-vectors alike
if size(r.x, 1) > size(r.x, 2)
    r.x = r.x';
end
if size(r.y, 1) > size(r.y, 2)
    r.y = r.y';
end
if size(r.eu, 1) > size(r.eu, 2)
    r.eu = r.eu';
end
if size(r.el, 1) > size(r.el, 2)
    r.el = r.el';
end

% Default error color
if ~isfield(r.lineProp,'errorColor')
    r.lineProp.errorColor = r.lineProp.color;
end

% Color characters
if ischar(r.lineProp.color)
    r.lineProp.color = translatecolorchar(r.lineProp.color);
end
if ischar(r.lineProp.errorColor)
    r.lineProp.errorColor = translatecolorchar(r.lineProp.errorColor);
end

if strcmpi(r.mode, 'semilogx')
    logx = true;
    logy = false;
elseif strcmpi(r.mode, 'semilogy')
    logx = false;
    logy = true;
elseif strcmpi(r.mode, 'loglog')
    logx = true;
    logy = true;
else
    logx = false;
    logy = false;
end    

% Setup plot
newplot
hold on


% Deal with logplots
x = r.x;
y = r.y;

if logx
    x  = log10(r.x);
end

if logy
    y  = log10(y);
end

% Plot error area
if ~isempty(r.eu)
    % Patch definition
    px = [r.x(1:end) r.x(end:-1:1)];
    
    if isempty(r.el)
        r.el = r.eu;
    end
    
    py = [r.y(1:end)-r.el(1:end) r.y(end:-1:1)+r.eu(end:-1:1)];
    ym = r.y-r.el;
    yp = r.y+r.eu;
    
    if logx
        px = log10(px);
    end
    
    if logy
        py = log10(py);
        ym = log10(ym);
        yp = log10(yp);
    end

    if strcmpi(r.rendering, 'edge')
        % Plot edges of area
        he = plot(x, ym, 'Color', r.lineProp.errorColor);
        he = [he plot(x, yp, 'Color', r.lineProp.errorColor)];
    elseif strcmpi(r.rendering, 'alpha')
        % Use OpenGL alpha channel
        he = patch(px, py, r.lineProp.errorColor, 'EdgeColor', 'none', 'FaceAlpha', r.alpha);
    elseif strcmpi(r.rendering, 'errorbar')
        % do nothing
    else
        % Just plot a lightly-shaded area
        he = patch(px, py, r.alpha*r.lineProp.errorColor + (1-r.alpha)*[1 1 1], 'EdgeColor', 'none');
    end
end

% remove properties invalid for plotting
r.lineProp = rmfield(r.lineProp, 'errorColor');

% Plot main data
if (strcmpi(r.rendering, 'errorbar')) && (~isempty(r.eu))
    h = errorbar(x, y, r.el, r.eu, r.lineProp);
elseif length(x) > 1
    h = plot(x, y, r.lineProp);
else
    h = scatter(x, y, [], r.lineProp.color, r.lineProp.marker, 'filled');
    
%     %Obtain the axes size (in axpos) in Points
%     currentunits = get(gca,'Units');
%     set(gca, 'Units', 'Points');
%     axpos = get(gca,'Position');
%     set(gca, 'Units', currentunits);
%     markerWidth = r.lineProp.MarkerSize/diff(xlim)*axpos(3); % Calculate Marker width in points
%     set(h, 'SizeData', markerWidth^2)
    set(h, 'SizeData', 100)
end
if ~ih
    hold off
end

% Avoid console output
if ~nargout
    clear h he
end

function y = ismonotonic(x)
    y = all(x(2:end)-x(1:end-1)>0);
end

function out=translatecolorchar(in)
    switch(in)
        case 'k', out=[0 0 0];
        case 'w', out=[1 1 1];
        case 'r', out=[1 0 0];
        case 'g', out=[0 1 0];
        case 'b', out=[0 0 1];
        case 'y', out=[1 1 0];
        case 'm', out=[1 0 1];
        case 'c', out=[0 1 1];
        otherwise
            out=in;
    end
end
end

    