function print_eps_alpha(filenamebase)
%PRINT_EPS_ALPHA Export figure to eps while preserving transparency areas.
%   This function is based on plot2svg
%  
%   AUTHOR:
%      Ivan Koryakovskiy <i.koryakovskiy@gmali.com>

% Always need at least filename base (x)

if nargin < 1
    error('Missing filename base');
end 

set(gcf, 'color', 'none'); % transparent background for tight bb

    plot2svg(strcat(filenamebase,'.svg'),1);
    system(['inkscape ',filenamebase,'.svg --export-eps=',filenamebase,'.eps']);
    %system(['epstool --copy --bbox ', filenamebase,'.eps _', filenamebase,'.eps']);
    system(['rm ',filenamebase,'.svg']);

set(gcf, 'color', 'w'); % back to white
end
