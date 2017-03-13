function print_pdf_alpha(filenamebase)
%PRINT_PDF_ALPHA Export figure to pdf while preserving transparency areas.
%   This function is based on plot2svg
%  
%   AUTHOR:
%      Erik Schuitema <e.schuitema@tudelft.nl

% Always need at least filename base (x)
if nargin < 1
    error('Missing filename base');
end

plot2svg(strcat(filenamebase,'.svg'),1);
system(['env -u LD_LIBRARY_PATH inkscape -T -f ',filenamebase,'.svg --export-pdf=',filenamebase,'.pdf']);
system(['rm ',filenamebase,'.svg']);
end

    
