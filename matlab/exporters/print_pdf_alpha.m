function print_pdf_alpha(filenamebase, export_area_drawing, export_latex)
%PRINT_PDF_ALPHA Export figure to pdf while preserving transparency areas.
%   This function is based on plot2svg
%  
%   AUTHOR:
%      Erik Schuitema <e.schuitema@tudelft.nl

% Always need at least filename base (x)
if nargin < 1
    error('Missing filename base');
end

if nargin < 2
    s_export_area_drawing = '--export-area-drawing';
else
    if export_area_drawing
        s_export_area_drawing = '--export-area-drawing';
    else
        s_export_area_drawing = '';
    end
end

if nargin < 3
    s_export_latex = '';
else
    if export_latex
        s_export_latex = '--export-latex';
    else
        s_export_latex = '';
    end
end

%plot2svg(strcat(filenamebase,'.svg'),1); % not working after R2014a
print([filenamebase '.svg'], '-dsvg');
system(['env -u LD_LIBRARY_PATH inkscape -T -f  --export-background-opacity=0 ',filenamebase,'.svg ', s_export_area_drawing, ' ', s_export_latex, ' --export-pdf=',filenamebase,'.pdf']);
system(['rm ',filenamebase,'.svg']);
end

    
