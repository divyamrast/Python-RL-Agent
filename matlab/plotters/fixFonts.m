function fixFonts(fontSize, fontFamily)

    set(gca, 'FontName', fontFamily,'FontSize', fontSize);
       
    % Change default axes fonts.
    set(gcf,'DefaultAxesFontName', fontFamily);
    set(gcf,'DefaultAxesFontSize', fontSize);

    % Change default text fonts.
    set(gcf,'DefaultTextFontName', fontFamily);
    set(gcf,'DefaultTextFontSize', fontSize);
    
    % title
    h = get(gca, 'title');
    set(h, 'FontName', fontFamily);
    set(h, 'FontSize', fontSize);
    
    % xlabel
    h = get(gca, 'xlabel');
    set(h, 'FontName', fontFamily);
    set(h, 'FontSize', fontSize);
    
    % ylabel
    h = get(gca, 'ylabel');
    set(h, 'FontName', fontFamily);
    set(h, 'FontSize', fontSize);
end
