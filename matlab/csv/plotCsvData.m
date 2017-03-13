function plotCsvData(data, axisLabels, plotVars, plotOptions, lineOptions, fixDiff)

[~, LabelsNum] = size(axisLabels);

%fix differences between "our" data representation and Manuel's
if fixDiff
    
    %invert cart positions
    if (LabelsNum >= 2)
        data(:, 2) = -data(:, 2);
    end
    
    %invert cart velocity
    if (LabelsNum >= 3)
        data(:, 3) = -data(:, 3);
    end
    
    % shift and invert pendulum angle
    if (LabelsNum >= 4)
        mi = data(:, 4) > pi;
        data(mi, 4) = mod( data(mi, 4), pi) - pi;
        data(:, 4) = -(data(:, 4) - pi);
    end
    
    %invert pendulum velocity
    if (LabelsNum >= 5)
        data(:, 5) = -data(:, 5);
    end
end

subplotsNum = sum(plotVars > 0);
subplotsX = floor(sqrt(subplotsNum));
subplotsY = ceil(subplotsNum/subplotsX);

for i = 2:LabelsNum
    if plotVars(i)
        if (LabelsNum > 2)
            subplotIndex = plotVars(i);
            subplot(subplotsX, subplotsY, subplotIndex);
        end
        hold on;
        if strcmp(plotOptions, 'plot')
            plot(data(:, 1), data(:, i), lineOptions);
        elseif strcmp(plotOptions, 'stairs')
            stairs(data(:, 1), data(:, i), lineOptions);
        end
        grid on;
        xlabel(axisLabels{1});
        ylabel(axisLabels{i});
    end
end

end