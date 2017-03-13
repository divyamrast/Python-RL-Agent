function [h, name] = plotCumulativeRegretSingle(folder, config, lineType, lineColour, lineWidth, name)

    modelChanges = config.modelChanges;
    timeStep = config.timeStep;
    dispStep = config.dispStep;
    dispOffset = config.dispOffset;
    episodeNum = config.episodeNum;
    baseline = config.baseline;
    playTrails = config.playTrails;
    plotPlays = config.plotPlays;

    [~, model_changes_num] = size(modelChanges);
    C = [];%zeros(model_changes_num, 1);
    M = [];
    P = []; %zeros(model_changes_num, 1);
    for i=1:model_changes_num
        fileName = sprintf('%s/sd_%0.2f.csv', folder, modelChanges(i));
        [data, ~] = readCsv(fileName);
        
        if ~isempty(data)
            % csv file was loaded successfully
            if (timeStep == 0)
                timeStep = (data(end, 1) - data(1, 1)) / (size(data, 1)-1);
            end
            timeSteps = timeStep*ones(size(data(:,1)));
            
            % calculate number of episodes, to normalise later
            % !!! Most crusual thing in calculating from CSV
            starts = (data(:, 2)==0) & (data(:, 3)~=0) & (data(:, 4)==0) & (data(:, 5) == 0);
        
            numEpisodes = sum(starts)-1;
            timeSteps(starts) = 0; 
            
            C(end+1, :) = evaluate(timeSteps, data(:, 2), data(:, 3), ...
                data(:, 4), data(:, 5));
            C(end) = C(end) / numEpisodes;
            M(end+1) = i;
        else
            % try to load cr file
            fileName = sprintf('%s/sd_%0.2f.cr', folder, modelChanges(i));
            [l, p] = readCr(fileName, dispStep, dispOffset, playTrails);
            
            if (isempty(l) || isempty(p))
                continue;
            end

            M(end+1) = i;
            if (dispStep)
                l = l' / (dispStep - dispOffset - playTrails);
                C(end+1,:) = l;
            else
                C(i) = l(end) / (size(l, 1)-1);
            end
            
            if (playTrails ~= 0)
                p = p' / playTrails;
                P(end+1,:) = p;
            end
        end
    end
        
    baseline = baseline * episodeNum;

    if (~plotPlays)
        C = C * episodeNum;
        if (size(C,2) > 1)
            % can calculate mean and variance
            m = mean(C, 2);
            s = std(C, 0, 2);
            e = s/sqrt(size(C,2)); %calculating standart error
            h = errorbaralpha(modelChanges(M), baseline(M)-m, 1.96*e, ...
                'Color', lineColour, 'LineWidth', lineWidth, 'LineStyle', lineType, ...
                'mode', 'semilogx', 'rendering', 'alpha');
        else
            h = errorbaralpha(modelChanges(M), baseline(M)-C, 'Color', lineColour, ...
                'LineWidth', lineWidth, 'LineStyle', lineType, 'mode', 'semilogx', 'rendering', 'alpha');
        end
    else
        P = P * episodeNum;
        if (size(P,2) > 1)
            % can calculate mean and variance
            m = sum(P,2) ./ sum(P~=0,2); %rowMean

            %s = std(P, 0, 2);
            x = P'; % std of the columns
            index = (x ~= 0);
            sumnz = sum(index, 1);
            meanx = sum(x, 1) ./ sumnz;
            xm    = bsxfun(@minus, x, meanx);
            xm(~index) = 0;
            s     = sqrt(sum(xm .* xm, 1) ./ (sumnz - 1));

            e = s/sqrt(size(P,2)); %calculating standart error
            h = errorbaralpha(modelChanges(M), baseline(M)-m, 1.96*e, ...
                'Color', lineColour, 'LineWidth', lineWidth, 'LineStyle', lineType, ...
                'mode', 'semilogx', 'rendering', 'alpha');
        end
    end
    
end
