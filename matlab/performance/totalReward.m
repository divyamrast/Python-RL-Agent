function [R] = totalReward(files, Tsampling, Tend)
    addpath('/media/ivan/ivan/scripts/matlab/csv/');

    R = zeros(length(files), 1);
    for i=1:length(files)
       % if (files(i), '.csv')
            % Read data
            [data, ~] = readCsv(files{i});
            
            t = 0;
            row = 1;
            newData = zeros(floor(Tend/Tsampling)+1, size(data,2));
            while (t < Tend)
                [~, idx] = min(abs(data(:,1)-t));
                newData(row, :) = data(idx, :);
                newData(row, 1) = t; % change time to the nearest
                t = t + Tsampling;
                row = row + 1;
            end
            timeSteps = Tsampling*ones(size(newData(:,1)));
            
            % calculate number of episodes, to normalise later
            % !!! Most crusual thing in calculating from CSV
            starts = (newData(:, 2)==0) & (newData(:, 4)==0) & (newData(:, 5) == 0);
        
            numEpisodes = sum(starts);
            timeSteps(starts) = 0; 
            
            R(i) = cartpoleRewards(timeSteps, newData(:, 2), newData(:, 3), ...
                newData(:, 4), newData(:, 5));
            R(i) = R(i) / numEpisodes;
%         else
%             % try to load cr file
%             fileName = sprintf('%s/sd_%0.2f.cr', folder, modelChanges(i));
%             [l, p] = readCr(fileName, dispStep, dispOffset, playTrails);
%             
%             if (isempty(l) || isempty(p))
%                 continue;
%             end
% 
%             M(end+1) = i;
%             if (dispStep)
%                 l = l' / (dispStep - dispOffset - playTrails);
%                 C(end+1,:) = l;
%             else
%                 C(i) = l(end) / (size(l, 1)-1);
%             end
%             
%             if (playTrails ~= 0)
%                 p = p' / playTrails;
%                 P(end+1,:) = p;
%             end
%         end
    end
    
end