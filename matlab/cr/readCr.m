function [learning playing] = readCr(fileName, dispStep, dispOffset, playTrails)
    fid = fopen(fileName);
    if (fid == -1)
        disp(['Warning: Could not open '  fileName]);
        learning = [];
        playing = [];
        return;
    end
    cc = textscan(fid,'%f','HeaderLines',0,'Delimiter',',','CollectOutput',1);
    c = cc{1,1};
    fclose(fid);

    if (dispStep ~= 0)
        % select from c every element @ 'dispStep'
        selector = 1+dispStep:dispStep:size(c, 1);
        
        s_learning = c(selector-playTrails-dispOffset);
        b_learning = c(selector-dispStep);
        learning = s_learning - b_learning;
        
        s_playing = c(selector);
        b_playing = c(selector-playTrails);
        playing = s_playing - b_playing; 
    end
end