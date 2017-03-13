function [data, axisLabels] = readCsv(fileName)

path = fileparts(fileName);

if strfind(fileName, '*')
    csvFileList = dir(fileName);
    num = length(csvFileList);
    if num == 1
        [data, axisLabels] = readCsvSingle(char({csvFileList.name}));
    else
        data = cell([num, 1]);
        axisLabels = cell([num, 1]);
        k = 1;
        for file = {csvFileList.name}
            [data{k}, axisLabels{k}] = readCsvSingle(char(fullfile(path, file)));
            k = k + 1;
        end
    end
else
    [data, axisLabels] = readCsvSingle(fileName);
end

end


function [data, axisLabels] = readCsvSingle(fileName)
    fid = fopen(fileName);

    if fid == -1
        error('File %s not found.',fileName);
    end

    reading_axis = 0;
    tline = '';
    axisLabels = [];
    while ischar(tline)

        tline = fgets(fid);
        tline = strtrim(tline);

        if (strcmp(tline, 'COLUMNS:'))
            reading_axis = 1;
            continue;
        end

        if (strcmp(tline, 'DATA:'))
            break;
        end

        if (reading_axis)
            C = strsplit(tline, ',');
            C(cellfun('isempty', C)) = [];
            axisLabels = [axisLabels, C];
        end

        % if there was is a meshup header then we should read the axes,
        % otherwice the csv file is without a header
        if (reading_axis == 0)
            fseek(fid,0,-1); % return back to the beginning
            break;
        end
    end
    [~, LabelsNum] = size(axisLabels);

    strFormat = '';
    for i = 1:LabelsNum
        strFormat = strcat(strFormat, '%f ');
    end

    c = textscan(fid,strFormat,'HeaderLines',0,'Delimiter',',','CollectOutput',1);
    fclose(fid);

    data = c{1,1};    
end