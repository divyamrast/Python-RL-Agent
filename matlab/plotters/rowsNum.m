function [rows] = rowsNum(path)
    files = dir(path);
    folder = fileparts(path);

    if isempty(files)
        error(['No files matching ''' path '''']);
    end

    % precalculate length of data to read and number of files to read
    rows = getFileSize( fullfile(folder, files(1).name) );
    for i = 2:length(files)
        numLines = getFileSize( fullfile(folder, files(i).name) );
        rows = min([rows, numLines]);
    end
end

function numLines = getFileSize(fileName)
    fid = fopen(fileName, 'rb');
    %# Get file size.
    fseek(fid, 0, 'eof');
    fileSize = ftell(fid);
    frewind(fid);
    %# Read the whole file.
    data = fread(fid, fileSize, 'uint8');
    %# Count number of line-feeds and increase by one.
    numLines = sum(data == 10);
    fclose(fid);
end