function [dataBank, rows] = readColunms(path, meas, rows, varargin)
    files = dir(path);
    folder = fileparts(path);

    if isempty(files)
        error(['No files matching ''' path '''']);
    end
    
    nVarargs = length(varargin);
    episodes_num = -1;
    if (nVarargs)
        episodes_num = varargin{1};
    end
    
    strFormat = makeFormat(12); % expect no more then 12 colums per file
    %dataBank = zeros(rows, length(files));
    for i = 1:length(files)
        fid = fopen( fullfile(folder, files(i).name) );
        c = textscan(fid, strFormat,'HeaderLines',0,'Delimiter','\t','CollectOutput',1);
        fclose(fid);

        data = c{1,1};
        nrows = rows;
        if episodes_num > -1
            % obtain first index after which episode > episodes_num
            nrows = find( data(1:rows, 1) > episodes_num, 1);
            if isempty(nrows)
                nrows = rows;
            else
                nrows = max([nrows-1 1]);
            end
        end
        dataBank(1:nrows, i) = data(1:nrows, meas);
    end
    rows = nrows;
end

function [strFormat] = makeFormat(maxColums)
    strFormat = '';
    for i = 1:maxColums
        strFormat = strcat(strFormat, '%f ');
    end
end