function data = readCsvDirect(fileName, colNum)
    fid = fopen(fileName);

    if fid == -1
        data = [];
        return;
    end

    strFormat = '';
    for i = 1:colNum
        strFormat = strcat(strFormat, '%f ');
    end

    c = textscan(fid,strFormat,'HeaderLines',0,'Delimiter',',','CollectOutput',1);
    fclose(fid);

    data = c{1,1};

end