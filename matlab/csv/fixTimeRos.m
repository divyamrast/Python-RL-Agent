function [dataFixed] = fixTimeRos(data)
    dataFixed = (data - data(1)) / 1E9;
end