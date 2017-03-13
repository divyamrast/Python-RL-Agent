function I = sortstruct(settings, fields)
%SORTSTRUCT Sorts a structure array by fields
%   I = SORTSTRUCT(STRUCTARR, FIELD) sorts the structure array STRUCTARR
%   by field FIELD. STRUCTARR(I) is sorted.
%   I = SORTSTRUCT(STRUCTARR, {FIELD1, FIELD2, ...}) sorts first by
%   FIELD1, then by FIELD2, etc.
%
%   Author:
%      Wouter Caarls <w.caarls@tudelft.nl>

    % Single-field case
    if ~iscell(fields)
        fields = {fields};
    end

    % Name lookup
    names = fieldnames(settings(1));
    cols = zeros(1, length(fields));
    
    for ii = 1:length(fields)
        cols(ii) = find(strcmp(fields{ii}, names));
    end
    
    % Make into a cell array with every row an entry, every column a field
    c = struct2cell(settings);
    sz = size(c);
    c = reshape(c, sz(1), [])';
    
    % Sort
    % http://newsgroups.derkeiler.com/Archive/Comp/comp.soft-sys.matlab/2006-05/msg02238.html
    for i=length(cols):-1:1,
        B = c(:,cols(i));
        if ~cellfun('isclass',B,'char')
            B = [B{:}];
        end
        [~,ix] = sort(B);
        c = c(ix,:);
        if i == length(cols)
            I = ix;
        else
            % Keep track of sorting of original data
            I = ix(I);
        end
    end
