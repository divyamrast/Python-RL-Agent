function s = val2str(c)        
%VAL2STR Convert arbitrary value to string
%   S = VAL2STR(C) converts C into a string. C can be
%   a numerical or string value, but also a struct or object.
%
%   AUTHOR:
%      Wouter Caarls <w.caarls@tudelft.nl>

    % Init
    s = [];
    
    if iscell(c)
        % Cell array
        lc = length(c);
        for i = 1:lc
            s = [s val2str(c{i})];
            if i < lc
                s = [s ','];
            end
        end
        s = ['{' s '}'];
    elseif ischar(c)
        % String
        s = ['''' c ''''];
    elseif isempty(c) || numel(c) > 1
        % Vector. Can be of anything. NOTE: flattens matrix.
        lc = numel(c);
        for i = 1:lc
            s = [s val2str(c(i))];
            if i < lc
                s = [s ','];
            end
        end
        s = ['[' s ']'];
    elseif isstruct(c)
        % Struct. Here because a vector of structs passes this test.
        fn = fieldnames(c);
        lc = length(fn);
        for i = 1:lc
            s = [s fn{i} ':' val2str(c.(fn{i})) ];
            if i < lc
                s = [s ','];
            end
        end
        s = ['<' s '>'];
    elseif isobject(c)
        % Object. Here because a vector of objects passes this test.
        s = class(c);
    elseif isa(c, 'function_handle')
        % Function handle
        s = ['@' func2str(c)];
    else
        % Regular value
        s = num2str(c);
    end
end
