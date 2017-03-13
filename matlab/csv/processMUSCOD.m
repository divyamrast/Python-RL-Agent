function y = processMUSCOD(x)

    dt = 0.01;
    t = x(:, 1);
    t = int64(round(t/dt));

    [~,ind] = unique(t, 'rows', 'last');

   
    y = x(ind, :);

    s = y(:, 2);
    s(s>0)=1;
    s(s<0)=0;
    c = 1 + find(diff(s)==1);

    y(c, 6) = 1;
end