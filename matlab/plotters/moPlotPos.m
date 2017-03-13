function AxisPos = moPlotPos(nCol, nRow, defPos, bshift, lshift)
% Position of diagrams - a very light SUBPLOT
% Tested: Matlab 6.5, 7.7, 7.8, WinXP
% Author: Jan Simon, Heidelberg, (C) 2005-2011
% This is the percent offset from the subplot grid of the plotbox.
% Formula: Space = a + b * n
% Increase [b] to increase the space between the diagrams.

if nRow < 3
   BplusT = bshift(1);                          % 0.18
else
   BplusT = bshift(1) + bshift(2) * nRow;       % 0.09 0.045
end

if nCol < 3
   LplusR = lshift(1);                          % 0.18
else
   LplusR = lshift(1) + lshift(2) * nCol;       % 0.09  0.03
end

nPlot = nRow * nCol;
plots = 0:(nPlot - 1);
row   = (nRow - 1) - fix(plots(:) / nCol);
col   = rem(plots(:), nCol);
col_offset  = defPos(3) * LplusR / (nCol - LplusR);
row_offset  = defPos(4) * BplusT / (nRow - BplusT);
totalwidth  = defPos(3) + col_offset;
totalheight = defPos(4) + row_offset;
width       = totalwidth  / nCol - col_offset;
height      = totalheight / nRow - row_offset;
if width * 2 > totalwidth / nCol
   if height * 2 > totalheight / nRow
      AxisPos = [(defPos(1) + col * totalwidth / nCol), ...
            (defPos(2) + row * totalheight / nRow), ...
            width(ones(nPlot, 1), 1), ...
            height(ones(nPlot, 1), 1)];
   else
       AxisPos = [(defPos(1) + col * totalwidth / nCol), ...
            (defPos(2) + row * defPos(4) / nRow), ...
            width(ones(nPlot, 1), 1), ...
            (0.7 * defPos(ones(nPlot, 1), 4) / nRow)];
   end
else
   if height * 2 <= totalheight / nRow
      AxisPos = [(defPos(1) + col * defPos(3) / nCol), ...
            (defPos(2) + row * defPos(4) / nRow), ...
            (0.7 * defPos(ones(nPlot, 1), 3) / nCol), ...
            (0.7 * defPos(ones(nPlot, 1), 4) / nRow)];
   else
      AxisPos = [(defPos(1) + col * defPos(3) / nCol), ...
            (defPos(2) + row * totalheight / nRow), ...
            (0.7 * defPos(ones(nPlot, 1), 3) / nCol), ...
            height(ones(nPlot, 1), 1)];
    end
end