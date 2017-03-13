function exponentizeaxis( ax, exponent, location )
%EXPONENTIZEAXIS Turn y axis into exponent notation
%   EXPONENTIZEAXIS modifies the Y axis ticks into exponent notation
%      with a shared exponent at the top of the axis.
%   EXPONENTIZEAXIS(AX, EXPONENT, LOCATION) allows the specification of
%      the axis the exponentize, the exponent, and the location of the
%      axis ('left' or 'right')
%
%   AUTHOR:
%      Grant Martin and Wouter Caarls <w.caarls@tudelft.nl>

if nargin < 3
    location = 'left';
end

if nargin < 1
    ax = gca;
end

oldLabels = str2num(get(ax,'YTickLabel'));

if nargin < 2
    exponent = ceil(log10(abs(max(oldLabels))));
end

scale = 10^exponent;
newLabels = num2str(oldLabels/scale);
set(ax,'YTickLabel',newLabels,'units','normalized');
posAxes = get(ax,'position');
textBox = annotation('textbox','linestyle','none','string',['x 10\it^{' num2str(exponent) '}']);
posAn = get(textBox,'position');

if strcmp(location, 'left')
    set(textBox,'position',[posAxes(1)-posAn(3)/2 posAxes(2)+posAxes(4) posAn(3) posAn(4)],'VerticalAlignment','cap');
else
    set(textBox,'position',[posAxes(1)+posAxes(3)-posAn(3)/2 posAxes(2)+posAxes(4) posAn(3) posAn(4)],'VerticalAlignment','cap');
end    

end

