function titleStr = formatTopologyTitle(approachName, volfrac, omega1)
%FORMATTOPOLOGYTITLE Build a standardized topology plot title.

name = char(string(approachName));
volPercent = 100 * volfrac;

if nargin < 3 || isempty(omega1) || ~isfinite(omega1)
    titleStr = sprintf('%s | vol=%.1f%%', name, volPercent);
    return;
end

omegaHz = omega1 / (2*pi);
omegaChar = char(969);
subOne    = char(8321);   % Unicode U+2081 SUBSCRIPT ONE
titleStr = sprintf('%s | vol=%.1f%% | %s%s=%.3f rad/s (%.3f Hz)', ...
    name, volPercent, omegaChar, subOne, omega1, omegaHz);
end
