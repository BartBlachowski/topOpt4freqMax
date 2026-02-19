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
titleStr = sprintf('%s | vol=%.1f%% | %s1=%.3f rad/s (%.3f Hz)', ...
    name, volPercent, omegaChar, omega1, omegaHz);
end
