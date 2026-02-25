function titleStr = formatTopologyTitle(approachName, volfrac, omega1, omega2)
%FORMATTOPOLOGYTITLE Build a standardized topology plot title.

name = char(string(approachName));
volPercent = 100 * volfrac;

if nargin < 3 || isempty(omega1) || ~isfinite(omega1)
    titleStr = sprintf('%s | vol=%.1f%%', name, volPercent);
    return;
end

omegaChar = char(969);
subOne    = char(8321);   % Unicode U+2081 SUBSCRIPT ONE
omegaHz = omega1 / (2*pi);
titleStr = sprintf('%s | vol=%.1f%% | %s%s=%.3f rad/s (%.3f Hz)', ...
    name, volPercent, omegaChar, subOne, omega1, omegaHz);

if nargin >= 4 && ~isempty(omega2) && isscalar(omega2) && isfinite(omega2)
    subTwo   = char(8322);   % Unicode U+2082 SUBSCRIPT TWO
    omega2Hz = omega2 / (2*pi);
    titleStr = sprintf('%s  |  %s%s=%.3f rad/s (%.3f Hz)', ...
        titleStr, omegaChar, subTwo, omega2, omega2Hz);
end
end
