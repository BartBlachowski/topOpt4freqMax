function name = confbench_display_name(methodKey)
%CONFBENCH_DISPLAY_NAME  The label a method carries in every table and export.
%
%   The Olhoff column is "Du-Olhoff reconstruction (M4)".  It must NOT be
%   called "Olhoff 2007": what runs is a reconstruction of the published nested
%   formulation whose continuation and inner-convergence details the paper does
%   not uniquely determine.  See OLHOFFM4_CAVEAT.
switch lower(char(string(methodKey)))
    case 'olhoff';                    name = 'Du-Olhoff reconstruction (M4)';
    case 'yuksel';                    name = 'Yuksel';
    case {'proposed','ourapproach'};  name = 'Proposed';
    otherwise
        error('confbench_display_name:UnknownMethod', 'Unknown method "%s".', methodKey);
end
end
