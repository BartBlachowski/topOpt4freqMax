function name = confbench_display_name(methodKey)
%CONFBENCH_DISPLAY_NAME  The label a method carries in every table and export.
%
%   The Olhoff column carries the display name of the preset the benchmark
%   runs (CONFBENCH_OLHOFF_PRESET), which states its material law and controller:
%   "Du-Olhoff reconstruction (Pedersen stiffness + linear mass, adaptive box)".
%   It must NOT be called "Olhoff 2007", and must not reuse the label of the
%   historical SIMP + eq. (4b) realization ("Du-Olhoff reconstruction (M4)"),
%   which is a different formulation.  See OLHOFFCURRENT_CAVEAT.
switch lower(char(string(methodKey)))
    case 'olhoff';                    name = olhoffcurrent_preset(confbench_olhoff_preset()).displayName;
    case 'yuksel';                    name = 'Yuksel';
    case {'proposed','ourapproach'};  name = 'Proposed';
    otherwise
        error('confbench_display_name:UnknownMethod', 'Unknown method "%s".', methodKey);
end
end
