function setpaths()
%SETPATHS  Put the project folders on the MATLAB path.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'fem'), fullfile(here,'filter'), ...
        fullfile(here,'algo'), fullfile(here,'mma'), fullfile(here,'runs'));
% Post-conference canonical layer.  Adding the PARENT of +olh is what makes the
% package resolvable as olh.config.*, olh.presets.*; it puts no bare function on
% the path and so cannot shadow anything in the folders above.
addpath(fullfile(here,'architecture'));
end
