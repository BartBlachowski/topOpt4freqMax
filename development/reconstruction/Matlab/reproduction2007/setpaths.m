function setpaths()
%SETPATHS  Put the project folders on the MATLAB path.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'fem'), fullfile(here,'filter'), ...
        fullfile(here,'algo'), fullfile(here,'mma'), fullfile(here,'runs'));
end
