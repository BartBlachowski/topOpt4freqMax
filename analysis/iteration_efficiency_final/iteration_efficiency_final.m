function summary = iteration_efficiency_final(runMode,olhoffVariant)
%ITERATION_EFFICIENCY_FINAL Paper-facing iteration-efficiency entry point.
% User-facing configuration:
%   runMode       = 'smoke'; % smoke | production
%   olhoffVariant = 'lp';    % lp | mma | both
%
% Production is intentionally locked pending the final pre-production audit.
if nargin<1||isempty(runMode),runMode='smoke';end
if nargin<2||isempty(olhoffVariant),olhoffVariant='lp';end

here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(here,fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
    fullfile(repo,'analysis','three_method_parametric_study'), ...
    fullfile(repo,'analysis','iteration_efficiency_study_design'), ...
    fullfile(repo,'tools','Matlab'));
summary=iefinal.run(char(runMode),char(olhoffVariant));
end
