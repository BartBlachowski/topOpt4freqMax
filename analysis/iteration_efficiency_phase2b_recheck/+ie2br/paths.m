function p=paths()
%PATHS Isolated Phase-2B recheck paths.
p.phase2br=fileparts(fileparts(mfilename('fullpath')));
p.repo=fileparts(fileparts(p.phase2br));
p.runs=fullfile(p.phase2br,'qualification_runs');
p.outputs=fullfile(p.phase2br,'outputs');
p.phase2a=fullfile(p.repo,'analysis','iteration_efficiency_phase2a');
p.phase2b=fullfile(p.repo,'analysis','iteration_efficiency_phase2b_precision');
end
