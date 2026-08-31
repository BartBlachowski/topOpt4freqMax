function p=paths()
p.phase2b=fileparts(fileparts(mfilename('fullpath')));p.repo=fileparts(fileparts(p.phase2b));
p.runs=fullfile(p.phase2b,'qualification_runs');
p.outputs=fullfile(p.phase2b,'outputs');p.phase2a=fullfile(p.repo,'analysis','iteration_efficiency_phase2a');
end
