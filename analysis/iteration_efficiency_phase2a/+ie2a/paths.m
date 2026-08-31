function p = paths()
%PATHS Resolve the isolated Phase-2A and repository paths without cwd assumptions.
p.phase2a = fileparts(fileparts(mfilename('fullpath')));
p.repo = fileparts(fileparts(p.phase2a));
p.contract = fullfile(p.phase2a, 'iteration_efficiency_contract.json');
p.validation = fullfile(p.phase2a, 'validation_outputs');
p.production = fullfile(p.phase2a, 'production_results');
end
