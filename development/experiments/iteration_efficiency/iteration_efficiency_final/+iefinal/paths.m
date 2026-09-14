function p=paths()
%PATHS Repository-relative final-harness paths.
p.root=fileparts(fileparts(mfilename('fullpath')));
p.repo=fileparts(fileparts(p.root));
p.runs=fullfile(p.root,'runs');
p.manifest=fullfile(p.root,'PRODUCTION_MANIFEST.json');
p.schema=fullfile(p.root,'RESULT_SCHEMA.json');
p.contract=fullfile(p.repo,'analysis','iteration_efficiency_phase2a','iteration_efficiency_contract.json');
end
