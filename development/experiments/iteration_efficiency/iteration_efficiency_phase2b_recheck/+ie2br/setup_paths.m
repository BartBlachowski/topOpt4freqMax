function [p,guard]=setup_paths()
%SETUP_PATHS Add every path the recheck harness needs.
%  GUARD must be retained by the caller: repro2007_paths returns an onCleanup
%  object that strips the reproduction paths as soon as it goes out of scope.
p=ie2br.paths();
addpath(p.phase2a, ...
    fullfile(p.repo,'analysis','three_method_parametric_study'), ...
    fullfile(p.repo,'analysis','olhoff_stabilization_audit'), ...
    fullfile(p.repo,'Matlab','reproduction2007','runner'));
guard=repro2007_paths();
end
