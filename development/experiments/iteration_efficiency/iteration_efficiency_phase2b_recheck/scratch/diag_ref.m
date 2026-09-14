repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck'));
[p,guard]=ie2br.setup_paths(); %#ok<ASGLU>
S=load(fullfile(p.runs,'probe_96x12_H1600.mat'));
r=S.ref; g=r.gain;
be=(100:100:1600).';
fprintf(' b      gainE1      gainE2      gainE3   allle1e-3   F_E1\n');
for i=1:numel(be)
  b=be(i);
  fprintf('%4d  %10.3e %10.3e %10.3e      %d      %.6g\n',b,g(b,1),g(b,2),g(b,3),r.freeze_candidate(b),r.F(b,1));
end
fprintf('\nvalid window endpoints: %d of %d\n',nnz(r.valid_window_endpoint),numel(r.valid_window_endpoint));
fprintf('first valid window endpoint: %d\n',find(r.valid_window_endpoint,1));
fprintf('min over blocks of max gain: %.4e\n',min(max(g(be,:),[],2)));
