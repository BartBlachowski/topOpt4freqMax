function res = run_case(name, over)
%RUN_CASE  Execute one configured experiment and save it under results/.
%   over is a struct of overrides applied to defaultCfg().
cfg = defaultCfg();
if nargin>1 && ~isempty(over)
    fn = fieldnames(over);
    for i=1:numel(fn), cfg.(fn{i}) = over.(fn{i}); end
end
cfg.name = name;
fprintf('=== run "%s" ===\n', name);
disp(cfg);
res = olhoffOpt(cfg);
if ~exist('results','dir'), mkdir('results'); end
save(fullfile('results',[name '.mat']),'res','-v7.3');
fprintf('\n--- %s: outer=%d  wallclock=%.1f s ---\n',name,res.nOuter,res.wallclock);
fprintf('final omega: '); fprintf('%.2f ',res.omega(1:min(4,end))); fprintf('\n');
fprintf('mode split Ex: '); fprintf('%.2f ',res.modeTable(1:min(4,end),2)); fprintf('\n');
for i=1:numel(res.log), fprintf('LOG: %s\n',res.log{i}); end
fprintf('mean t/iter: eig %.2fs  grad %.2fs  inner %.2fs\n', ...
        mean(res.hist.tEig),mean(res.hist.tGrad),mean(res.hist.tInner));
end
