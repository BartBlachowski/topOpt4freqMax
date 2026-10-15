function wstudy_export(campaignDir, outFile)
%WSTUDY_EXPORT  Final designs + objective histories of a campaign -> v7 .mat for Python.
S = load(fullfile(campaignDir, 'benchmark_records.mat'), 'records');
r = S.records; if ~iscell(r); r = num2cell(r); end
E = struct('method', {}, 'nelx', {}, 'nely', {}, 'x', {}, 'obj', {});
for k = 1:numel(r)
    q = r{k};
    if isfield(q, 'is_warmup') && q.is_warmup; continue; end
    E(end+1) = struct('method', q.method_key, 'nelx', q.mesh(1), 'nely', q.mesh(2), ...
        'x', q.x(:), 'obj', q.telemetry.objective_history(:)); %#ok<AGROW>
end
save(outFile, 'E', '-v7');
fprintf('exported %d records from %s\n', numel(E), campaignDir);
end
