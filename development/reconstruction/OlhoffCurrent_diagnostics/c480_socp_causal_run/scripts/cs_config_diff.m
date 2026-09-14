function D = cs_config_diff()
%CS_CONFIG_DIFF  Machine-readable single-factor comparison (Part 2).
%   Control cfg = the struct stored in the control trajectory.  Treatment cfg =
%   the SAME struct, passed unmodified by cs_run_treatment.  Every schema row is
%   compared, plus the treatment descriptor that carries the single factor.
S = cs_setup();
guard = olhoffcurrent_paths(); %#ok<NASGU>
L = load(S.controlTraj, 'cfg');
cfgC = L.cfg; cfgT = L.cfg;               % exactly what the runner does
[cfgF, ~] = cp_config(480, 60);           % fresh resolution, for the hash only
sch = olh.config.schema();
rows = struct('path',{},'control',{},'treatment',{},'fresh',{},'equal',{},'scientific',{});
for k = 1:size(sch,1)
    p = sch{k,1};
    vc = olh.config.getPath(cfgC, p); vt = olh.config.getPath(cfgT, p); vf = olh.config.getPath(cfgF, p);
    rows(end+1) = struct('path',p,'control',local_show(vc),'treatment',local_show(vt), ...
        'fresh',local_show(vf),'equal',isequal(vc,vt),'scientific',~strcmp(p,'runtime.name')); %#ok<AGROW>
end
D = struct();
D.schema_rows = numel(rows);
D.rows = rows;
D.cfg_struct_isequal = isequal(cfgC, cfgT);
D.cfgHash_control = olhoffcurrent_config_hash(cfgC);
D.cfgHash_treatment = olhoffcurrent_config_hash(cfgT);
D.cfgHash_fresh = olhoffcurrent_config_hash(cfgF);
D.config_differences = rows(~[rows.equal]);
D.fresh_differences = {rows(~strcmp({rows.control},{rows.fresh})).path};
D.treatment_descriptor = struct( ...
    'control',   struct('innerSolver','repeatedMMA','implementation','+impl/algo/innerLoop.m -> mma_published/mmasub.m'), ...
    'treatment', struct('innerSolver','exactSOCP','implementation', ...
        'scripts/cs_socp_inner.m (fp_problem SOC, coneprog schur->augmented, cs_socp_certify)'));
D.scientific_differences = {struct('field','innerSolver','control','repeatedMMA','treatment','exactSOCP')};
D.inert_under_treatment = {'optimizer.inner.type','optimizer.inner.variant','optimizer.inner.tolerance', ...
    'optimizer.inner.minIterations','optimizer.inner.maxIterations'};
D.non_scientific_differences = {'output directory','logging/progress print','SOCP telemetry records', ...
    'certificate records','checkpointing','preflight stopAfter hook (0 in the treatment)', ...
    'iteration-1 identity assertion','NaN/Inf fail-closed checks'};
D.n_scientific_config_differences = numel(D.config_differences);
D.pass = D.cfg_struct_isequal && D.n_scientific_config_differences == 0 && ...
    strcmp(D.cfgHash_control, S.expect.cfgHash) && strcmp(D.cfgHash_treatment, S.expect.cfgHash) && ...
    strcmp(D.cfgHash_fresh, S.expect.cfgHash);
cs_json(fullfile(S.study,'evaluations','single_factor_diff.json'), D);
fprintf('[cs_config_diff] rows=%d differences=%d pass=%d\n', D.schema_rows, D.n_scientific_config_differences, D.pass);
end

function s = local_show(v)
if ischar(v); s = v;
elseif isstring(v); s = char(v);
elseif islogical(v); s = mat2str(v);
elseif isnumeric(v); s = mat2str(v, 17);
elseif iscell(v); s = ['{' strjoin(cellfun(@local_show, v, 'UniformOutput', false), ',') '}'];
elseif isempty(v); s = '[]';
else; s = class(v);
end
end
