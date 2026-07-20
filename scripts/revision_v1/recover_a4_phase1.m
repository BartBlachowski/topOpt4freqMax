function summary = recover_a4_phase1(outDir)
%RECOVER_A4_PHASE1  Post hoc C-3/C-4 recovery without optimization.
%
%   Reuses the completed A4 .mat topology arrays. Only threshold endpoints and
%   hash-derived provenance are changed. Tracked frequencies, decisions,
%   classifications, trajectories, and topologies are asserted unchanged.

if nargin < 1 || isempty(outDir)
    repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    outDir = fullfile(repoRoot,'examples','Revision_v1','output','a4');
else
    repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
addpath(fullfile(repoRoot,'scripts','revision_v1'));

matPath = fullfile(outDir,'a4_eigenpair_refresh_results.mat');
loaded = load(matPath,'res');
original = loaded.res;
if isfield(original,'recovery_phase1')
    error('recover_a4_phase1:AlreadyRecovered', ...
        ['This artifact already carries Recovery Phase 1 metadata. Restore the ' ...
         'original optimization artifact before regenerating it.']);
end
res = original;
cfg = jsondecode(fileread(res.base_config));

originalHash = res.base_config_hash;
correctedHash = a4_hash_file(res.base_config);
trackedBefore = [res.arms.omega1_tracked];
thresholdBefore = [res.arms.omega1_thresholded];
topologyBefore = {res.arms.topology};

updates = repmat(struct('tag','','before',NaN,'after',NaN,'mode_index',0, ...
    'mac_to_solid',NaN,'configured_floor',NaN),0,1);
for i = 1:numel(res.arms)
    res.arms(i).base_config_hash = correctedHash;
    if isempty(res.arms(i).topology)
        continue;
    end
    ep = a4_threshold_endpoint_from_topology(res.arms(i).topology,cfg,20);
    u = struct('tag',res.arms(i).tag, ...
        'before',res.arms(i).omega1_thresholded, ...
        'after',ep.omega1_thresholded, ...
        'mode_index',ep.mode_index, ...
        'mac_to_solid',ep.mac_to_solid, ...
        'configured_floor',ep.configured_floor);
    updates(end+1,1) = u; %#ok<AGROW>
    res.arms(i).omega1_thresholded = ep.omega1_thresholded;
end
res.base_config_hash = correctedHash;

% Prove the in-memory scientific result differs only in the two authorized
% fields before adding recovery metadata.
comparison = res;
comparison.base_config_hash = original.base_config_hash;
for i = 1:numel(comparison.arms)
    comparison.arms(i).base_config_hash = original.arms(i).base_config_hash;
    comparison.arms(i).omega1_thresholded = original.arms(i).omega1_thresholded;
end
if ~isequaln(comparison,original)
    error('recover_a4_phase1:UnexpectedMutation', ...
        'A field outside threshold endpoint/provenance changed; artifacts were not written.');
end
if ~isequaln(trackedBefore,[res.arms.omega1_tracked]) || ...
        ~isequaln(topologyBefore,{res.arms.topology})
    error('recover_a4_phase1:OptimizationMutation', ...
        'Tracked endpoints or topology changed; artifacts were not written.');
end

recovery = struct( ...
    'phase','A4 Recovery Phase 1', ...
    'kind','post_hoc_endpoint_and_provenance_regeneration', ...
    'regenerated_utc',localUtcNow(), ...
    'original_optimization_created_utc',original.created_utc, ...
    'original_optimization_commit_sha',original.commit_sha, ...
    'optimization_rerun',false, ...
    'tracked_frequencies_reused_bitwise',true, ...
    'topologies_reused_bitwise',true, ...
    'optimization_histories_regenerated',false, ...
    'threshold_endpoint_recomputed',true, ...
    'provenance_hash_recomputed',true, ...
    'hash_algorithm','FNV-1a 32-bit, explicit modulo-2^32 wrapping', ...
    'original_base_config_hash',originalHash, ...
    'corrected_base_config_hash',correctedHash, ...
    'threshold_updates',updates);
res.recovery_phase1 = recovery;

save(matPath,'res','-v7.3');

slim = res;
for i = 1:numel(slim.arms), slim.arms(i).topology = []; end
localWriteJson(fullfile(outDir,'a4_result.json'),slim);
localWriteTable(fullfile(outDir,'a4_table.md'),res);

manifestPath = fullfile(outDir,'a4_manifest.json');
man = jsondecode(fileread(manifestPath));
man.original_base_config_hash = originalHash;
man.base_config_hash = correctedHash;
man.recovery_phase1 = recovery;
localWriteJson(manifestPath,man);

stageResultPath = fullfile(outDir,'a4_stage_result.json');
stageResult = jsondecode(fileread(stageResultPath));
stageOriginalHash = stageResult.config_hash;
stageResult.original_config_hash = stageOriginalHash;
stageResult.config_hash = fnv1a32_canonical_struct(stageResult.config);
stageRecovery = recovery;
stageRecovery.original_stage_config_hash = stageOriginalHash;
stageRecovery.corrected_stage_config_hash = stageResult.config_hash;
stageResult.recovery_phase1 = stageRecovery;
localWriteJson(stageResultPath,stageResult);

stageManifestPath = fullfile(outDir,'a4_stage_manifest.json');
stageManifest = jsondecode(fileread(stageManifestPath));
stageManifest.recovery_phase1 = stageRecovery;
localWriteJson(stageManifestPath,stageManifest);

summary = struct( ...
    'tracked_before',trackedBefore, ...
    'tracked_after',[res.arms.omega1_tracked], ...
    'threshold_before',thresholdBefore, ...
    'threshold_after',[res.arms.omega1_thresholded], ...
    'base_hash_before',originalHash, ...
    'base_hash_after',correctedHash, ...
    'stage_hash_before',stageOriginalHash, ...
    'stage_hash_after',stageResult.config_hash, ...
    'optimization_bitwise_unchanged',true, ...
    'topologies_bitwise_unchanged',true, ...
    'recovery',recovery);
end

function localWriteTable(path,res)
L = {};
L{end+1} = '# Table A4-1 — Eigenpair-refresh study';
L{end+1} = '';
L{end+1} = sprintf('Spec: `A4_SPECIFICATION_V3`. Base config hash: `%s`. Commit: `%s`.', ...
    res.base_config_hash,res.commit_sha);
L{end+1} = sprintf('Pre-declared equivalence margin delta = %.1f%%.',100*res.delta);
L{end+1} = '';
L{end+1} = ['**Recovery Phase 1:** optimization outputs and topologies are the original ' ...
    'artifacts; only `omega1_thresholded` and hash-derived provenance were regenerated post hoc.'];
L{end+1} = '';
L{end+1} = '`Δω₁ vs N=∞` is populated **only for clean arms** (Class B, or Class C/B1–B2 —';
L{end+1} = 'spec §7.6). It is left BLANK for B3/B4 and REJECTED arms — a contaminated or';
L{end+1} = 'unstable arm is disqualified as an accuracy reference and its endpoint must not';
L{end+1} = 'be read as one.';
L{end+1} = '';
L{end+1} = '| N | ω₁_tracked | ω₁_min | ω₁_thresh | MAC | j* | iters | conv | refreshes | eigensolves | grayness | comps | class | Δω₁ vs N=∞ |';
L{end+1} = '|---|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---|---:|';
frozen = res.arms([res.arms.N] == Inf);
ref = NaN;
if ~isempty(frozen) && strcmp(frozen(1).class,'ACCEPTED')
    ref = frozen(1).omega1_tracked;
end
for i = 1:numel(res.arms)
    a = res.arms(i);
    clean = strcmp(a.class,'ACCEPTED') || ...
        (strcmp(a.class,'ACCEPTED_WITH_BREAKDOWN') && any(strcmp(a.breakdown,{'B1','B2'})));
    if clean && isfinite(ref) && ref > 0
        dstr = sprintf('%+.2f%%',100*(a.omega1_tracked-ref)/ref);
    else
        dstr = '';
    end
    conv = 'no';
    if isfinite(a.final_design_change) && isfinite(a.tol) && ...
            a.final_design_change <= a.tol && a.iterations < a.cap
        conv = 'yes';
    end
    cls = a.class;
    if ~isempty(a.breakdown), cls = sprintf('%s/%s',a.class,a.breakdown); end
    L{end+1} = sprintf('| %s | %.4f | %.4f | %.4f | %.4f | %d | %d | %s | %d | %d | %.4f | %d | %s | %s |', ...
        a.tag,a.omega1_tracked,a.omega1_min,a.omega1_thresholded,a.mac_to_phi0, ...
        a.mode_index_jstar,a.iterations,conv,a.n_refresh,a.eigensolves_analytic, ...
        a.grayness,a.n_components,cls,dstr); %#ok<AGROW>
end
L{end+1} = '';
L{end+1} = sprintf('**Decision: %s**',res.decision.outcome);
L{end+1} = '';
L{end+1} = res.decision.statement;
L{end+1} = '';
L{end+1} = '_Wall-clock time is recorded for original-run provenance only and may not appear in any';
L{end+1} = 'performance claim (spec §4.5)._';
fid = fopen(path,'w');
if fid < 0, error('recover_a4_phase1:WriteFailed','Cannot write %s',path); end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',strjoin(L,newline));
end

function localWriteJson(path,data)
txt = jsonencode(data,PrettyPrint=true);
fid = fopen(path,'w');
if fid < 0, error('recover_a4_phase1:WriteFailed','Cannot write %s',path); end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',txt);
end

function t = localUtcNow()
t = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
end
