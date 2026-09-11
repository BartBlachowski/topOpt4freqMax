function out = tr_inventory()
%TR_INVENTORY  A0 + A1: repository/host inventory and C320 oracle currentness.
%
%   Part A0 records the host, source and repository state.  Part A1 re-confirms
%   the frozen C320 oracle anchors -- the RHO and omega prefix hashes through
%   S3 = 352 -- WITHOUT repeating the full oracle study, which the stopped
%   attempt already performed and recorded in
%   three_rung_promotion_validation/C320_ORACLE.md.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));          % analysis/OlhoffCurrent
repo  = fileparts(fileparts(root));
addpath(root); addpath(here);
guard = olhoffcurrent_paths(); %#ok<NASGU>

out = struct();
out.generated = char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssXXX','TimeZone','local'));
out.matlab    = version;
out.threadsDefault = maxNumCompThreads;
out.computer  = computer;

% ---- A0: source integrity ----------------------------------------------
man = olhoffcurrent_source_manifest('Verify', true);
out.implTree = man.treeHash;
out.implNFiles = man.nFiles;
out.implOk = man.ok;

st = olhoffcurrent_currentness('Verbose', false);
out.currentness = st.state;
out.currentnessDetail = st.detail;

% ---- A0: hashes of the documents and sources this retry depends on -----
dep = { ...
 'prior_C320_ORACLE',      fullfile(study,'..','three_rung_promotion_validation','C320_ORACLE.md'); ...
 'prior_CONTROLLER_RECOVERY', fullfile(study,'..','three_rung_promotion_validation','CONTROLLER_RECOVERY.md'); ...
 'prior_PROVENANCE',       fullfile(study,'..','three_rung_promotion_validation','PROVENANCE.md'); ...
 'prior_REPORT',           fullfile(study,'..','three_rung_promotion_validation','REPORT.md'); ...
 'exhaustion_m',           fullfile(root,'+impl','architecture','+olh','+move','exhaustion.m'); ...
 'limit_m',                fullfile(root,'+impl','architecture','+olh','+move','limit.m'); ...
 'olhoffSolve_m',          fullfile(root,'+impl','architecture','olhoffSolve.m'); ...
 'tb_branches_m',          fullfile(root,'diagnostics','two_branch_maturity_240','scripts','tb_branches.m'); ...
 'tb_PREREGISTRATION',     fullfile(root,'diagnostics','two_branch_maturity_240','PREREGISTRATION.md'); ...
 'cv_config_m',            fullfile(root,'diagnostics','two_branch_controller_validation','scripts','cv_config.m'); ...
 'cv_run_m',               fullfile(root,'diagnostics','two_branch_controller_validation','scripts','cv_run.m'); ...
 'cv_telemetry_m',         fullfile(root,'diagnostics','two_branch_controller_validation','scripts','cv_telemetry.m'); ...
 'cv_export_m',            fullfile(root,'diagnostics','two_branch_controller_validation','scripts','cv_export.m'); ...
 'C320_record_json',       fullfile(root,'diagnostics','two_branch_controller_validation','runs','C320x40_record.json'); ...
 'C320_iterations_csv',    fullfile(root,'diagnostics','two_branch_controller_validation','runs','C320x40_iterations.csv'); ...
 'arch_PREREGISTRATION',   fullfile(root,'diagnostics','three_rung_architecture','PREREGISTRATION.md'); ...
 'arch_REPORT',            fullfile(root,'diagnostics','three_rung_architecture','REPORT.md'); ...
 'arch_THREE_RUNG_ANALYSIS', fullfile(root,'diagnostics','three_rung_architecture','THREE_RUNG_ANALYSIS.md'); ...
 'r240_REPORT',            fullfile(root,'diagnostics','three_rung_resolution_240','REPORT.md'); ...
 'r240_THRESHOLD_SPLITTING', fullfile(root,'diagnostics','three_rung_resolution_240','THRESHOLD_SPLITTING_ANALYSIS.md'); ...
 'r240_C240_ANALYSIS',     fullfile(root,'diagnostics','three_rung_resolution_240','C240_ANALYSIS.md'); ...
 'r240_RUNG4_MATERIALITY', fullfile(root,'diagnostics','three_rung_resolution_240','RUNG4_MATERIALITY.md'); ...
};
out.deps = struct('name',{},'path',{},'sha256',{},'exists',{});
for k = 1:size(dep,1)
    p = dep{k,2};
    e = exist(p,'file') == 2;
    h = '';
    if e, h = olhoffcurrent_sha256_file(p); end
    out.deps(end+1) = struct('name',dep{k,1}, ...
        'path', strrep(local_canon(p),[repo filesep],''), 'sha256', h, 'exists', e); %#ok<AGROW>
end

% ---- A1: the C320 oracle trajectory ------------------------------------
traj = fullfile(root,'evidence','two_branch_controller_validation','C320x40_trajectory.mat');
out.oracleTrajectory = strrep(traj,[repo filesep],'');
out.oracleTrajectoryPresent = exist(traj,'file') == 2;

EXP = struct( ...
  'rho_final', '0348b288da711f2bdcd89263f7feaae33ccaedcb73a133a1a5e941421795bfb3', ...
  'rho_prefix352', 'b8c0f18d887c7f5a81ece452f0a4097a0cc1a67710a137d66e677fcfd78b5ba3', ...
  'omega_prefix352','fd63608399edaede2c4f928a25d5478eacc3711f6d626ae770ad107e7f4c488d');
out.expected = EXP;

if out.oracleTrajectoryPresent
    d = dir(traj); out.oracleTrajectoryBytes = d.bytes;
    S = load(traj, 'RHO', 'hist', 'exh', 'meta', 'cfg');
    out.oracleSize = size(S.RHO);
    out.oracleImplTree = S.meta.implTree;
    out.got = struct( ...
      'rho_final',      local_vecHash(S.RHO(:,end)), ...
      'rho_prefix352',  local_vecHash(S.RHO(:,1:352)), ...
      'omega_prefix352',local_vecHash(S.hist.omega(1:2,1:352)));
    out.match = struct( ...
      'rho_final',      strcmp(out.got.rho_final,      EXP.rho_final), ...
      'rho_prefix352',  strcmp(out.got.rho_prefix352,  EXP.rho_prefix352), ...
      'omega_prefix352',strcmp(out.got.omega_prefix352,EXP.omega_prefix352));

    % frozen event structure
    out.stageStarts = S.exh.stageStarts(:).';
    out.descents    = S.exh.descents;
    out.exW = S.exh.W; out.exP = S.exh.P; out.exWnp = S.exh.Wnp; out.exTol = S.exh.tol;
    out.eventBranch = S.exh.eventBranch;
    % S3 terminal scientific state at iteration 352
    h = S.hist;
    out.s3 = struct( ...
      'omega1', h.omega(1,352), 'omega2', h.omega(2,352), ...
      'gap12', (h.omega(2,352)-h.omega(1,352))/h.omega(1,352), ...
      'volume', mean(S.RHO(:,352)), ...
      'move', h.move(352), 'stage', h.stage(352), 'multN', h.N(352), ...
      'nInner', h.nInner(352), 'innerConv', double(h.innerConv(352)), ...
      'cumInner', sum(h.nInner(1:352)));
    out.oracleOk = out.match.rho_final && out.match.rho_prefix352 && out.match.omega_prefix352;
else
    out.oracleOk = false;
end

out.verdict = 'C320_ORACLE_DEPENDENCY_FAIL';
if out.oracleOk && out.implOk && strcmp(out.implTree, ...
        'edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb')
    out.verdict = 'C320_ORACLE_DEPENDENCY_PASS';
end

fid = fopen(fullfile(study,'evidence','inventory.json'),'w');
c = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));

fprintf('\n== A0/A1 inventory ==\n');
fprintf('matlab        : %s\n', out.matlab);
fprintf('threads       : %d\n', out.threadsDefault);
fprintf('implTree      : %s (ok=%d, n=%d)\n', out.implTree, out.implOk, out.implNFiles);
fprintf('currentness   : %s\n', out.currentness);
if out.oracleTrajectoryPresent
  fprintf('oracle RHO    : %dx%d  implTree=%s\n', out.oracleSize(1), out.oracleSize(2), out.oracleImplTree);
  fprintf('  rho_final       %d  %s\n', out.match.rho_final,       out.got.rho_final);
  fprintf('  rho_prefix352   %d  %s\n', out.match.rho_prefix352,   out.got.rho_prefix352);
  fprintf('  omega_prefix352 %d  %s\n', out.match.omega_prefix352, out.got.omega_prefix352);
  fprintf('  W=%d P=%d Wnp=%d tol=%.6g\n', out.exW, out.exP, out.exWnp, out.exTol);
  fprintf('  stageStarts = %s\n', mat2str(out.stageStarts));
  disp(out.descents);
  fprintf('  branches: %s\n', strjoin(out.eventBranch, ' '));
  fprintf('  S3: omega1=%.17g omega2=%.17g vol=%.17g move=%g stage=%d N=%d nInner=%d cumInner=%d\n', ...
     out.s3.omega1, out.s3.omega2, out.s3.volume, out.s3.move, out.s3.stage, out.s3.multN, out.s3.nInner, out.s3.cumInner);
else
  fprintf('oracle trajectory ABSENT: %s\n', out.oracleTrajectory);
end
fprintf('VERDICT: %s\n', out.verdict);
end

function h = local_vecHash(v)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end

function p = local_canon(p)
f = java.io.File(p); p = char(f.getCanonicalPath());
end
