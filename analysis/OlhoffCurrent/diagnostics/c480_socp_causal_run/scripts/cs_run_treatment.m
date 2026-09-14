function cs_run_treatment(mode)
%CS_RUN_TREATMENT  THE ONE C480 exact-SOCP treatment run.
%
%   cs_run_treatment()          launch (refuses if already launched)
%   cs_run_treatment('resume')  continue after a PROCESS interruption from the
%                               latest automatic checkpoint, unmodified code
%                               (AUDIT_PREREGISTRATION.md section 14)
%
%   Order: identity gates (prereg + amendment hashes, preflight PASS, treatment
%   code hashes, +impl tree, config hash) -> launch lock -> the solve ->
%   durable evidence -> post-run identity.
if nargin < 1, mode = 'launch'; end
S = cs_setup();
guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);
runDir = fullfile(S.study, 'run');
if ~isfolder(runDir), mkdir(runDir); end
tag  = 'C480x60_socp';
lock = fullfile(S.evDir, 'LAUNCHED.lock');
trajFile = fullfile(S.evDir, sprintf('%s_trajectory.mat', tag));
ckFile = fullfile(S.evDir, 'checkpoint.mat');

% ---- identity gates --------------------------------------------------------
pf = jsondecode(fileread(fullfile(S.study, 'evaluations', 'preflight.json')));
assert(strcmp(pf.verdict, 'C480_SOCP_SINGLE_FACTOR_PREFLIGHT_PASS'), 'cs_run:preflight', 'preflight not PASS');
assert(strcmp(cs_filehash(fullfile(S.study,'AUDIT_PREREGISTRATION.md')), pf.preregistration_sha256), 'cs_run:prereg');
assert(strcmp(cs_filehash(fullfile(S.study,'PREREGISTRATION_AMENDMENT_1.md')), pf.amendment1_sha256), 'cs_run:amend');
codeNames = fieldnames(pf.treatment_code_sha256_at_launch);
codeOk = true; codeBad = {}; nChecked = 0;
for i = 1:numel(codeNames)
    fn = regexprep(codeNames{i}, '_m$', '.m');           % jsondecode mangles '.' to '_'
    if ~isfile(fullfile(S.here, fn)), codeOk = false; codeBad{end+1} = [fn ' (missing)']; continue; end %#ok<AGROW>
    nChecked = nChecked + 1;
    if ~strcmp(cs_filehash(fullfile(S.here, fn)), pf.treatment_code_sha256_at_launch.(codeNames{i}))
        codeOk = false; codeBad{end+1} = fn; %#ok<AGROW>
    end
end
assert(codeOk && nChecked == numel(codeNames), 'cs_run:code', 'treatment code changed since preflight: %s', strjoin(codeBad, ','));
fprintf('[cs_run] %d treatment code files verified against the preflight freeze\n', nChecked);
sm = olhoffcurrent_source_manifest('Verify', true);
assert(sm.ok && strcmp(sm.treeHash, S.expect.implTree), 'cs_run:impl', '+impl tree mismatch');
L = load(S.controlTraj, 'cfg', 'hist');
cfg = L.cfg;
assert(strcmp(olhoffcurrent_config_hash(cfg), S.expect.cfgHash), 'cs_run:cfg');

treat = struct('innerSolver','exactSOCP','stopAfter',0,'expectOmega1',L.hist.omega(:,1), ...
    'checkpointEvery',25,'checkpointFile',ckFile,'resumeFrom','','crossEvery',1,'progress',true);

switch mode
    case 'dryrun'
        fprintf('[cs_run] DRYRUN: all identity gates passed; lock %s exists=%d; trajectory exists=%d\n', ...
            lock, isfile(lock), isfile(trajFile));
        return
    case 'launch'
        assert(~isfile(lock), 'cs_run:AlreadyLaunched', ...
            'the one treatment run was already launched (%s); use ''resume'' only after a process interruption', lock);
        assert(~isfile(trajFile), 'cs_run:TrajectoryExists', 'treatment trajectory already exists');
        launch = struct('mode','launch','time',char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssZ','TimeZone','local')), ...
            'matlab',version,'implTree',sm.treeHash,'cfgHash',olhoffcurrent_config_hash(cfg), ...
            'preregistration',pf.preregistration_sha256,'amendment1',pf.amendment1_sha256, ...
            'treat',rmfield(treat,'expectOmega1'),'host',getenv('HOSTNAME'),'pid',feature('getpid'));
        cs_json(fullfile(runDir, 'launch.json'), launch);
        fid = fopen(lock, 'w'); fprintf(fid, '%s\n', launch.time); fclose(fid);
    case 'resume'
        assert(isfile(lock), 'cs_run:NotLaunched', 'nothing to resume');
        assert(~isfile(trajFile), 'cs_run:TrajectoryExists', 'run already completed');
        assert(isfile(ckFile), 'cs_run:NoCheckpoint', 'no checkpoint to resume from');
        treat.resumeFrom = ckFile;
        CK = load(ckFile, 'outer');
        rs = struct('mode','resume','time',char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssZ','TimeZone','local')), ...
            'resumeAfterOuter',CK.outer,'pid',feature('getpid'));
        f = fullfile(runDir, sprintf('resume_after_%d.json', CK.outer));
        cs_json(f, rs);
    otherwise
        error('cs_run:mode', 'unknown mode');
end

fprintf('\n%s\n[cs_run] C480 EXACT-SOCP TREATMENT (%s)  cfg %s\n%s\n', repmat('=',1,72), mode, ...
    olhoffcurrent_config_hash(cfg), repmat('=',1,72));
t0 = tic;
res = cs_olhoffSolveSOCP(cfg, treat);
wall = toc(t0);
fprintf('[cs_run] status=%s nOuter=%d wall=%.1fs\n', res.status, numel(res.hist.N), wall);

sm2 = olhoffcurrent_source_manifest('Verify', true);
meta = struct('matlab', version, 'cfgHash', olhoffcurrent_config_hash(cfg), ...
    'implTree_pre', sm.treeHash, 'implTree_post', sm2.treeHash, 'implTree_post_ok', sm2.ok, ...
    'preregistration', pf.preregistration_sha256, 'amendment1', pf.amendment1_sha256, ...
    'mode', mode, 'wall_this_process_s', wall, ...
    'finished', char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssZ','TimeZone','local')));
out = cs_postprocess(res, cfg, tag, S.evDir, runDir, meta);
fprintf('[cs_run] evidence written: %s\n', out.files.trajectory);
end
