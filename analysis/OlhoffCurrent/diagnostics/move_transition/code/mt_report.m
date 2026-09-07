function M = mt_report()
%MT_REPORT  Analyse both arms at both meshes, run every preregistered gate,
%   emit METRICS.json, CONFIG_DIFF.json, the per-iteration CSVs and the figures.
%
%   Nothing here is fitted.  Every threshold is read from mt_candidate() /
%   mt_config(), which were frozen in PREREGISTRATION.md before the first run.

repo = fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))))));
addpath(fullfile(repo,'analysis','OlhoffCurrent'));
addpath(fullfile(repo,'analysis','OlhoffCurrent','diagnostics','move_transition','code'));
olhoffcurrent_scrub_forbidden_paths(repo);
guard = olhoffcurrent_paths(); %#ok<NASGU>

base    = fullfile(repo,'analysis','OlhoffCurrent','diagnostics','move_transition');
diagRt  = fullfile(repo,'analysis','OlhoffCurrent','diagnostics');
runsDir = fullfile(base,'runs');  figDir = fullfile(base,'figures');

meshes = [160 20; 320 40];
cand = struct('name','settledLocalObjective','tauAbs',0.01,'tauRel',0.50, ...
              'D',10,'W',10,'tauObj',5e-3);

man = olhoffcurrent_source_manifest('Verify', true);
cur = olhoffcurrent_currentness('Verbose', false);
assert(man.ok && strcmp(cur.state,'CURRENT'), 'mt_report:Integrity', ...
    'source integrity/currentness failed AFTER the runs: ok=%d state=%s', man.ok, cur.state);

R = struct('arm',{},'label',{},'mesh',{},'per',{},'rhoAtStop',{},'stopIter',{}, ...
           'status',{},'MndAtStop',{},'midAtStop',{},'grayAtStop',{}, ...
           'omega1AtStop',{},'volAtStop',{},'stages',{},'trans',{},'nOuter',{}, ...
           'log',{},'omega1PostLoop',{});
M = struct();
M.schema = 'olhoff_move_transition_experiment/1';
M.generated = char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssXXX','TimeZone','local'));
M.implementation = 'analysis/OlhoffCurrent';
M.production_preset = 'duOlhoffFixedPenaltySensitivityFiltered';
M.matlab = version;
M.threads = maxNumCompThreads;
M.source_tree_sha256 = man.treeHash;
M.currentness = cur.state;
M.preregistration_sha256 = olhoffcurrent_sha256_file(fullfile(base,'PREREGISTRATION.md'));
M.candidate = struct('metric','maxUtilization','threshold',0.5,'persistence',10);
M.terminal_admission = cand;
M.safety_cap = 600;
M.runs = {}; M.baseline_gate = {}; M.transitions = {}; M.stages = {}; M.gates = {};

for m = 1:size(meshes,1)
    nelx = meshes(m,1); nely = meshes(m,2);
    key  = sprintf('%dx%d', nelx, nely);
    LP = load(fullfile(runsDir, sprintf('armP_%dx%d.mat', nelx, nely)));
    LU = load(fullfile(runsDir, sprintf('armU_%dx%d.mat', nelx, nely)));
    oP = LP.out; oU = LU.out;

    % ---- BASELINE REGRESSION GATE (blocking) ---------------------------
    G = mt_gate(oP, LP.RHO, nelx, nely, diagRt);
    M.baseline_gate{end+1} = local_jsonable(G);

    tolOuter = olh.config.getPath(olhoffcurrent_config(nelx,nely), 'stop.tolerance');
    [pStop, pInfo] = mt_prodAdmit(oP.per, tolOuter);
    if isempty(pStop); pStop = oP.nOuter; pStat = 'CAP_HIT';
    else;              pStat = 'CONVERGED'; end

    % ---- ARM U terminal admission, offline, preregistered rule ----------
    E = mt_predicate(oU.per, cand);
    if isempty(E.stopIter); uStop = oU.nOuter; uStat = 'CAP_HIT';
    else;                   uStop = E.stopIter; uStat = 'CONVERGED'; end

    for a = {{'P',oP,LP.RHO,pStop,pStat,'ARM P -- production move transition'}, ...
             {'U',oU,LU.RHO,uStop,uStat,'ARM U -- utilization-gated move transition'}}
        q = a{1}; arm = q{1}; o = q{2}; RH = q{3}; si = q{4}; stt = q{5};
        P = o.per;
        w1pl = local_postloopOmega1(RH(:,si), nelx, nely);
        SP_  = mt_spatial(RH, P, olh.config.getPath(olhoffcurrent_config(nelx,nely),'design.initial'), ...
                          olh.config.getPath(olhoffcurrent_config(nelx,nely),'design.minimum'));
        R(end+1) = struct('arm',arm,'label',q{6},'mesh',[nelx nely],'per',P, ...
            'rhoAtStop',RH(:,si),'stopIter',si,'status',stt, ...
            'MndAtStop',P.Mnd(si),'midAtStop',P.mid(si),'grayAtStop',P.gray(si), ...
            'omega1AtStop',P.omega1(si),'volAtStop',P.volume(si), ...
            'stages',mt_stages(P,si),'trans',mt_transitions(P),'nOuter',o.nOuter, ...
            'log',{o.log},'omega1PostLoop',w1pl); %#ok<AGROW>
        mt_export(o, fullfile(runsDir, sprintf('arm%s_%s_iterations.csv', arm, key)));
        M.runs{end+1} = struct('key',sprintf('arm%s_%s',arm,key),'arm',arm,'mesh',[nelx nely], ...
            'NE',o.NE,'nOuterRun',o.nOuter,'solverStatus',o.status, ...
            'admissionStatus',stt,'stopIter',si, ...
            'omega1_traj',P.omega1(si),'omega1_postloop_atCap',o.omega(1), ...
            'Mnd_pct',P.Mnd(si),'gray_fraction',P.gray(si),'mid_fraction',P.mid(si), ...
            'volume',P.volume(si),'gap12',P.gap12(si),'omega2',P.omega2(si), ...
            'l2',P.l2(si),'maxAbs',P.maxAbs(si),'r_rho',P.ratio(si), ...
            'utilCount',P.utilCount(si),'move',P.move(si),'stage',P.stage(si), ...
            'itersSinceMoveChange',P.itersSinceMoveChange(si), ...
            'innerTotal',sum(P.nInner(1:si)),'innerAtStop',P.nInner(si), ...
            'innerConvAll',all(P.innerConv(1:si)~=0), ...
            'omega1_postloop',w1pl, ...
            'spatial_at_stop', struct('quantileLevels',SP_.quantileLevels, ...
                'utilQuantiles',SP_.utilQuantiles(si,:), ...
                'fracAtBound',SP_.fracAtBound(si),'fracAbove50',SP_.fracAbove50(si), ...
                'nAtBound',SP_.nAtBound(si),'NE',SP_.NE), ...
            'spatial_median_fracAtBound', median(SP_.fracAtBound), ...
            'spatial_median_fracAbove50', median(SP_.fracAbove50), ...
            'descents',P.outer(P.descent).','wall_s',o.wall_s,'cfgHash',o.cfgHash);
    end

    % ---- transitions / stages -------------------------------------------
    % Built field-by-field, NOT with struct(name,value,...): passing a cell
    % array as a value to struct() triggers struct-array replication and would
    % silently flatten these tables.
    tr = struct('mesh',[nelx nely]);
    tr.armP = local_jsonable(R(end-1).trans);
    tr.armU = local_jsonable(R(end).trans);
    M.transitions{end+1} = tr;

    sg = struct('mesh',[nelx nely]);
    sg.armP = local_jsonable(R(end-1).stages);
    sg.armU = local_jsonable(R(end).stages);
    M.stages{end+1} = sg;

    % ---- historical fixed-move reference (evidence only) ----------------
    FM = load(fullfile(diagRt,'move_stop','runs',sprintf('fixedmove_%s.mat',key)), 'out');
    fp = FM.out.per;
    fp.ratio = fp.maxAbs ./ fp.move;
    fp.utilCount = zeros(size(fp.outer));
    for k = 2:numel(fp.outer)
        fp.utilCount(k) = mt_utilCount(fp.ratio(1:k-1), 1, k-1, 0.5);
    end
    R(end+1) = struct('arm','FIXED','label','historical fixed move 0.04 (evidence)', ...
        'mesh',[nelx nely],'per',fp,'rhoAtStop',FM.out.rhoFinal(:), ...
        'stopIter',FM.out.nOuter,'status',FM.out.status, ...
        'MndAtStop',FM.out.Mnd_final,'midAtStop',FM.out.mid_final, ...
        'grayAtStop',FM.out.gray_final,'omega1AtStop',fp.omega1(end), ...
        'volAtStop',FM.out.volume_final,'stages',[],'trans',[],'nOuter',FM.out.nOuter, ...
        'log',{{}},'omega1PostLoop',FM.out.omega(1)); %#ok<AGROW>

    % ---- GATES -----------------------------------------------------------
    M.gates{end+1} = mt_gates(R(end-2), R(end-1), R(end), pInfo, E, cand);
end

% ---- CONFIG_DIFF ------------------------------------------------------
M.config_diff = mt_configDiff(meshes);
local_writejson(fullfile(base,'CONFIG_DIFF.json'), M.config_diff);
local_writejson(fullfile(base,'METRICS.json'), M);
mt_figures(R, figDir);
save(fullfile(runsDir,'analysis.mat'), 'R', 'M', '-v7.3');
fprintf('[mt_report] done\n');
end

% =========================================================================
function w1 = local_postloopOmega1(rho, nelx, nely)
cfg = olhoffcurrent_config(nelx, nely);
mdl = model2D(olh.config.toLegacy(cfg));
[K,M_] = assemble2D(mdl, rho, olh.config.getPath(cfg,'material.stiffness.p'), ...
                    olh.config.getPath(cfg,'material.mass'));
Jc = olh.config.getPath(cfg,'eigen.targetMode') + olh.config.getPath(cfg,'eigen.maxCluster');
w  = eigSolve(K, M_, Jc, olh.config.getPath(cfg,'eigen.solver'));
w1 = w(1);
end

function s = local_jsonable(x)
if isstruct(x) && numel(x) > 1
    s = arrayfun(@local_jsonable, x, 'UniformOutput', false); return
end
s = x;
end

function local_writejson(p, s)
fid = fopen(p,'w'); fprintf(fid,'%s', jsonencode(s,'PrettyPrint',true)); fclose(fid);
fprintf('[mt_report] %s\n', p);
end
