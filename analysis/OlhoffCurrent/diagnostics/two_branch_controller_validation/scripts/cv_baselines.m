function B = cv_baselines()
%CV_BASELINES  Phase 8.  Freeze the production baselines from DURABLE EVIDENCE.
%
%   No production run is made.  Every field is read from an existing frozen
%   study's tracked METRICS.json and per-iteration CSV, or from the one
%   surviving production trajectory (400x50).  A field that does not exist in
%   the surviving evidence is recorded as unavailable -- never reconstructed,
%   never guessed.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root);

D = fullfile(root,'diagnostics');
src = struct( ...
 'm160', struct('mesh',[160 20], ...
    'metrics', fullfile(D,'move_stop','METRICS.json'), 'key','baseline_160x20', ...
    'csv', fullfile(D,'move_stop','runs','baseline_160x20_iterations.csv'), ...
    'traj', '', 'matlab','25.2.0.3042426 (R2025b) Update 1'), ...
 'm320', struct('mesh',[320 40], ...
    'metrics', fullfile(D,'move_stop','METRICS.json'), 'key','baseline_320x40', ...
    'csv', fullfile(D,'move_stop','runs','baseline_320x40_iterations.csv'), ...
    'traj', '', 'matlab','25.2.0.3042426 (R2025b) Update 1'), ...
 'm400', struct('mesh',[400 50], ...
    'metrics', fullfile(D,'move_activity_400','METRICS.json'), 'key','P400', ...
    'csv', fullfile(D,'move_activity_400','runs','P400_400x50_iterations.csv'), ...
    'traj', fullfile(root,'evidence','move_activity_400','P400_400x50_trajectory.mat'), ...
    'matlab','25.2.0.2998904 (R2025b)'));

B = struct(); keys = fieldnames(src);
fprintf('\n%s\nCV_BASELINES  (Phase 8, durable evidence only, no reruns)\n%s\n', ...
    repmat('=',1,72), repmat('=',1,72));

for i = 1:numel(keys)
    s = src.(keys{i});
    b = struct('mesh', s.mesh, 'NE', prod(s.mesh), 'matlab', s.matlab);
    b.metricsFile = strrep(s.metrics,[repo filesep],'');
    b.csvFile     = strrep(s.csv,[repo filesep],'');
    b.metricsSha  = olhoffcurrent_sha256_file(s.metrics);
    b.csvSha      = olhoffcurrent_sha256_file(s.csv);

    % ---- scalars from the frozen METRICS.json ---------------------------
    M = jsondecode(fileread(s.metrics));
    if isfield(M,'runs')
        R = M.runs; if ~iscell(R), R = num2cell(R); end
        rec = []; for j=1:numel(R), if strcmp(R{j}.key, s.key), rec = R{j}; end, end
        b.status = rec.status; b.nOuter = rec.nOuter; b.innerTotal = rec.innerTotal;
        b.wall_s = rec.wall_s; b.omega1 = rec.omega1; b.omega2 = rec.omega2;
        b.gap12 = rec.gap12; b.volume = rec.volume; b.Mnd = rec.Mnd_pct;
        b.gray = rec.gray_fraction; b.mid = rec.mid_fraction;
        b.tolOuter = rec.tolOuter; b.cfgHash = rec.cfgHash;
    else
        A = M.arms; if ~iscell(A), A = num2cell(A); end
        rec = []; for j=1:numel(A), if strcmp(A{j}.arm, s.key), rec = A{j}; end, end
        b.status = rec.status; b.nOuter = rec.nOuter; b.innerTotal = rec.innerTotal;
        b.wall_s = rec.wall_s; b.omega1 = rec.omega1_final; b.omega2 = rec.omega2_final;
        b.gap12 = rec.gap12_final; b.volume = rec.volume_final; b.Mnd = rec.Mnd_final;
        b.gray = rec.gray_final; b.mid = rec.mid_final;
        b.tolOuter = 0.05*sqrt(prod(s.mesh)/3200);
        b.cfgHash = M.config_hash_400x50_production;
    end

    % ---- per-iteration facts from the tracked CSV -----------------------
    T = readtable(s.csv);
    mv = T.move;
    b.moveTransitions = struct('iter',{},'from',{},'to',{});
    for k = 2:numel(mv)
        if mv(k) ~= mv(k-1)
            b.moveTransitions(end+1) = struct('iter',k,'from',mv(k-1),'to',mv(k)); %#ok<AGROW>
        end
    end
    b.firstDescentIter = NaN;
    if ~isempty(b.moveTransitions), b.firstDescentIter = b.moveTransitions(1).iter; end
    b.move_final  = mv(end);
    b.stage_final = T.stage(end);
    b.laddersReached = numel(unique(mv));
    b.reachedMoveMin = any(mv == 0.005);

    % beta-stall replay, exactly olh.move.limit's boundVariable branch
    beta = T.beta; W = 10; sTol = 5e-3;
    fires = false(numel(beta),1);
    for k = 1:numel(beta)
        bb = beta(1:k-1);
        if numel(bb) >= 2*W
            w2 = mean(bb(end-W+1:end)); w1 = mean(bb(end-2*W+1:end-W));
            fires(k) = (w2-w1)/max(abs(w1),eps) < sTol;
        end
    end
    b.betaStallFirst = local_first(fires);
    b.convergenceIter = b.nOuter;

    % ---- the final density vector, only where it survives ---------------
    if ~isempty(s.traj) && isfile(s.traj)
        S = load(s.traj,'RHO');
        b.rho_available = true;
        b.rho_sha256 = local_vecHash(S.RHO(:,end));
        b.trajectory = strrep(s.traj,[repo filesep],'');
        b.trajectorySha = olhoffcurrent_sha256_file(s.traj);
    else
        b.rho_available = false;
        b.rho_sha256 = 'UNAVAILABLE -- raw .mat lost (see EVIDENCE_POLICY.md)';
        b.trajectory = 'UNAVAILABLE';
        b.trajectorySha = '';
    end

    B.(keys{i}) = b;
    fprintf('  %3dx%-3d %-10s outer=%-4d inner=%-5d wall=%-7.1f omega1=%.6f Mnd=%.4f\n', ...
        b.mesh(1), b.mesh(2), b.status, b.nOuter, b.innerTotal, b.wall_s, b.omega1, b.Mnd);
    tr = arrayfun(@(t) sprintf('%d(%g->%g)', t.iter, t.from, t.to), b.moveTransitions, 'uni', 0);
    fprintf('           transitions: %s | final move=%g stage=%d | reached 0.005: %d\n', ...
        strjoin(tr,', '), b.move_final, b.stage_final, b.reachedMoveMin);
    fprintf('           beta stall first fires at %s | final rho available: %d\n', ...
        mat2str(b.betaStallFirst), b.rho_available);
end

outF = fullfile(study,'evidence','baselines.json');
fid = fopen(outF,'w'); fwrite(fid, jsonencode(B,'PrettyPrint',true)); fclose(fid);
fprintf('  wrote %s\n', outF);
end

function k = local_first(v), k = find(v,1); if isempty(k), k = NaN; end, end
function h = local_vecHash(v)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
