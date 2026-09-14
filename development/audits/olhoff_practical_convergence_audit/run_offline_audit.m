function run_offline_audit()
%RUN_OFFLINE_AUDIT Offline Olhoff practical-convergence evidence extraction.
%
% This script only loads existing MAT/JSON artifacts.  It never calls the
% optimizer and never writes under Matlab/reproduction2007.  Outputs are
% derived evidence for OLHOFF_PRACTICAL_CONVERGENCE_AUDIT.md.

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
outDir = fullfile(repo, 'analysis', 'olhoff_practical_convergence_audit', 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

sourceDir = '/Users/piotrek/Programming/Matlab/Olhoff/results';
spec = trajectorySpecifications(sourceDir, repo);

inventory = struct([]);
trajectories = struct([]);
for i = 1:numel(spec)
    if ~exist(spec(i).path, 'file')
        inventory(end+1) = missingInventory(spec(i)); %#ok<AGROW>
        continue
    end
    S = load(spec(i).path);
    res = findResultStruct(S);
    tr = normalizeTrajectory(spec(i), res);
    if isempty(trajectories)
        trajectories = tr;
    else
        trajectories(end+1) = orderfields(tr, trajectories(1)); %#ok<AGROW>
    end
    invRow = inventoryRow(tr);
    if isempty(inventory)
        inventory = invRow;
    else
        inventory(end+1) = orderfields(invRow, inventory(1)); %#ok<AGROW>
    end
end
writetable(struct2table(inventory), fullfile(outDir, 'artifact_inventory.csv'));

longMask = [trajectories.valid] & [trajectories.n_iter] >= 400;
longTraj = trajectories(longMask);

writePerIteration(longTraj, fullfile(outDir, 'trajectory_metrics.csv'));
writeTerminalLoss(longTraj, fullfile(outDir, 'terminal_loss_checkpoints.csv'), ...
    fullfile(outDir, 'retained_fraction_thresholds.csv'));
writeStoppingReplay(longTraj, fullfile(outDir, 'stopping_rule_replay.csv'));
writeMultiplicitySummary(longTraj, fullfile(outDir, 'multiplicity_summary.csv'));
writeMoveSummary(trajectories, fullfile(outDir, 'move_limit_summary.csv'));
writeTopologyEvidence(trajectories, fullfile(outDir, 'topology_checkpoint_evidence.csv'));
writeCrossResolution(repo, fullfile(outDir, 'cross_resolution_checkpoints.csv'));
writeSummaryJson(longTraj, inventory, fullfile(outDir, 'offline_audit_summary.json'));
makePrimaryPlot(longTraj, fullfile(outDir, 'authoritative_240x30_diagnostics.png'));

fprintf('Offline Olhoff audit outputs written to %s\n', outDir);
end

function spec = trajectorySpecifications(sourceDir, repo)
% Explicit list: every row has a scientific role rather than being selected
% by outcome.  Duplicates of migrated baselines are intentionally omitted.
names = { ...
    'auth240_rmin13_move005', 'lp240_rmin1.3.mat', 'Du-Olhoff reproduction';
    '240_rmin11_move005', 'lp240_rmin1.1.mat', 'rmin sensitivity';
    '240_rmin15_move005', 'lp240_rmin1.5.mat', 'rmin sensitivity';
    '240_rmin18_move005', 'lp240_rmin1.8.mat', 'rmin sensitivity';
    '240_rmin22_move005', 'lp240_rmin2.2.mat', 'rmin sensitivity';
    'fig4_240_rmin13_move020_400', 'FIG4_definitive.mat', 'move-limit long';
    'fig4_240_rmin13_move010_100', 'fig4_mv0.01.mat', 'move-limit endpoint';
    'fig4_240_rmin13_move020_100', 'fig4_mv0.02.mat', 'move-limit checkpoint';
    'fig4_240_rmin13_move030_100', 'fig4_mv0.03.mat', 'move-limit endpoint';
    'fig4_240_rmin13_move050_100', 'fig4_mv0.05.mat', 'move-limit endpoint';
    '160_rmin11_move005', 'lprmin1.1.mat', 'cross-resolution/rmin';
    '160_rmin12_move005', 'lprmin1.2.mat', 'cross-resolution/rmin';
    '160_rmin15_move005', 'lprmin1.5.mat', 'cross-resolution/rmin';
    '160_rmin20_move005', 'lprmin2.mat', 'current-profile analogue';
    '160_rmin25_move005', 'lprmin2.5.mat', 'rmin sensitivity';
    '160_rmin30_move005', 'lprmin3.mat', 'move-limit family';
    '160_rmin40_move005', 'lprmin4.mat', 'rmin sensitivity';
    '160_rmin30_move001', 'lp_mv0.001_tm0.05.mat', 'move-limit family';
    '160_rmin30_move002', 'lp_mv0.002_tm0.05.mat', 'move-limit family';
    '160_rmin30_move010', 'lp_mv0.01_tm0.05.mat', 'move-limit family';
    '160_rmin30_move020', 'lp_mv0.02_tm0.05.mat', 'move-limit family'};

spec = repmat(struct('id','','path','','role','','provenance',''), size(names,1), 1);
for i = 1:size(names,1)
    spec(i).id = names{i,1};
    spec(i).path = fullfile(sourceDir, names{i,2});
    spec(i).role = names{i,3};
    spec(i).provenance = 'documented original clean-room exploratory artifact directory';
end

% The authoritative migrated path is recorded separately in the inventory
% JSON/report; byte identity was checked against SOURCE_SHA256.txt in WP0.
assert(exist(fullfile(repo, 'Matlab', 'reproduction2007', 'baseline', ...
    'lp240_rmin1.3.mat'), 'file') == 2);
end

function res = findResultStruct(S)
fn = fieldnames(S);
for i = 1:numel(fn)
    candidate = S.(fn{i});
    if isstruct(candidate) && isscalar(candidate) && ...
            isfield(candidate,'cfg') && isfield(candidate,'hist') && ...
            isfield(candidate,'rho') && isfield(candidate,'omega')
        res = candidate;
        return
    end
end
error('run_offline_audit:NoResult', 'No clean-room result struct found.');
end

function tr = normalizeTrajectory(spec, res)
h = res.hist;
c = res.cfg;
n = numel(h.N);
omega = h.omega(:,1:n);
gap = (omega(2,:) - omega(1,:)) ./ omega(1,:);
if isfield(h,'innerConv')
    innerOK = logical(h.innerConv(1:n));
else
    innerOK = true(1,n);
end
if isfield(h,'vol')
    vol = h.vol(1:n);
else
    vol = NaN(1,n);
end
if isfield(h,'dxOuter')
    dInf = h.dxOuter(1:n);
else
    dInf = NaN(1,n);
end
lateStart = max(1, floor(0.9*n));
lateCandidates = [omega(1,lateStart:end), res.omega(1)];
reference = max(lateCandidates);

tr = struct();
tr.id = spec.id;
tr.path = spec.path;
tr.role = spec.role;
tr.provenance = spec.provenance;
tr.cfg = c;
tr.nelx = c.nelx;
tr.nely = c.nely;
tr.n_elem = c.nelx*c.nely;
tr.rmin = c.rminEl;
tr.move = c.move;
tr.tol_mult = c.tolMult;
tr.tol_outer = c.tolOuter;
tr.max_outer = c.maxOuter;
tr.inner_solver = char(string(c.innerSolver));
tr.n_iter = n;
tr.omega = omega;
tr.N = h.N(1:n);
tr.gap = gap;
tr.vol = vol;
tr.d_inf = dInf;
tr.inner_ok = innerOK;
tr.n_failures = sum(~innerOK);
tr.valid = all(innerOK) && strcmpi(tr.inner_solver,'lp');
tr.rho_final = res.rho(:);
tr.omega_final = res.omega(:);
tr.reference_omega1 = reference;
tr.reference_definition = 'max omega1 over final 10% plus post-update final state';
end

function row = inventoryRow(tr)
row = struct( ...
    'trajectory_id', tr.id, 'path', tr.path, 'role', tr.role, ...
    'provenance', tr.provenance, 'exists', true, ...
    'nelx', tr.nelx, 'nely', tr.nely, 'n_elements', tr.n_elem, ...
    'rmin_elements', tr.rmin, 'move', tr.move, ...
    'tol_mult', tr.tol_mult, 'tol_outer', tr.tol_outer, ...
    'max_outer', tr.max_outer, 'n_iterations', tr.n_iter, ...
    'inner_solver', tr.inner_solver, 'n_subproblem_failures', tr.n_failures, ...
    'valid_for_offline_replay', tr.valid, ...
    'full_scalar_history', true, 'full_density_history', false, ...
    'final_density_only', true, 'omega1_final', tr.omega_final(1), ...
    'omega2_final', tr.omega_final(2), 'omega3_final', tr.omega_final(3), ...
    'final_gap_rel', (tr.omega_final(2)-tr.omega_final(1))/tr.omega_final(1), ...
    'tail_bimodal_fraction_50', mean(tr.N(max(1,end-49):end)>=2), ...
    'reference_omega1', tr.reference_omega1);
end

function row = missingInventory(spec)
row = struct('trajectory_id',spec.id,'path',spec.path,'role',spec.role, ...
    'provenance',spec.provenance,'exists',false,'nelx',NaN,'nely',NaN, ...
    'n_elements',NaN,'rmin_elements',NaN,'move',NaN,'tol_mult',NaN, ...
    'tol_outer',NaN,'max_outer',NaN,'n_iterations',NaN,'inner_solver','', ...
    'n_subproblem_failures',NaN,'valid_for_offline_replay',false, ...
    'full_scalar_history',false,'full_density_history',false, ...
    'final_density_only',false,'omega1_final',NaN,'omega2_final',NaN, ...
    'omega3_final',NaN,'final_gap_rel',NaN,'tail_bimodal_fraction_50',NaN, ...
    'reference_omega1',NaN);
end

function writePerIteration(trs, outPath)
rows = cell(0,1);
for i = 1:numel(trs)
    tr = trs(i);
    w1 = tr.omega(1,:);
    w2 = tr.omega(2,:);
    w3 = tr.omega(3,:);
    for k = 1:tr.n_iter
        r1 = NaN;
        if k > 1, r1 = abs(w1(k)-w1(k-1))/max(abs(w1(k)),eps); end
        r5 = lagChange(w1,k,5); r10 = lagChange(w1,k,10); r20 = lagChange(w1,k,20);
        b5 = bandChange(w1,k,5); b10 = bandChange(w1,k,10); b20 = bandChange(w1,k,20);
        rows{end+1} = struct( ... %#ok<AGROW>
            'trajectory_id',tr.id,'iteration',k,'omega1',w1(k), ...
            'omega2',w2(k),'omega3',w3(k),'gap_rel',tr.gap(k), ...
            'N',tr.N(k),'volume',tr.vol(k), ...
            'volume_residual_rel',abs(tr.vol(k)-tr.cfg.volfrac)/tr.cfg.volfrac, ...
            'max_density_update',tr.d_inf(k), ...
            'move_saturated',abs(tr.d_inf(k)-tr.move)<1e-12, ...
            'lp_ok',tr.inner_ok(k),'rel_change_1',r1, ...
            'rel_change_5',r5,'rel_change_10',r10,'rel_change_20',r20, ...
            'rel_band_5',b5,'rel_band_10',b10,'rel_band_20',b20, ...
            'terminal_loss_rel',(tr.reference_omega1-w1(k))/tr.reference_omega1);
    end
end
writetable(struct2table([rows{:}]), outPath);
end

function r = lagChange(w,k,q)
if k <= q
    r = NaN;
else
    r = abs(w(k)-w(k-q))/max(abs(w(k)),eps);
end
end

function r = bandChange(w,k,q)
if k < q
    r = NaN;
else
    z = w(k-q+1:k);
    r = (max(z)-min(z))/max(abs(w(k)),eps);
end
end

function writeTerminalLoss(trs, checkpointPath, thresholdPath)
checkpoints = [50 100 150 200 250 300 400 800];
fractions = [0.95 0.975 0.99 0.995 0.999];
cpRows = cell(0,1); thRows = cell(0,1);
for i = 1:numel(trs)
    tr = trs(i); w1 = tr.omega(1,:);
    for k = checkpoints
        if k <= tr.n_iter
            cpRows{end+1} = struct( ... %#ok<AGROW>
                'trajectory_id',tr.id,'iteration',k,'omega1',w1(k), ...
                'omega2',tr.omega(2,k),'omega3',tr.omega(3,k), ...
                'N',tr.N(k),'gap_rel',tr.gap(k),'volume',tr.vol(k), ...
                'd_inf',tr.d_inf(k),'reference_omega1',tr.reference_omega1, ...
                'subsequent_gain_abs',tr.reference_omega1-w1(k), ...
                'subsequent_gain_rel',(tr.reference_omega1-w1(k))/tr.reference_omega1, ...
                'retained_fraction',w1(k)/tr.reference_omega1);
        end
    end
    for f = fractions
        threshold = f*tr.reference_omega1;
        suffixMin = flip(cummin(flip(w1)));
        k = find(suffixMin >= threshold,1,'first');
        if isempty(k), k = NaN; end
        thRows{end+1} = struct( ... %#ok<AGROW>
            'trajectory_id',tr.id,'target_fraction',f, ...
            'earliest_then_retained_iteration',k, ...
            'reference_omega1',tr.reference_omega1, ...
            'reference_definition',tr.reference_definition);
    end
end
writetable(struct2table([cpRows{:}]), checkpointPath);
writetable(struct2table([thRows{:}]), thresholdPath);
end

function writeStoppingReplay(trs, outPath)
metricKinds = {'endpoint','band'};
windows = [5 10 20];
tolerances = [1e-2 5e-3 2e-3 1e-3 5e-4 2e-4 1e-4 5e-5 2e-5 1e-5];
persistence = [5 10 20];
modalNames = {'none','N_stable','bimodal_gap_5pct','bimodal_gap_2pct', ...
    'bimodal_gap_1pct','bimodal_gap_0p5pct'};
modalGaps = [Inf Inf 0.05 0.02 0.01 0.005];
rows = cell(0,1);
for i = 1:numel(trs)
    tr = trs(i); w1 = tr.omega(1,:); n = tr.n_iter;
    volOK = abs(tr.vol-tr.cfg.volfrac)/tr.cfg.volfrac <= 1e-8;
    for mk = 1:numel(metricKinds)
        for q = windows
            metric = NaN(1,n);
            nStable = false(1,n);
            for k = q:n
                if strcmp(metricKinds{mk},'endpoint')
                    metric(k) = lagChange(w1,k,q);
                else
                    metric(k) = bandChange(w1,k,q);
                end
                nStable(k) = all(tr.N(k-q+1:k)==tr.N(k));
            end
            for tol = tolerances
                objOK = metric <= tol;
                for m = 1:numel(modalNames)
                    switch modalNames{m}
                        case 'none'
                            modalOK = true(1,n);
                        case 'N_stable'
                            modalOK = nStable;
                        otherwise
                            modalOK = nStable & tr.N>=2 & tr.gap<=modalGaps(m);
                    end
                    base = objOK & modalOK & volOK & tr.inner_ok;
                    for p = persistence
                        fire = firstPersistent(base,p);
                        if isnan(fire)
                            fw1=NaN; fw2=NaN; fw3=NaN; fN=NaN; fgap=NaN;
                            fvol=NaN; flossAbs=NaN; flossRel=NaN;
                            fdinf=NaN; postMin=NaN;
                        else
                            fw1=w1(fire); fw2=tr.omega(2,fire); fw3=tr.omega(3,fire);
                            fN=tr.N(fire); fgap=tr.gap(fire); fvol=tr.vol(fire);
                            flossAbs=tr.reference_omega1-fw1;
                            flossRel=flossAbs/tr.reference_omega1;
                            fdinf=tr.d_inf(fire);
                            postMin=min(w1(fire:end))/tr.reference_omega1;
                        end
                        rows{end+1} = struct( ... %#ok<AGROW>
                            'trajectory_id',tr.id,'metric_kind',metricKinds{mk}, ...
                            'window',q,'objective_tolerance',tol, ...
                            'modal_requirement',modalNames{m}, ...
                            'gap_tolerance',modalGaps(m),'volume_tolerance_rel',1e-8, ...
                            'persistence',p,'fire_iteration',fire, ...
                            'omega1_at_fire',fw1,'omega2_at_fire',fw2, ...
                            'omega3_at_fire',fw3,'N_at_fire',fN, ...
                            'gap_rel_at_fire',fgap,'volume_at_fire',fvol, ...
                            'd_inf_at_fire',fdinf,'reference_omega1',tr.reference_omega1, ...
                            'terminal_loss_abs',flossAbs,'terminal_loss_rel',flossRel, ...
                            'minimum_retained_fraction_after_fire',postMin, ...
                            'density_distance_available',false);
                    end
                end
            end
        end
    end
end
writetable(struct2table([rows{:}]), outPath);
end

function k = firstPersistent(mask,p)
k = NaN;
run = 0;
for i = 1:numel(mask)
    if mask(i)
        run = run+1;
        if run >= p
            k = i;
            return
        end
    else
        run = 0;
    end
end
end

function writeMultiplicitySummary(trs,outPath)
rows = cell(0,1);
for i=1:numel(trs)
    tr=trs(i); n=tr.n_iter;
    rows{end+1}=struct( ... %#ok<AGROW>
        'trajectory_id',tr.id,'first_bimodal',firstTrue(tr.N>=2), ...
        'persistent_bimodal_5',firstSuffixRun(tr.N>=2,5), ...
        'persistent_bimodal_10',firstSuffixRun(tr.N>=2,10), ...
        'persistent_bimodal_20',firstSuffixRun(tr.N>=2,20), ...
        'last_nonbimodal_iteration',lastTrue(tr.N<2), ...
        'bimodal_fraction_all',mean(tr.N>=2), ...
        'bimodal_fraction_last50',mean(tr.N(max(1,n-49):n)>=2), ...
        'first_persistent_gap_below_5pct_20',firstSuffixRun(tr.gap<=0.05,20), ...
        'first_persistent_gap_below_2pct_20',firstSuffixRun(tr.gap<=0.02,20), ...
        'first_persistent_gap_below_1pct_20',firstSuffixRun(tr.gap<=0.01,20), ...
        'final_gap_rel',tr.gap(end),'final_N',tr.N(end), ...
        'reference_omega1',tr.reference_omega1);
end
writetable(struct2table([rows{:}]),outPath);
end

function k = firstTrue(mask)
k=find(mask,1,'first'); if isempty(k), k=NaN; end
end

function k = lastTrue(mask)
k=find(mask,1,'last'); if isempty(k), k=NaN; end
end

function k = firstSuffixRun(mask,p)
% First online detection of p consecutive true values; later false values are
% allowed here and are exposed by last_nonbimodal and the replay tables.
k=firstPersistent(mask,p);
end

function writeMoveSummary(trs,outPath)
rows=cell(0,1);
for i=1:numel(trs)
    tr=trs(i);
    if ~contains(tr.role,'move-limit'), continue; end
    conn=connectivityMetrics(tr.rho_final,tr.nely,tr.nelx,0.5);
    [minOmega1,minOmega1Iter]=min(tr.omega(1,:));
    rows{end+1}=struct( ... %#ok<AGROW>
        'trajectory_id',tr.id,'nelx',tr.nelx,'nely',tr.nely, ...
        'rmin_elements',tr.rmin,'move',tr.move,'n_iterations',tr.n_iter, ...
        'omega1_final',tr.omega_final(1),'omega2_final',tr.omega_final(2), ...
        'omega3_final',tr.omega_final(3), ...
        'gap_rel_final',(tr.omega_final(2)-tr.omega_final(1))/tr.omega_final(1), ...
        'minimum_history_omega1',minOmega1, ...
        'minimum_history_omega1_iteration',minOmega1Iter, ...
        'iterations_omega1_below_50',nnz(tr.omega(1,:)<50), ...
        'tail_bimodal_fraction_50',mean(tr.N(max(1,end-49):end)>=2), ...
        'move_saturated_fraction',mean(abs(tr.d_inf-tr.move)<1e-12), ...
        'solid_components_at_0p5',conn.n_components, ...
        'left_right_connected_at_0p5',conn.left_right_connected, ...
        'largest_component_fraction',conn.largest_component_fraction, ...
        'gray_fraction_0p1_0p9',mean(tr.rho_final>0.1 & tr.rho_final<0.9));
end
writetable(struct2table([rows{:}]),outPath);
end

function writeTopologyEvidence(trs,outPath)
rows=cell(0,1);
short=find(strcmp({trs.id},'fig4_240_rmin13_move020_100'),1);
long=find(strcmp({trs.id},'fig4_240_rmin13_move020_400'),1);
if isempty(short)||isempty(long), return; end
a=trs(short); b=trs(long);
prefixIdentical = isequaln(a.omega,b.omega(:,1:a.n_iter)) && ...
    isequaln(a.N,b.N(1:a.n_iter)) && isequaln(a.d_inf,b.d_inf(1:a.n_iter));
d=a.rho_final-b.rho_final;
for threshold=[0.01 0.05 0.1]
    rows{end+1}=struct( ... %#ok<AGROW>
        'trajectory_id',b.id,'checkpoint_iteration',100,'reference_iteration',400, ...
        'prefix_history_bit_identical',prefixIdentical, ...
        'density_l1_mean',mean(abs(d)),'density_l2_rms',sqrt(mean(d.^2)), ...
        'fraction_density_difference_above_threshold',mean(abs(d)>threshold), ...
        'difference_threshold',threshold, ...
        'binary_disagreement_at_0p5',mean((a.rho_final>=0.5)~=(b.rho_final>=0.5)), ...
        'checkpoint_omega1_postupdate',a.omega_final(1), ...
        'reference_omega1_postupdate',b.omega_final(1), ...
        'checkpoint_loss_rel',(b.omega_final(1)-a.omega_final(1))/b.omega_final(1));
end
writetable(struct2table([rows{:}]),outPath);
end

function conn=connectivityMetrics(rho,nely,nelx,threshold)
B=reshape(rho,nely,nelx)>=threshold;
visited=false(size(B)); sizes=[]; labels=zeros(size(B)); component=0;
for r=1:nely
    for c=1:nelx
        if ~B(r,c)||visited(r,c), continue; end
        component=component+1; qr=zeros(nnz(B),1); qc=qr; head=1; tail=1;
        qr(1)=r; qc(1)=c; visited(r,c)=true; count=0;
        while head<=tail
            rr=qr(head); cc=qc(head); head=head+1; count=count+1;
            labels(rr,cc)=component;
            nbr=[rr-1 cc; rr+1 cc; rr cc-1; rr cc+1];
            for z=1:4
                r2=nbr(z,1); c2=nbr(z,2);
                if r2>=1&&r2<=nely&&c2>=1&&c2<=nelx&&B(r2,c2)&&~visited(r2,c2)
                    tail=tail+1; qr(tail)=r2; qc(tail)=c2; visited(r2,c2)=true;
                end
            end
        end
        sizes(component)=count; %#ok<AGROW>
    end
end
left=unique(labels(:,1)); left(left==0)=[]; right=unique(labels(:,end)); right(right==0)=[];
conn=struct('n_components',component, ...
    'left_right_connected',~isempty(intersect(left,right)), ...
    'largest_component_fraction',max([0 sizes])/max(1,nnz(B)));
end

function writeCrossResolution(repo,outPath)
meshes={'160x20','240x30','320x40','400x50'}; rows=cell(0,1);
for i=1:numel(meshes)
    p=fullfile(repo,'examples','Performance','equivalence','r3', ...
        ['run_summary_' meshes{i} '.json']);
    J=jsondecode(fileread(p)); m=J.meshes;
    cp=m.checkpoint_results;
    for k=1:numel(cp)
        rows{end+1}=struct( ... %#ok<AGROW>
            'mesh',meshes{i},'nelx',m.nelx,'nely',m.nely, ...
            'profile_id',m.profile_id,'iteration',cp(k).iter, ...
            'omega1',cp(k).omega1_A,'omega2',cp(k).omega2_A, ...
            'omega3',cp(k).omega3_A,'N',cp(k).N_A, ...
            'gap_rel',cp(k).gap_rel_A,'d_inf',cp(k).d_inf_A, ...
            'volume',cp(k).vol_A,'lp_flag',cp(k).lp_flag_A, ...
            'inner_converged',logical(cp(k).inner_converged_A), ...
            'run_status',m.stop_A.status,'run_stop_reason',m.stop_A.stop_reason, ...
            'run_n_outer',m.stop_A.n_outer);
    end
end
writetable(struct2table([rows{:}]),outPath);
end

function writeSummaryJson(trs,inventory,outPath)
summary=struct();
summary.generated=char(datetime('now','Format','yyyy-MM-dd HH:mm:ss Z'));
summary.analysis='offline only; no optimization executed';
summary.verdict='1600 CONFIRMED EXCESSIVE, BUT STOPPING RULE NOT YET ROBUST';
summary.n_inventory=numel(inventory);
summary.n_long_valid=numel(trs);
summary.authoritative_trajectory='auth240_rmin13_move005';
summary.reference_definition='maximum omega1 in final 10% plus post-update final state';
summary.per_iteration_density_history_available=false;
summary.unavailable_diagnostics={ ...
    'RMS density update','fraction of materially moving elements', ...
    'density/topology distance at arbitrary candidate stop iterations'};
summary.status_precedence={'SOLVER_FAILURE','CONVERGED','CAP_HIT'};
summary.outputs={ ...
    'artifact_inventory.csv','trajectory_metrics.csv', ...
    'terminal_loss_checkpoints.csv','retained_fraction_thresholds.csv', ...
    'stopping_rule_replay.csv','multiplicity_summary.csv', ...
    'move_limit_summary.csv','topology_checkpoint_evidence.csv', ...
    'cross_resolution_checkpoints.csv','authoritative_240x30_diagnostics.png'};
fid=fopen(outPath,'w');
if fid < 0, error('run_offline_audit:WriteFailed','Cannot open %s',outPath); end
fwrite(fid,jsonencode(summary,PrettyPrint=true),'char');
fclose(fid);
end

function makePrimaryPlot(trs,outPath)
i=find(strcmp({trs.id},'auth240_rmin13_move005'),1);
if isempty(i), return; end
tr=trs(i); k=1:tr.n_iter;
f=figure('Visible','off','Color','w','Position',[100 100 1100 850]);
tiledlayout(3,1,'TileSpacing','compact');
nexttile; plot(k,tr.omega(1:3,:),'LineWidth',1); ylabel('\omega [rad/s]');
legend('\omega_1','\omega_2','\omega_3','Location','best'); grid on;
nexttile; yyaxis left; semilogy(k,max(tr.gap,eps),'LineWidth',1); ylabel('relative gap');
yyaxis right; stairs(k,tr.N,'LineWidth',1); ylabel('N'); ylim([0.5 max(2.5,max(tr.N)+0.5)]); grid on;
nexttile; yyaxis left; plot(k,tr.d_inf,'LineWidth',1); ylabel('max |\Delta\rho|');
yyaxis right; semilogy(k,abs(tr.vol-tr.cfg.volfrac)/tr.cfg.volfrac+eps,'LineWidth',1); ylabel('relative volume residual');
xlabel('outer iteration'); grid on;
exportgraphics(f,outPath,'Resolution',160); close(f);
end
