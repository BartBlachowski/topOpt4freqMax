function out = run_spurious_mode_crosscheck(mode)
%RUN_SPURIOUS_MODE_CROSSCHECK  Offline modal re-evaluation of the 27 saved designs.
%
%   run_spurious_mode_crosscheck('probe')     800x100 Du-Olhoff design, model P, timing probe
%   run_spurious_mode_crosscheck('campaign')  all 27 designs: model P, E1 and each method's
%                                             native model; controls; results/ files
%   run_spurious_mode_crosscheck('figures')   figures/ from results/results.mat
%
%   NO optimizer is run.  The saved final density fields of the nine-mesh
%   benchmark are re-analysed as they are: no threshold, filter, clip or rescale
%   (the only operation is the evaluator's own no-op clamp to [0,1]).
%
%   Material models (E0 = 1e7, rho0 = 1, nu = 0.3, plane stress, t = 1):
%     P   Proposed native   E = E0(1e-9 + (1-1e-9)x^3),  rho = 1e-9 + (1-1e-9)x
%     Y   Yuksel native     E as P,  rho = 1e-9 + (1-1e-9)g,  g = x (x > 0.1), x^6 (x <= 0.1)
%     D   Du-Olhoff native  E = E0 g,  g = x^3 (x >= 0.1), 0.01x (x < 0.1),  rho = x
%     E1  common evaluator  E = E0(1e-6 + (1-1e-6)x^3),  rho = 1e-6 + (1-1e-6)x
%
%   FE model, supports, eigensolver options and the structural-mode classifier
%   are copied VERBATIM from examples/Performance/benchmark_profile/
%   study_evaluate_design.m (candidate_c_unanimous_v1); only the P, Y and D
%   material branches are added to PENCIL, and SOLVE_BATCH additionally returns
%   the eigenvectors and the eigs time.  Classifier thresholds are unchanged:
%   void = x <= 0.1; a mode is structural iff residual <= 1e-6, voidKE < 0.5,
%   voidSE < 0.5 and KE-weighted density > 0.5.

if nargin < 1, mode = 'campaign'; end
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
resDir = fullfile(here, 'results'); figDir = fullfile(here, 'figures');
if exist(resDir,'dir') ~= 7, mkdir(resDir); end
if exist(figDir,'dir') ~= 7, mkdir(figDir); end
maxNumCompThreads(1);

switch mode
    case 'probe',    out = probe(repo, resDir);
    case 'campaign', out = campaign(repo, resDir);
    case 'figures',  out = figures(resDir, figDir, repo);
    otherwise, error('Unknown mode %s.', mode);
end
end

% =========================================================================
function S = sources(repo)
root = fullfile(repo, 'examples', 'Performance', 'conference_benchmark');
S = struct( ...
    'label', {'campaign_9mesh_r2', 'nine_mesh_pedersen_b21483b'}, ...
    'file',  {fullfile(root,'campaign_9mesh_r2','benchmark_records.mat'), ...
              fullfile(root,'nine_mesh_pedersen_b21483b','benchmark_records.mat')}, ...
    'methods', {{'proposed','yuksel'}, {'olhoff'}}, ...
    'expected_sha256', {'873125858df02664d9a7d37bd09e4e9b0b1ccff5c37940d088f9b1e424921fae', ...
                        '261bf8fc94a1efb7da223aae64ac72da336a4d8f48341f0d8bf79b0a4d2abe82'});
end

function [cases, prov] = load_cases(repo)
% The 27 designs of the published (composed) nine-mesh table, from the two
% source campaigns named in nine_mesh_comparison_pedersen_b21483b's manifest.
% The Olhoff rows of campaign_9mesh_r2 are the historical SIMP + eq. (4b)
% realization and are deliberately NOT used.
S = sources(repo); cases = struct([]); prov = struct('sources', struct([]));
meshes = [160 20;240 30;320 40;400 50;480 60;560 70;640 80;720 90;800 100];
for s = 1:numel(S)
    h = sha256(S(s).file);
    assert(strcmp(h, S(s).expected_sha256), 'Source %s hash %s differs from the composition manifest.', S(s).label, h);
    L = load(S(s).file, 'records', 'manifest');
    src = struct('label', S(s).label, 'file', relpath(S(s).file, repo), 'sha256', h, ...
        'hash_matches_composition_manifest', true, ...
        'campaign_generated', L.manifest.generated_datetime, ...
        'repository', L.manifest.repository, 'environment', L.manifest.environment, ...
        'methods_taken', {S(s).methods});
    if isfield(L.manifest, 'method_configurations'), src.method_configurations = L.manifest.method_configurations; end
    prov.sources = [prov.sources, src];
    for key = S(s).methods
        for m = 1:size(meshes,1)
            R = L.records(strcmp({L.records.method_key}, key{1}) & ...
                arrayfun(@(r) isequal(r.mesh(:).', meshes(m,:)), L.records) & ~[L.records.is_warmup]);
            assert(numel(R) == 1, 'Expected exactly one %s %dx%d record in %s.', key{1}, meshes(m,1), meshes(m,2), S(s).label);
            if strcmp(key{1}, 'olhoff')
                assert(strcmp(R.production_preset, 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered'), ...
                    'Olhoff record is not the Pedersen/adaptive-box preset.');
            end
            c = struct('method', key{1}, 'nelx', meshes(m,1), 'nely', meshes(m,2), ...
                'source', S(s).label, 'x', double(R.x(:)), 'x_sha256', vec_sha(double(R.x(:))), ...
                'omega_native_recorded', R.omega(:), 'status', R.status, 'profile_id', R.profile_id, ...
                'E1_recorded', R.evaluator.selected_omega_raw_E1, ...
                'E1_ordinal_recorded', R.evaluator.selected_ordinal_raw_E1, ...
                'E1_modes_recorded', R.evaluator.omega_raw_E1(:));
            cases = [cases, c]; %#ok<AGROW>
        end
    end
end
% Row order of the published table: mesh, then Proposed, Yuksel, Du-Olhoff.
ord = zeros(1,numel(cases)); keyOrder = {'proposed','yuksel','olhoff'};
for i = 1:numel(cases)
    ord(i) = cases(i).nelx*10 + find(strcmp(keyOrder, cases(i).method));
end
[~, ix] = sort(ord); cases = cases(ix);
end

% =========================================================================
function out = probe(repo, resDir)
cases = load_cases(repo);
c = cases(strcmp({cases.method},'olhoff') & [cases.nelx] == 800);
fprintf('Probe: Du-Olhoff 800x100 saved design, model P, 12 eigenpairs.\n');
tA = tic; [Kf,Mf,md,Ee,rr,zeff] = pencil(c.x, c.nelx, c.nely, 'P'); tAsm = toc(tA);
tB = tic; [b, ok, msg, ~, tEig] = solve_batch(Kf,Mf,md,Ee,rr,zeff,12); tBatch = toc(tB);
assert(ok, msg);
out = struct('case','olhoff 800x100','model','P','k',12,'free_dofs',size(Kf,1), ...
    'assembly_s',tAsm,'eigs_s',tEig,'batch_total_s',tBatch, ...
    'omega',b.omega','valid_structural',b.valid_structural','voidKE',b.voidKE', ...
    'residual',b.eigenpair_residual');
fprintf('free DOFs %d | assembly %.2f s | eigs %.2f s | batch %.2f s\n', size(Kf,1), tAsm, tEig, tBatch);
disp(table((1:12)', b.omega, b.valid_structural, b.voidKE, b.voidSE, b.densityParticipation, b.eigenpair_residual, ...
    'VariableNames', {'j','omega','structural','voidKE','voidSE','dwp','residual'}));
writejson(fullfile(resDir, 'probe_800x100_olhoff_P.json'), out);
end

% =========================================================================
function out = campaign(repo, resDir)
t0 = tic;
[cases, prov] = load_cases(repo);
nativeModel = struct('proposed','P','yuksel','Y','olhoff','D');
rows = struct([]); modes = struct([]); controls = struct([]); timing = struct([]);
macStore = struct([]);
for i = 1:numel(cases)
    c = cases(i); tag = sprintf('%s %dx%d', c.method, c.nelx, c.nely);
    fprintf('[%2d/27] %s\n', i, tag);
    ds = density_stats(c.x);

    % ---- model P, adaptive 12 -> 24 ------------------------------------
    [P, VP, tP] = adaptive(c, 'P', [12 24]);
    % ---- E1 recomputed (control: must reproduce the recorded E1) --------
    [E, VE, tE] = adaptive(c, 'E1', 12);
    % ---- the generating method's own native model (control 1) ----------
    nm = nativeModel.(c.method);
    if strcmp(nm, 'P'), N = P; tN = struct('assembly_s',0,'eigs_s',0,'batch_s',0);
    else, [N, ~, tN] = adaptive(c, nm, 12); end

    % ---- control 4: stringent / independently started re-solve ---------
    suspicious = ~P.eigenpair_valid(1) || ~P.valid_structural(1) || ...
        abs(P.omega(1) - E.selected_omega)/E.selected_omega > 0.05;
    stringent = struct('run', false, 'max_rel_diff_omega', NaN, 'classification_unchanged', NaN);
    if suspicious
        [Kf,Mf,md,Ee,rr,zeff] = pencil(c.x, c.nelx, c.nely, 'P');
        k = numel(P.omega);
        [B2, ok2] = solve_batch(Kf,Mf,md,Ee,rr,zeff,k, 1e-14, 7);
        if ok2
            stringent = struct('run', true, ...
                'max_rel_diff_omega', max(abs(B2.omega - P.omega)./P.omega), ...
                'classification_unchanged', isequal(B2.valid_structural, P.valid_structural));
        end
    end

    % ---- supplementary: targeted structural search and exact count -----
    % Only used where the declared 24-mode search is exhausted.  Shift-invert
    % around the E1 structural eigenvalue; the P mode best matching the E1
    % structural mode (MAC) that also passes the unchanged classifier.  The
    % number of eigenvalues below it is counted exactly by Sylvester inertia.
    % Mode correspondence uses a MASS-WEIGHTED MAC with the E1 mass matrix
    % (same saved field): unweighted MAC is dominated by the near-massless,
    % large-amplitude nodes of the P model and understates correspondence.
    [~,ME1] = pencil(c.x, c.nelx, c.nely, 'E1');
    vE = VE(:, E.selected_ordinal);
    [Kf,Mf,md,Ee,rr,zeff] = pencil(c.x, c.nelx, c.nely, 'P');
    tgt = struct('omega', NaN, 'mac_E1', NaN, 'structural', false, 'voidKE', NaN, 'dwp', NaN, ...
        'best_mac_any', NaN, 'best_mac_any_structural', NaN, 'n_structural_near', NaN);
    if isfinite(P.selected_ordinal)
        lamS = P.omega(P.selected_ordinal)^2; srcS = 'FIRST_STRUCTURAL_WITHIN_SEARCH';
    else
        [T, okT, ~, VT] = solve_batch(Kf,Mf,md,Ee,rr,zeff,16,[],[],E.selected_omega^2);
        lamS = NaN; srcS = 'NONE';
        if okT
            macT = macw(VT, vE, ME1);
            [bm, bi] = max(macT);
            tgt.best_mac_any = bm; tgt.best_mac_any_structural = T.valid_structural(bi);
            tgt.n_structural_near = sum(T.valid_structural);
            cand = find(T.valid_structural);
            if ~isempty(cand)
                [~, b] = max(macT(cand)); jT = cand(b);
                tgt.omega = T.omega(jT); tgt.mac_E1 = macT(jT); tgt.structural = true;
                tgt.voidKE = T.voidKE(jT); tgt.dwp = T.densityParticipation(jT);
                lamS = T.omega(jT)^2; srcS = 'TARGETED_NEAR_E1';
            end
        end
    end
    nBelow = NaN;
    if isfinite(lamS), nBelow = count_below(Kf, Mf, lamS*(1 - 1e-7)); end
    clear Kf Mf

    % ---- modal correspondence: MAC between P modes and E1 modes ---------
    mac = macmatrix(VP, VE); macM = macw(VP, VE, ME1);
    jP = P.selected_ordinal; jE = E.selected_ordinal;
    macSel = NaN; macSelM = NaN;
    if isfinite(jP) && isfinite(jE), macSel = mac(jP, jE); macSelM = macM(jP, jE); end
    % Shape correspondence of the lowest P mode with ANY E1 mode (weighted).
    [macLowBest, macLowIdx] = max(macM(1,:));
    macStore = [macStore, struct('case', tag, 'mac', mac, 'mac_weighted_E1mass', macM)]; %#ok<AGROW>

    nRej = NaN; if isfinite(jP), nRej = jP - 1; end
    classes = classify_lowest(P);
    rows = [rows, struct( ...
        'method', c.method, 'mesh', sprintf('%dx%d', c.nelx, c.nely), 'nelx', c.nelx, 'nely', c.nely, ...
        'omega1_native_recorded', c.omega_native_recorded(1), ...
        'omega1_native_recomputed', N.omega(1), ...
        'native_rel_diff_modes_1to3', max(abs(N.omega(1:3) - c.omega_native_recorded(1:3)) ./ c.omega_native_recorded(1:3)), ...
        'omega1_E1_recorded', c.E1_recorded, 'omega1_E1_recomputed', E.selected_omega, ...
        'E1_rel_diff', abs(E.selected_omega - c.E1_recorded)/c.E1_recorded, ...
        'E1_ordinal_recomputed', E.selected_ordinal, ...
        'P_omega1_algebraic', P.omega(1), 'P_mode1_class', classes, ...
        'P_mode1_voidKE', P.voidKE(1), 'P_mode1_voidSE', P.voidSE(1), 'P_mode1_dwp', P.densityParticipation(1), ...
        'P_structural_omega', P.selected_omega, 'P_structural_index', jP, 'P_rejected_below_structural', nRej, ...
        'P_status', P.status, 'P_modes_requested', P.modes_requested_final, ...
        'P_targeted_structural_omega', tgt.omega, 'P_targeted_MACw_E1', tgt.mac_E1, ...
        'P_targeted_voidKE', tgt.voidKE, 'P_targeted_dwp', tgt.dwp, ...
        'P_structural_reference', srcS, 'P_count_below_structural_exact', nBelow, ...
        'P_structural_vs_E1_rel_diff', (P.selected_omega - E.selected_omega)/E.selected_omega, ...
        'MACw_Pstructural_E1selected', macSelM, 'MAC_unweighted_Pstructural_E1selected', macSel, ...
        'MACw_Pmode1_bestE1', macLowBest, 'MACw_Pmode1_bestE1_index', macLowIdx, ...
        'E1_omega_at_Pmode1_match', E.omega(macLowIdx), ...
        'E1_classified_structural_at_Pmode1_match', E.valid_structural(macLowIdx), ...
        'P_targeted_best_MACw_any', tgt.best_mac_any, 'P_targeted_best_is_structural', tgt.best_mac_any_structural, ...
        'P_targeted_n_structural_among16', tgt.n_structural_near, ...
        'frac_x_zero', ds.zero, 'frac_x_0_001', ds.b0_001, 'frac_x_001_01', ds.b001_01, ...
        'frac_x_ge_01', ds.ge01, 'frac_x_1e3_to_1p1e3', ds.nearfloor, 'frac_x_0_to_2e3', ds.below2e3, ...
        'frac_x_3e4_to_3e3', ds.worst, ...
        'x_min', min(c.x), 'grayness', mean(4*c.x.*(1-c.x)), ...
        'stringent_rerun', stringent.run, 'stringent_max_rel_diff', stringent.max_rel_diff_omega, ...
        'stringent_classification_unchanged', stringent.classification_unchanged, ...
        'x_sha256', c.x_sha256, 'source', c.source)]; %#ok<AGROW>

    sets = {P, 'P'; E, 'E1'};
    if ~strcmp(nm, 'P'), sets(end+1,:) = {N, ['native_' nm]}; end
    for m = 1:size(sets,1)
        B = sets{m,1}; label = sets{m,2};
        for j = 1:numel(B.omega)
            modes = [modes, struct('method', c.method, 'mesh', sprintf('%dx%d', c.nelx, c.nely), ...
                'model', label, 'j', j, 'omega', B.omega(j), 'residual', B.eigenpair_residual(j), ...
                'voidKE', B.voidKE(j), 'voidSE', B.voidSE(j), 'dwp', B.densityParticipation(j), ...
                'IPR', B.IPR(j), 'eigenpair_valid', B.eigenpair_valid(j), ...
                'structural', B.valid_structural(j))]; %#ok<AGROW>
        end
    end
    timing = [timing, struct('method', c.method, 'mesh', sprintf('%dx%d', c.nelx, c.nely), ...
        'free_dofs', 2*(c.nelx+1)*(c.nely+1)-4, ...
        'P_assembly_s', tP.assembly_s, 'P_eigs_s', tP.eigs_s, 'P_batch_s', tP.batch_s, ...
        'E1_batch_s', tE.batch_s, 'native_batch_s', tN.batch_s)]; %#ok<AGROW>
end

% ---- control 2: element stiffness-to-mass ratio against solid --------
xs = [0 1e-9 1e-6 1e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 0.1 0.2 0.5 1];
km = struct('x', num2cell(xs));
for i = 1:numel(xs)
    for mdl = {'P','Y','D','E1'}
        [e, r] = material(xs(i), mdl{1});
        km(i).(['ratio_' mdl{1}]) = (e/1e7) / r;
    end
end

runtime = toc(t0);
prov.study = struct('script', 'examples/Performance/spurious_mode_crosscheck/run_spurious_mode_crosscheck.m', ...
    'script_sha256', sha256([mfilename('fullpath') '.m']), ...
    'evaluator_copied_from', 'examples/Performance/benchmark_profile/study_evaluate_design.m', ...
    'evaluator_sha256', sha256(fullfile(repo,'examples','Performance','benchmark_profile','study_evaluate_design.m')), ...
    'repository_head', strtrim(gitcmd(repo,'rev-parse HEAD')), ...
    'repository_dirty', ~isempty(strtrim(gitcmd(repo,'status --porcelain'))), ...
    'matlab_version', version, 'computer', computer, 'max_num_comp_threads', maxNumCompThreads, ...
    'hostname', strtrim(gitcmd(repo,'','hostname')), 'generated', char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssXXX','TimeZone','local')), ...
    'campaign_runtime_s', runtime);

writetable(struct2table(rows), fullfile(resDir, 'per_case.csv'));
writetable(struct2table(modes), fullfile(resDir, 'modes_all.csv'));
writetable(struct2table(timing), fullfile(resDir, 'timing.csv'));
writetable(struct2table(km), fullfile(resDir, 'stiffness_to_mass_ratio.csv'));
writejson(fullfile(resDir, 'per_case.json'), rows);
writejson(fullfile(resDir, 'provenance.json'), prov);
save(fullfile(resDir, 'results.mat'), 'rows', 'modes', 'timing', 'km', 'prov', 'macStore', '-v7.3');
fprintf('Campaign runtime %.1f s\n', runtime);
out = rows;
end

% =========================================================================
function [out, V, t] = adaptive(c, model, schedule)
tA = tic; [Kf,Mf,md,Ee,rr,zeff] = pencil(c.x, c.nelx, c.nely, model); t.assembly_s = toc(tA);
t.eigs_s = 0; t.batch_s = 0;
for s = 1:numel(schedule)
    tB = tic; [out, ok, msg, V, tEig] = solve_batch(Kf,Mf,md,Ee,rr,zeff,schedule(s));
    t.batch_s = t.batch_s + toc(tB); t.eigs_s = t.eigs_s + tEig;
    assert(ok, '%s %dx%d model %s: %s', c.method, c.nelx, c.nely, model, msg);
    out.modes_requested_final = schedule(s);
    j = find(out.valid_structural, 1, 'first');
    if ~isempty(j)
        out.status = 'PASS'; out.selected_ordinal = j; out.selected_omega = out.omega(j); return
    end
end
out.status = sprintf('UNRESOLVED_WITHIN_%d', schedule(end));
out.selected_ordinal = NaN; out.selected_omega = NaN;
end

function n = count_below(Kf, Mf, sigma)
% Number of eigenvalues of (Kf, Mf) below sigma = number of negative
% eigenvalues of Kf - sigma*Mf (Sylvester's law of inertia), from the
% block-diagonal factor of a sparse LDL' decomposition.
[~, D, ~] = ldl(Kf - sigma*Mf, 'vector');
d = full(diag(D)); e = full([diag(D,1); 0]); n = 0; i = 1; N = numel(d);
while i <= N
    if i < N && e(i) ~= 0
        a = d(i); b = e(i); c = d(i+1); dt = a*c - b*b;
        if dt < 0, n = n + 1; elseif a + c < 0, n = n + 2; end
        i = i + 2;
    else
        n = n + (d(i) < 0); i = i + 1;
    end
end
end

function s = classify_lowest(B)
if ~B.eigenpair_valid(1), s = 'INVALID_EIGENPAIR';
elseif B.valid_structural(1), s = 'STRUCTURAL';
else
    f = {};
    if B.voidKE(1) >= 0.5, f{end+1} = 'voidKE'; end
    if B.voidSE(1) >= 0.5, f{end+1} = 'voidSE'; end
    if B.densityParticipation(1) <= 0.5, f{end+1} = 'dwp'; end
    s = ['REJECTED:' strjoin(f, '+')];
end
end

function d = density_stats(x)
d = struct('zero', mean(x == 0), 'b0_001', mean(x > 0 & x < 0.01), ...
    'b001_01', mean(x >= 0.01 & x < 0.1), 'ge01', mean(x >= 0.1), ...
    'nearfloor', mean(x >= 1e-3 & x < 1.1e-3), 'below2e3', mean(x > 0 & x < 2e-3), ...
    'worst', mean(x >= 3e-4 & x <= 3e-3));   % half a decade around the P-model k/m minimum (x ~ 1e-3)
end

function M = macmatrix(A, B)
M = abs(A'*B).^2 ./ (sum(A.^2,1)' * sum(B.^2,1));
end

function W = macw(A, B, M)
% Mass-weighted MAC with one common mass matrix M.
MB = M*B; MA = M*A;
W = abs(A'*MB).^2 ./ (sum(A.*MA,1)' * sum(B.*MB,1));
end

function [E, rho] = material(x, model)
switch model
    case 'P',  E = 1e7*(1e-9+(1-1e-9)*x.^3); rho = 1e-9+(1-1e-9)*x;
    case 'Y',  E = 1e7*(1e-9+(1-1e-9)*x.^3); g = x; lo = x <= 0.1; g(lo) = x(lo).^6; rho = 1e-9+(1-1e-9)*g;
    case 'D',  g = x.^3; lo = x < 0.1; g(lo) = 0.01*x(lo); E = 1e7*g; rho = x;
    case 'E1', E = 1e7*(1e-6+(1-1e-6)*x.^3); rho = 1e-6+(1-1e-6)*x;
end
end

% =========================================================================
% Verbatim from study_evaluate_design.m, with P/Y/D branches added to PENCIL,
% optional (tol, seed) for the control-4 re-solve, and V/eigs-time outputs.
function [out,ok,message,V,tEig] = solve_batch(Kf,Mf,md,Ee,rr,zeff,k,tol,seed,sigma)
if nargin < 8 || isempty(tol), tol = 1e-10; end
if nargin < 9 || isempty(seed), seed = 42; end
if nargin < 10, sigma = []; end
out = empty_modal(''); ok = false; message = ''; V = []; tEig = NaN;
opts=struct('disp',0,'maxit',200000,'tol',tol,'v0',deterministic_v0(size(Kf,1),seed));
try
    tE = tic;
    if ~isempty(sigma)
        [V,D]=eigs(Kf,Mf,k,sigma,opts);   % supplementary targeted search only
    else
        try, [V,D]=eigs(Kf,Mf,k,'smallestabs',opts);
        catch, [V,D]=eigs(Kf,Mf,k,'sm',opts); end
    end
    tEig = toc(tE);
catch ME
    message = ['EIGENSOLVER_FAILURE:' ME.identifier]; return
end
lam=real(diag(D)); [lam,ix]=sort(lam,'ascend'); V=V(:,ix);
omega=sqrt(max(lam,0)); ndof=md.ndof; U=zeros(ndof,k); U(md.free,:)=V;
voidKE=nan(k,1);voidSE=nan(k,1);dwp=nan(k,1);ipr=nan(k,1);
keTotal=nan(k,1);seTotal=nan(k,1);residual=nan(k,1);
diagnosticFinite=false(k,1);eigenpairValid=false(k,1);low=zeff<=0.1;
for j=1:k
    u=V(:,j); denom=norm(Kf*u)+abs(lam(j))*norm(Mf*u)+eps;
    residual(j)=norm(Kf*u-lam(j)*(Mf*u))/denom;
    eigenpairValid(j)=isfinite(lam(j))&&lam(j)>0&&isfinite(residual(j))&&residual(j)<=1e-6;
    ue=reshape(U(md.edof,j),size(md.edof));
    ke=rr.*sum((ue*md.ME).*ue,2); se=Ee.*sum((ue*md.KE).*ue,2);
    ke=max(ke,0);se=max(se,0);keTotal(j)=sum(ke);seTotal(j)=sum(se);
    if isfinite(keTotal(j))&&keTotal(j)>0&&isfinite(seTotal(j))&&seTotal(j)>0
        ken=ke/keTotal(j);sen=se/seTotal(j);voidKE(j)=sum(ken(low));voidSE(j)=sum(sen(low));
        dwp(j)=sum(ken.*zeff);ipr(j)=sum(ken.^2);
        diagnosticFinite(j)=all(isfinite([voidKE(j),voidSE(j),dwp(j),ipr(j)]));
    end
end
margins=[0.5-voidKE,0.5-voidSE,dwp-0.5];
valid=eigenpairValid&diagnosticFinite&all(margins>0,2);
out.lambda=lam;out.omega=omega;out.eigenpair_residual=residual;out.eigenpair_valid=eigenpairValid;
out.diagnostic_finite=diagnosticFinite;out.voidKE=voidKE;out.voidSE=voidSE;
out.densityParticipation=dwp;out.IPR=ipr;out.kinetic_energy_total=keTotal;
out.strain_energy_total=seTotal;out.condition_margins=margins;
out.minimum_margin=min(margins,[],2);out.valid_structural=valid;ok=true;
end

function out=empty_modal(model)
out=struct('model',model,'classifier','UNANIMOUS_ALL_THREE','rho_void_threshold',0.1, ...
    'voidKE_threshold',0.5,'voidSE_threshold',0.5,'densityParticipation_threshold',0.5, ...
    'IPR_role','NONBINDING_QA','status','STRUCTURAL_MODE_NOT_FOUND','solver_status','NOT_RUN', ...
    'failure_reason','','modes_requested_final',0,'escalation_count',0,'batch_schedule',nan(0,1), ...
    'lambda',nan(0,1),'omega',nan(0,1),'eigenpair_residual',nan(0,1), ...
    'eigenpair_valid',false(0,1),'diagnostic_finite',false(0,1),'voidKE',nan(0,1), ...
    'voidSE',nan(0,1),'densityParticipation',nan(0,1),'IPR',nan(0,1), ...
    'kinetic_energy_total',nan(0,1),'strain_energy_total',nan(0,1), ...
    'condition_margins',nan(0,3),'minimum_margin',nan(0,1),'valid_structural',false(0,1), ...
    'selected_ordinal',NaN,'selected_lambda',NaN,'selected_omega',NaN, ...
    'selected_voidKE',NaN,'selected_voidSE',NaN,'selected_densityParticipation',NaN, ...
    'selected_IPR',NaN,'selected_condition_margins',nan(1,3), ...
    'selected_minimum_margin',NaN,'matrix_free_dofs',NaN,'technical_mode_limit',NaN);
end

function [Kf,Mf,md,Ee,rr,zeff]=pencil(z,nelx,nely,model)
z = max(0, min(1, double(z(:))));   % the evaluator's own clamp; a no-op on these fields
[KE,ME]=q4_matrices(8/nelx,1/nely,0.3,1.0);[iK,jK,edof]=assembly_indices(nelx,nely);
switch model
    case 'E1'
        zeff=z;Ee=1e7*(1e-6+(1-1e-6)*z.^3);rr=1e-6+(1-1e-6)*z;
    case {'P','Y','D'}
        zeff=z;[Ee,rr]=material(z,model);
    otherwise, error('ie2a:UnknownEvaluator','Unknown evaluator model %s.',model);
end
ndof=2*(nelx+1)*(nely+1);K=sparse(iK,jK,reshape(KE(:)*Ee',[],1),ndof,ndof);K=(K+K')/2;
M=sparse(iK,jK,reshape(ME(:)*rr',[],1),ndof,ndof);M=(M+M')/2;
jMid=round(nely/2);nL=jMid;nR=nelx*(nely+1)+jMid;
fixed=[2*nL+1;2*nL+2;2*nR+1;2*nR+2];free=setdiff((1:ndof)',fixed);
Kf=K(free,free);Mf=M(free,free);md=struct('KE',KE,'ME',ME,'edof',edof,'ndof',ndof,'free',free);
end

function v=deterministic_v0(n,seed)
if nargin < 2, seed = 42; end
s=RandStream('twister','Seed',seed);v=randn(s,n,1);v=v/norm(v);
end

function [iK,jK,edof]=assembly_indices(nelx,nely)
nEl=nelx*nely;edof=zeros(nEl,8);
for ex=0:nelx-1
    for ey=0:nely-1
        e=ey+ex*nely+1;n1=(nely+1)*ex+ey;n2=(nely+1)*(ex+1)+ey;
        edof(e,:)=[2*n1+1 2*n1+2 2*n2+1 2*n2+2 2*(n2+1)+1 2*(n2+1)+2 2*(n1+1)+1 2*(n1+1)+2];
    end
end
iK=reshape(kron(edof,ones(1,8))',[],1);jK=reshape(kron(edof,ones(8,1))',[],1);
end

function [KE,ME]=q4_matrices(hx,hy,nu,t)
D=(1/(1-nu^2))*[1 nu 0;nu 1 0;0 0 0.5*(1-nu)];invJ=[2/hx 0;0 2/hy];
detJ=0.25*hx*hy;gp=1/sqrt(3);KE=zeros(8);
for xi=[-gp gp]
    for eta=[-gp gp]
        a=0.25*[-(1-eta) (1-eta) (1+eta) -(1+eta)];
        b=0.25*[-(1-xi) -(1+xi) (1+xi) (1-xi)];d=invJ*[a;b];B=zeros(3,8);
        B(1,1:2:end)=d(1,:);B(2,2:2:end)=d(2,:);B(3,1:2:end)=d(2,:);B(3,2:2:end)=d(1,:);
        KE=KE+B'*D*B*detJ;
    end
end
KE=t*KE;Ms=(hx*hy/36)*[4 2 1 2;2 4 2 1;1 2 4 2;2 1 2 4];ME=t*kron(Ms,eye(2));
end

% =========================================================================
function out = figures(resDir, figDir, repo)
R = load(fullfile(resDir, 'results.mat'));
rows = R.rows; cases = load_cases(repo);
keys = {'proposed','yuksel','olhoff'}; names = {'Proposed','Yuksel-Yilmaz','Du-Olhoff'};
col = [0.4660 0.6740 0.1880; 0.8500 0.3250 0.0980; 0 0.4470 0.7410];
out = struct();

% ---- F1: frequencies per method and mesh ------------------------------
f = figure('Visible','off','Color','white','Position',[50 50 1500 430]);
tl = tiledlayout(f,1,3,'TileSpacing','compact');
for m = 1:3
    ax = nexttile(tl); hold(ax,'on'); r = rows(strcmp({rows.method}, keys{m}));
    ne = [r.nelx].*[r.nely];
    plot(ax, ne, [r.omega1_native_recorded], 'o-', 'Color', [.5 .5 .5], 'DisplayName', '\omega_1 native (own model)');
    plot(ax, ne, [r.omega1_E1_recorded], 's-', 'Color', 'k', 'DisplayName', '\omega_1 E1');
    plot(ax, ne, [r.P_omega1_algebraic], 'v--', 'Color', [.85 .1 .1], 'MarkerFaceColor', [.85 .1 .1], 'DisplayName', 'Proposed model: lowest \omega');
    ps = [r.P_structural_omega]; tg = [r.P_targeted_structural_omega]; useT = ~isfinite(ps);
    ps(useT) = tg(useT);
    plot(ax, ne, ps, '^-', 'Color', col(m,:), 'MarkerFaceColor', col(m,:), 'DisplayName', 'Proposed model: structural \omega');
    if any(useT)
        plot(ax, ne(useT), ps(useT), 'o', 'MarkerSize', 11, 'Color', col(m,:), 'LineWidth', 1.2, ...
            'DisplayName', 'above the 24-mode search (targeted)');
    end
    set(ax,'XScale','log','YScale','log'); grid(ax,'on'); box(ax,'on'); title(ax, names{m});
    xlabel(ax,'Number of elements N_e'); ylabel(ax,'\omega [rad/s]'); ylim(ax,[10 300]);
    legend(ax,'Location','east','FontSize',8);
end
title(tl, 'Saved designs re-evaluated with the Proposed material model (E_{min}=10^{-9}E_0, linear mass)');
out.F1 = savefigpair(f, fullfile(figDir, 'F1_frequencies_by_model'));

% ---- F2: stiffness-to-mass ratio of an element ------------------------
f = figure('Visible','off','Color','white','Position',[50 50 700 480]); ax = axes(f); hold(ax,'on');
xs = logspace(-9, 0, 400);
mdl = {'P','Y','D','E1'}; lab = {'Proposed native','Yuksel-Yilmaz native','Du-Olhoff native','E1'};
st = {'-','--','-.',':'}; c4 = [.85 .1 .1; col(2,:); col(3,:); 0 0 0];
for i = 1:4
    [e, r] = material(xs, mdl{i}); plot(ax, xs, (e/1e7)./r, st{i}, 'Color', c4(i,:), 'LineWidth', 1.8, 'DisplayName', lab{i});
end
set(ax,'XScale','log','YScale','log'); grid(ax,'on'); box(ax,'on');
xlabel(ax,'Element density x'); ylabel(ax,'(E(x)/E_0) / (\rho(x)/\rho_0)');
title(ax,'Element stiffness-to-mass ratio relative to solid'); legend(ax,'Location','northwest');
yline(ax,1,':','HandleVisibility','off'); xline(ax,1e-3,':','Du-Olhoff floor','HandleVisibility','off');
out.F2 = savefigpair(f, fullfile(figDir, 'F2_stiffness_to_mass_ratio'));

% ---- F3: mode shapes / energy localization, representative cases ------
reps = pick_representatives(rows);
out.representatives = reps;
f = figure('Visible','off','Color','white','Position',[50 50 1700 170*numel(reps)]);
tl = tiledlayout(f, numel(reps), 3, 'TileSpacing','compact');
for i = 1:numel(reps)
    c = cases(strcmp({cases.method}, reps(i).method) & [cases.nelx] == reps(i).nelx);
    [Kf,Mf,md,Ee,rr,zeff] = pencil(c.x, c.nelx, c.nely, 'P');
    r = rows(strcmp({rows.method}, c.method) & [rows.nelx] == c.nelx);
    k = r.P_modes_requested; [B, ~, ~, V] = solve_batch(Kf,Mf,md,Ee,rr,zeff,k);
    if ~isfinite(r.P_structural_index) && isfinite(r.P_targeted_structural_omega)
        % structural mode lies above the declared search: re-derive the targeted one
        [T, ~, ~, VT] = solve_batch(Kf,Mf,md,Ee,rr,zeff,16,[],[],r.omega1_E1_recomputed^2);
        [~, jT] = min(abs(T.omega - r.P_targeted_structural_omega));
        tB = T; tV = VT; tj = jT;
    else
        tB = B; tV = V; tj = r.P_structural_index;
    end
    nameI = names{strcmp(keys, c.method)};
    ax = nexttile(tl); imagesc(ax, reshape(c.x, c.nely, c.nelx)); axis(ax,'image'); set(ax,'YDir','normal');
    colormap(ax, flipud(gray)); clim(ax,[0 1]); colorbar(ax);
    title(ax, sprintf('%s %dx%d: saved density', nameI, c.nelx, c.nely), 'FontWeight', 'normal', 'FontSize', 9);
    for q = 1:2
        if q == 1, j = 1; BB = B; VV = V; else, j = tj; BB = tB; VV = tV; end
        ax = nexttile(tl);
        if ~isfinite(j), axis(ax,'off'); title(ax,'no structural mode found'); continue; end
        U = zeros(md.ndof,1); U(md.free) = VV(:,j); ue = reshape(U(md.edof), size(md.edof));
        ke = rr.*sum((ue*md.ME).*ue,2); ke = ke/sum(ke);
        imagesc(ax, reshape(log10(max(ke,1e-12)), c.nely, c.nelx)); axis(ax,'image'); set(ax,'YDir','normal');
        colormap(ax, parula); clim(ax, [-8 max(log10(max(ke)))]); colorbar(ax);
        if BB.valid_structural(j), kind = 'structural'; else, kind = 'rejected'; end
        if q == 2 && ~isfinite(r.P_structural_index)
            lbl = sprintf('mode #%d (targeted)', r.P_count_below_structural_exact + 1);
        else
            lbl = sprintf('mode %d', j);
        end
        title(ax, sprintf('%s, \\omega = %.2f, %s, voidKE %.2f', lbl, BB.omega(j), kind, BB.voidKE(j)), ...
            'FontWeight', 'normal', 'FontSize', 9);
    end
end
title(tl, {'Proposed material model on saved designs: density | lowest mode | structural mode', ...
    'mode panels: log_{10} of each element''s share of the modal kinetic energy'});
out.F3 = savefigpair(f, fullfile(figDir, 'F3_mode_energy_localization'));

% ---- F4: low-density distribution ------------------------------------
f = figure('Visible','off','Color','white','Position',[50 50 1100 420]);
tl = tiledlayout(f,1,3,'TileSpacing','compact');
bands = {'frac_x_zero','frac_x_0_001','frac_x_001_01'}; bl = {'x = 0','0 < x < 0.01','0.01 \leq x < 0.1'};
for b = 1:3
    ax = nexttile(tl); hold(ax,'on');
    for m = 1:3
        r = rows(strcmp({rows.method}, keys{m}));
        plot(ax, [r.nelx].*[r.nely], 100*[r.(bands{b})], 'o-', 'Color', col(m,:), 'MarkerFaceColor', col(m,:), 'DisplayName', names{m});
    end
    set(ax,'XScale','log'); grid(ax,'on'); box(ax,'on'); title(ax, bl{b});
    xlabel(ax,'Number of elements N_e'); ylabel(ax,'Share of elements [%]'); legend(ax,'Location','best','FontSize',8);
end
title(tl, 'Low-density element fractions of the saved designs');
out.F4 = savefigpair(f, fullfile(figDir, 'F4_low_density_fractions'));
writejson(fullfile(resDir, 'figures_manifest.json'), out);
end

function reps = pick_representatives(rows)
% Fixed before looking at the figures: the published anomaly (Proposed
% 160x20), the only Yuksel design with rejected low modes (160x20), and the
% finest Du-Olhoff design (800x100).
reps = struct('method', {'proposed','yuksel','olhoff'}, 'nelx', {160, 160, 800});
reps = reps(arrayfun(@(q) any(strcmp({rows.method}, q.method) & [rows.nelx] == q.nelx), reps));
end

function files = savefigpair(f, base)
exportgraphics(f, [base '.png'], 'Resolution', 170, 'BackgroundColor', 'white');
set(f, 'CreateFcn', 'set(gcbo,''Visible'',''on'')'); savefig(f, [base '.fig']); close(f);
files = {[base '.png'], [base '.fig']};
end

% =========================================================================
function h = sha256(path)
[st, o] = system(sprintf('shasum -a 256 "%s"', path)); assert(st == 0, o); h = strtok(o);
end
function h = vec_sha(v)
tmp = [tempname '.bin']; fid = fopen(tmp,'w'); fwrite(fid, v, 'double'); fclose(fid);
h = sha256(tmp); delete(tmp);
end
function p = relpath(p, repo), p = strrep(p, [repo filesep], ''); end
function o = gitcmd(repo, args, raw)
if nargin == 3, [~, o] = system(raw); return; end
[~, o] = system(sprintf('git -C "%s" %s', repo, args));
end
function writejson(p, s)
fid = fopen(p,'w'); c = onCleanup(@() fclose(fid)); fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true));
end
