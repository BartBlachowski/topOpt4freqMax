function T = r3_w1_collect(recordsFile)
%R3_W1_COLLECT  Reviewer 3, W1: mesh convergence of the response.  Collects
%   (a) proposed / Yuksel-Yilmaz with a fixed physical filter radius R = 0.06 m
%       (results/r3/w1a_fixed_radius.csv, w1a_designs.mat from r3_w1_fixed_radius),
%   (b) the native runs (radius 2 / 2.5 elements) from the W2 timing runs
%       (results/r3/w2_timing.csv; designs are deterministic, the frequencies are those
%       of Table 2),
%   (c) Du-Olhoff: the stored nine-mesh designs of the Table 2 campaign (research
%       repository, examples/Performance/paper_benchmark/campaign_mac_stagnation_x_PAPER/
%       benchmark_records.mat), evaluated here with the common evaluator,
%   writes results/r3/w1_mesh_convergence.csv and the 3x3 design figure
%   results/r3/w1_designs_3x3.png (rows: Du-Olhoff, Yuksel-Yilmaz R = 0.06 m, proposed
%   R = 0.06 m; columns 160x20, 400x50, 800x100).
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
maxNumCompThreads(1);
out = results_dir('r3');
meshes = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
A = readStr(fullfile(out, 'w1a_fixed_radius.csv'));
B = readStr(fullfile(out, 'w2_timing.csv'));
S = load(fullfile(out, 'w1a_designs.mat'));  W1a = S.designs;
R = load(recordsFile, 'records');  recs = R.records;
% omega1 under the common evaluator stored with the campaign (JSON next to the .mat file)
J = jsondecode(fileread(fullfile(fileparts(recordsFile), 'benchmark_results.json')));  J = J.runs;
if ~iscell(J), J = num2cell(J); end
rows = {};  DOx = cell(9,1);
for k = 1:9
    prob = problem_ss_beam(meshes(k,1), meshes(k,2));  ms = sprintf('%dx%d', meshes(k,:));
    % (c) Du-Olhoff stored design
    q = [];
    for j = 1:numel(recs)
        r = recs(j);
        if isequal(double(r.mesh(:)'), meshes(k,:)) && contains(lower(string(r.method_key)), 'olhoff') && ~r.is_warmup
            q = r;
        end
    end
    if isempty(q), error('Du-Olhoff design %s missing', ms); end
    xs = double(q.x(:));
    cand = {xs, reshape(flipud(reshape(xs, prob.nely, prob.nelx)), [], 1), reshape(reshape(xs, prob.nelx, prob.nely)', [], 1)};
    om = zeros(2,3);
    for c = 1:3, om(:,c) = structural_modes(prob, cand{c}, 2); end
    ref = NaN;
    for j = 1:numel(J)
        if isequal(double(J{j}.mesh(:)'), meshes(k,:)) && contains(lower(string(J{j}.method_key)), 'olhoff') && ~J{j}.is_warmup
            ref = J{j}.omega1_common_E1;
        end
    end
    [dmin, c] = min(abs(om(1,:) - ref));  DOx{k} = cand{c};
    assert(dmin <= 1e-3*ref, 'stored Du-Olhoff design %s: omega1 %.3f does not reproduce the stored %.3f', ms, om(1,c), ref);
    rows(end+1,:) = {ms, prod(meshes(k,:)), 'du_olhoff', 'R = 0.06 m (native)', 0.06/(prob.L/prob.nelx), ...
        sprintf('%d+%d', q.counts.outer_iterations, q.counts.inner_iterations_total), om(1,c), om(2,c), ref}; %#ok<AGROW>
    % (b) native static methods (W2)
    for m = ["proposed", "yuksel"]
        b = B(B.mesh == ms & B.method == m, :);
        rel = 2;  if m == "yuksel", rel = 2.5; end
        rows(end+1,:) = {ms, prod(meshes(k,:)), char(m), sprintf('%.1f elements (native)', rel), rel, ...
            char(b.iterations), b.omega1, b.omega2, NaN}; %#ok<AGROW>
    end
    % (a) fixed physical radius
    for m = ["proposed", "yuksel"]
        a = A(A.mesh == ms & A.method == m, :);
        rows(end+1,:) = {ms, prod(meshes(k,:)), char(m), 'R = 0.06 m (fixed)', a.R_elements, ...
            char(a.iterations), a.omega1, a.omega2, NaN}; %#ok<AGROW>
    end
end
T = cell2table(rows, 'VariableNames', {'mesh','Ne','method','filter','radius_elements','iterations', ...
    'omega1','omega2','omega1_stored_E1'});
writetable(T, fullfile(out, 'w1_mesh_convergence.csv'));
% 3x3 figure
cols = [1 4 9];  rowsName = {'Du-Olhoff (R = 0.06 m)', 'Yuksel-Yilmaz, R = 0.06 m', 'Proposed, R = 0.06 m'};
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1800 520]);
tl = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:3
    for jj = 1:3
        k = cols(jj);  prob = problem_ss_beam(meshes(k,1), meshes(k,2));  ms = sprintf('%dx%d', meshes(k,:));
        switch i
            case 1, x = DOx{k};  t = T(strcmp(T.mesh, ms) & strcmp(T.method, "du_olhoff"), :);
            case 2, x = W1a(arrayfun(@(d) isequal(d.mesh, meshes(k,:)) && strcmp(d.method, 'yuksel'), W1a)).x;
                    t = T(strcmp(T.mesh, ms) & strcmp(T.method, "yuksel") & contains(T.filter, "fixed"), :);
            case 3, x = W1a(arrayfun(@(d) isequal(d.mesh, meshes(k,:)) && strcmp(d.method, 'proposed'), W1a)).x;
                    t = T(strcmp(T.mesh, ms) & strcmp(T.method, "proposed") & contains(T.filter, "fixed"), :);
        end
        ax = nexttile(tl);
        imagesc(ax, [0 prob.L], [0 prob.H], 1 - reshape(x, prob.nely, prob.nelx));
        set(ax, 'YDir', 'normal');  colormap(ax, gray);  clim(ax, [0 1]);  axis(ax, 'image', 'off');
        title(ax, sprintf('%s, %s: \\omega_1 = %.1f rad/s', rowsName{i}, ms, t.omega1), 'FontWeight', 'normal', 'FontSize', 10);
    end
end
exportgraphics(fig, fullfile(out, 'w1_designs_3x3.png'), 'Resolution', 250);  close(fig);
disp(T);
end

function T = readStr(file)
o = detectImportOptions(file);
o = setvartype(o, intersect(o.VariableNames, {'mesh','method','iterations','stop'}), 'string');
T = readtable(file, o);
end
