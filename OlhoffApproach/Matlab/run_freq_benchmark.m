% Benchmark runner for Olhoff & Du (2014) 2D beam cases.
% Runs CC, CS, SS boundary conditions and reports initial/final eigenfrequencies.
%
% Paper targets (Section 3.1):
%   CC: initial 146.1 -> optimal 456.4 (212% improvement)
%   CS: initial 104.1 -> optimal 288.7 (177% improvement)
%   SS: initial 68.7  -> optimal 174.7 (154% improvement)

clear; clc; close all;

% Task parameters (easy to tweak in one spot)
baseCfg = struct();
baseCfg.L    = 8;
baseCfg.H    = 1;
baseCfg.nelx = 240;
baseCfg.nely = 30;
baseCfg.volfrac = 0.5;
baseCfg.penal   = 3.0;
baseCfg.rmin    = 2 * baseCfg.L / baseCfg.nelx;
baseCfg.maxiter = 300;   % enough for beta continuation to reach 32
baseCfg.J       = 3;

% Material (same for all three cases)
baseCfg.E0      = 1e7;
baseCfg.Emin    = max(1e-6*baseCfg.E0, 1e-3);
baseCfg.rho0    = 1.0;
baseCfg.rho_min = 1e-6;
baseCfg.nu      = 0.3;
baseCfg.t       = 1.0;

opts = struct('doDiagnostic', true, 'diagnosticOnly', false, 'diagModes', 5, ...
    'visualization_quality', 'regular');
if isfield(baseCfg, 'visualization_quality') && ~isempty(baseCfg.visualization_quality)
    opts.visualization_quality = baseCfg.visualization_quality;
end

% Paper reference values
paper = struct();
paper.CC = struct('init', 146.1, 'opt', 456.4);
paper.CS = struct('init', 104.1, 'opt', 288.7);
paper.SS = struct('init', 68.7,  'opt', 174.7);

cases = { ...
    struct('code',"CC", 'label','Clamped-Clamped'); ...
    struct('code',"CS", 'label','Clamped-Simply'); ...
    struct('code',"SS", 'label','Simply-Simply'); ...
};

results = cell(numel(cases), 1);
for k = 1:numel(cases)
    c = cases{k};
    cfg = baseCfg;
    cfg.supportType = c.code;

    fprintf('\n================== %s ==================\n', c.label);
    fprintf('Paper: init=%.1f, opt=%.1f\n', paper.(c.code).init, paper.(c.code).opt);
    fprintf('=========================================\n');

    tic;
    [omega_best, xPhys_best, diag_out] = topFreqOptimization_MMA(cfg, opts);
    elapsed = toc;

    results{k} = struct('code', c.code, 'label', c.label, ...
        'diag', diag_out, 'omega', omega_best, 'xPhys', xPhys_best, 'time', elapsed);

    fprintf('\n--- %s Summary ---\n', c.label);
    print_block('Initial', diag_out.initial);
    print_block('Final',   diag_out.final);
    fprintf('Volume: %.4f (target %.2f)\n', mean(xPhys_best), cfg.volfrac);
    fprintf('Grayness: %.4f\n', mean(4*xPhys_best.*(1-xPhys_best)));
    fprintf('Time: %.1f sec\n', elapsed);

    % Plot topology
    figure('Position', [100+300*(k-1), 100, 400, 100]);
    theme("light");
    imgDisp = buildTopologyDisplayImage(xPhys_best, cfg.nelx, cfg.nely, opts.visualization_quality, true);
    imagesc(1 - imgDisp);
    set(gca, 'YDir', 'normal');
    axis equal tight off; colormap(gray(256));
    title(sprintf('%s: omega1=%.1f (paper: %.1f)', c.code, omega_best, paper.(c.code).opt));
end

% Final summary table
fprintf('\n\n========== FINAL BENCHMARK SUMMARY ==========\n');
fprintf('BC   | Init(code) | Init(paper) | Opt(code) | Opt(paper) | Improve\n');
fprintf('-----|------------|-------------|-----------|------------|--------\n');
for k = 1:numel(cases)
    r = results{k};
    p = paper.(r.code);
    improv = (r.diag.final.omega(1) / r.diag.initial.omega(1) - 1) * 100;
    fprintf('%s  | %10.1f | %11.1f | %9.1f | %10.1f | %5.0f%%\n', ...
        r.code, r.diag.initial.omega(1), p.init, r.diag.final.omega(1), p.opt, improv);
end
fprintf('==============================================\n');

function print_block(name, data)
    fprintf('%s eigenfreqs (rad/s): %8.2f %8.2f %8.2f\n', name, data.omega(1:min(3,end)));
end
