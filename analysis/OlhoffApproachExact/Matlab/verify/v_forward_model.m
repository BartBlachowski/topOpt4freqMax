function results = v_forward_model(meshes)
% V_FORWARD_MODEL  Phase 2 acceptance test: Fig. 2 initial eigenfrequencies.
%
%   results = v_forward_model()
%   results = v_forward_model([40 5; 80 10; 160 20])
%
%   Olhoff & Du (2014) Fig. 2 caption: for a uniform initial design with
%   rho = 0.5 the fundamental eigenfrequencies of the three beams are all
%   UNIMODAL with values
%
%       omega_1a = 68.7   (SS)     omega_1b = 104.1  (CS)     omega_1c = 146.1  (CC)
%
%   Two things this test pins down:
%
%   1. The published values are the p = 3 SIMP-PENALIZED frequencies.  Since
%      K_e = rho^p Ke* and M_e = rho Me*, omega^2 is proportional to
%      0.5^(p-1); p = 1 gives ~143/211/295, a clean factor of 2 too high.  The
%      table below prints both so the factor is visible.
%   2. The SS/CS support is a pin at MID-HEIGHT of the end edge (Fig. 2a, apex
%      of the triangle on the edge at ~0.45-0.50 b), not a bottom-corner pin.
%      The 'SS_corner' row shows what the corner reading gives (~100, arch
%      action), i.e. why it is ruled out.
%
%   Run from analysis/OlhoffApproachExact/Matlab:  addpath('.'); v_forward_model

if nargin < 1 || isempty(meshes)
    meshes = [40 4; 80 10; 160 20; 240 30];
end

targets = struct('SS', 68.7, 'CS', 104.1, 'CC', 146.1);
bcs     = {'SS','CS','CC'};

fprintf('\n=== Phase 2: forward model vs Olhoff & Du (2014) Fig. 2 ===\n');
fprintf('uniform rho = 0.5, E = 1e7, nu = 0.3, rho_m = 1, a = 8, b = 1\n\n');

results = struct('mesh',{},'bc',{},'omega1_p3',{},'omega1_p1',{},'err_pct',{}, ...
                 'omega2_over_1',{});

for mi = 1:size(meshes,1)
    nelx = meshes(mi,1);  nely = meshes(mi,2);
    fprintf(' mesh %3dx%-3d  %-4s  %-10s %-10s %-9s %-9s %s\n', ...
        nelx, nely, 'BC', 'w1(p=3)', 'w1(p=1)', 'target', 'err %', 'w2/w1');
    for bi = 1:numel(bcs)
        bc = bcs{bi};
        o3 = initial_omega(nelx, nely, bc, 3);
        o1 = initial_omega(nelx, nely, bc, 1);
        tgt = targets.(bc);
        err = 100*(o3(1) - tgt)/tgt;
        fprintf('               %-4s  %-10.4f %-10.4f %-9.1f %+8.3f  %.3f\n', ...
            bc, o3(1), o1(1), tgt, err, o3(2)/o3(1));
        r = struct('mesh',[nelx nely],'bc',bc,'omega1_p3',o3(1),'omega1_p1',o1(1), ...
                   'err_pct',err,'omega2_over_1',o3(2)/o3(1));
        results(end+1) = r; %#ok<AGROW>
    end
    fprintf('\n');
end

% --- Support-interpretation check on one mesh -------------------------------
nelx = 160; nely = 20;
o_mid    = initial_omega(nelx, nely, 'SS', 3);
o_corner = initial_omega_corner(nelx, nely, 3);
fprintf(' SS support interpretation at %dx%d (p = 3):\n', nelx, nely);
fprintf('   mid-height pin  (Fig. 2a, USED)  omega_1 = %8.4f   target 68.7\n', o_mid(1));
fprintf('   bottom-corner pin (ruled out)    omega_1 = %8.4f\n\n', o_corner(1));

% --- Odd-nely guard ---------------------------------------------------------
fprintf(' odd-nely guard: ');
try
    initial_omega(40, 5, 'SS', 3);
    fprintf('FAIL - no error raised for nely = 5\n');
catch ME
    if strcmp(ME.identifier, 'build_supports_exact:OddNely')
        fprintf('PASS - %s\n', ME.identifier);
    else
        fprintf('FAIL - wrong error: %s\n', ME.identifier);
    end
end

fprintf('\n=== unimodality: all w2/w1 above should be ~2.5-3.7 per Fig. 2 ===\n\n');
end

% ---------------------------------------------------------------------------
function omega = initial_omega(nelx, nely, bc, penal)
    cfg = struct('nelx',nelx,'nely',nely,'support_type',bc,'penal',penal, ...
                 'outer_max_iter',0,'verbose',false);
    omega = eval_initial(cfg);
end

function omega = initial_omega_corner(nelx, nely, penal)
    nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
    nl = nodeNrs(1,1);  nr = nodeNrs(1,end);      % bottom corners
    fixed = [2*nl-1; 2*nl; 2*nr-1; 2*nr];
    cfg = struct('nelx',nelx,'nely',nely,'fixed_dofs',fixed,'penal',penal, ...
                 'outer_max_iter',0,'verbose',false);
    omega = eval_initial(cfg);
end

function omega = eval_initial(cfg)
% Zero-optimization evaluation: one assembly + eigensolve at uniform rho.
    c = cfg;
    c.outer_max_iter = 1;
    c.move    = 0;            % Delta_rho is forced to zero -> design unchanged
    c.verbose = false;
    [~, h]    = topopt_freq_exact(c);
    omega     = h.omega(1,:)';
end
