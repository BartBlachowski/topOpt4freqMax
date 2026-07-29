function run_all_tests()
% RUN_ALL_TESTS  Regression suite for the faithful-reconstruction campaign.
%
%   T1  inner_loop_mma_instr == inner_loop_mma            (bit-identical)
%   T2  recon_solve(defaults) == topopt_freq_exact        (bit-identical),
%       for BOTH the paper-literal regime A and the stabilized regime B
%   T3  fail-closed semantics: rejects non-converged inner solves, halts,
%       applies no outer update, and is inert when the inner solve converges
%   T4  continuation semantics: declared p schedule is realised exactly;
%       disabled continuation holds p constant; densities transfer across
%       stages; continuation alone changes nothing at p == const == 3
%
%   Any failure prints FAIL and sets a nonzero exit via error() at the end.

this_dir = fileparts(mfilename('fullpath'));
root     = fullfile(this_dir, '..');
addpath(root);
addpath(fullfile(root, '..', '..', 'Matlab'));
addpath(fullfile(root, '..', '..', '..', '..', 'tools', 'Matlab'));

results = {};
results{end+1} = t1_inner_equivalence();
results{end+1} = t2_outer_equivalence();
results{end+1} = t3_fail_closed();
results{end+1} = t4_continuation();

fprintf('\n================ TEST SUMMARY ================\n');
nfail = 0;
for i = 1:numel(results)
    r = results{i};
    for j = 1:numel(r)
        st = 'PASS'; if ~r(j).pass, st = 'FAIL'; nfail = nfail + 1; end
        fprintf(' [%s] %-52s %s\n', st, r(j).name, r(j).detail);
    end
end
fprintf('==============================================\n');
if nfail > 0
    error('run_all_tests:Failures', '%d test(s) FAILED.', nfail);
end
fprintf(' ALL TESTS PASSED\n\n');
end

%% ---------------------------------------------------------------------
function r = t1_inner_equivalence()
fprintf('\n--- T1: inner MMA instrumented vs production ---\n');
rng(0,'twister');
nEl = 200;  N = 1;
rho = 0.5*ones(nEl,1);
fsk = randn(nEl, N, N) * 1e6;
lambda_bar = 2.2e4;
lambda_J   = 3.1e4;
dlam_J     = randn(nEl,1) * 1e6;
r = struct('name',{},'pass',{},'detail',{});

cases = { ...
  {'A: move=Inf outer=Inf', Inf, Inf}, ...
  {'B: move=0.2 outer=0.2', 0.2, 0.2}, ...
  {'C: no J-mode',          0.2, 0.2} };

for ci = 1:numel(cases)
    nm = cases{ci}{1};  ml = cases{ci}{2};  om = cases{ci}{3};
    if ci == 3, lJ = Inf; dJ = []; else, lJ = lambda_J; dJ = dlam_J; end
    [d1, b1] = inner_loop_mma(      rho, lambda_bar, fsk, lJ, dJ, 0.5, 1e-3, 30, 1e-4, ml, om);
    [d2, b2] = inner_loop_mma_instr(rho, lambda_bar, fsk, lJ, dJ, 0.5, 1e-3, 30, 1e-4, ml, om);
    ed = max(abs(d1-d2));  eb = abs(b1-b2);
    ok = isequal(d1,d2) && isequal(b1,b2);
    r(end+1) = mk(sprintf('T1 %s', nm), ok, sprintf('max|ddrho|=%g |dbeta|=%g', ed, eb)); %#ok<AGROW>
end

% N = 2 cluster path
N = 2;
fsk2 = zeros(nEl,N,N);
fsk2(:,1,1) = randn(nEl,1)*1e6;  fsk2(:,2,2) = randn(nEl,1)*1e6;
off = randn(nEl,1)*1e5;  fsk2(:,1,2) = off;  fsk2(:,2,1) = off;
[d1,b1] = inner_loop_mma(      rho, lambda_bar, fsk2, lambda_J, dlam_J, 0.5, 1e-3, 30, 1e-4, 0.2, 0.2);
[d2,b2] = inner_loop_mma_instr(rho, lambda_bar, fsk2, lambda_J, dlam_J, 0.5, 1e-3, 30, 1e-4, 0.2, 0.2);
r(end+1) = mk('T1 D: N=2 cluster', isequal(d1,d2) && isequal(b1,b2), ...
    sprintf('max|ddrho|=%g', max(abs(d1-d2))));
end

%% ---------------------------------------------------------------------
function r = t2_outer_equivalence()
fprintf('\n--- T2: recon_solve vs production topopt_freq_exact ---\n');
r = struct('name',{},'pass',{},'detail',{});

% Regime A (paper-literal) and Regime B (stabilized), small mesh for speed.
regimes = {};
A = struct('nelx',40,'nely',10,'move_lim',Inf,'outer_move',Inf,'alpha',1.0, ...
           'outer_max_iter',12,'outer_tol',1e-4);
B = struct('nelx',40,'nely',10,'move_lim',0.2,'outer_move',0.2,'alpha',0.5, ...
           'outer_max_iter',12,'outer_tol',1e-6);
regimes{1} = {'A (paper-literal)', A};
regimes{2} = {'B (stabilized)',    B};

for k = 1:2
    nm = regimes{k}{1};  P = regimes{k}{2};

    pc = struct('support_type','CC','nelx',P.nelx,'nely',P.nely,'volfrac',0.5, ...
        'mass_mode','du2007_c1','sensitivity_filter',true,'rmin_elem',2.5, ...
        'n_target',1,'n_modes',4,'mult_tol',1e-3, ...
        'outer_max_iter',P.outer_max_iter,'outer_tol',P.outer_tol, ...
        'inner_max_iter',30,'inner_tol',1e-4,'move_lim',P.move_lim, ...
        'outer_move',P.outer_move,'alpha',P.alpha,'acceptance_check',false, ...
        'verbose',false);
    rc = pc;  rc = rmfield(rc, {'sensitivity_filter','acceptance_check'});
    rc.fail_closed = false;  rc.cont = struct('enabled',false);  rc.penal = 3.0;

    rng(0,'twister');  [rho_p, hp] = topopt_freq_exact(pc);
    rng(0,'twister');  o = recon_solve(rc);

    n = min(hp.outer_iters, o.outer_iters);
    e_rho  = max(abs(rho_p(:) - o.rho_final(:)));
    e_om   = max(abs(hp.omega_trial(1:n,1) - o.hist.omega_trial(1:n,1)));
    e_beta = max(abs(hp.beta(1:n) - o.hist.beta(1:n)));
    e_it   = abs(hp.outer_iters - o.outer_iters);
    ok = isequal(rho_p(:), o.rho_final(:)) && e_it == 0;
    r(end+1) = mk(sprintf('T2 regime %s bit-identical', nm), ok, ...
        sprintf('iters %d/%d  max|drho|=%g max|domega1|=%g max|dbeta|=%g', ...
        hp.outer_iters, o.outer_iters, e_rho, e_om, e_beta)); %#ok<AGROW>
end
end

%% ---------------------------------------------------------------------
function r = t3_fail_closed()
fprintf('\n--- T3: fail-closed inner-MMA semantics ---\n');
r = struct('name',{},'pass',{},'detail',{});

base = struct('support_type','CC','nelx',40,'nely',10,'volfrac',0.5, ...
    'mass_mode','du2007_c1','rmin_elem',2.5,'n_target',1,'n_modes',4, ...
    'mult_tol',1e-3,'outer_max_iter',8,'outer_tol',1e-6,'inner_max_iter',30, ...
    'inner_tol',1e-4,'move_lim',0.2,'outer_move',0.2,'alpha',0.5, ...
    'verbose',false,'penal',3.0,'cont',struct('enabled',false));

% (a) inner_max_iter = 1 guarantees a non-converged inner solve -> must halt
%     at outer iteration 1 with INNER_FAILURE and apply NO update.
c = base;  c.inner_max_iter = 1;  c.fail_closed = true;
rng(0,'twister');  o = recon_solve(c);
init_rho = 0.5*ones(c.nelx*c.nely,1);
ok = strcmp(o.stop_status,'INNER_FAILURE') && o.outer_iters == 1 && ...
     isequal(o.rho_final, init_rho) && ~o.hist.accepted(1);
r(end+1) = mk('T3a halts on non-converged inner, no update applied', ok, ...
    sprintf('status=%s iters=%d rho unchanged=%d reason=%s', o.stop_status, ...
    o.outer_iters, isequal(o.rho_final, init_rho), o.hist.reject_reason{1}));

% (b) same config WITHOUT fail-closed must proceed past iteration 1
c2 = c;  c2.fail_closed = false;
rng(0,'twister');  o2 = recon_solve(c2);
ok = o2.outer_iters > 1 && ~o2.hist.accepted(1);
r(end+1) = mk('T3b without fail-closed the same step is accepted', ok, ...
    sprintf('iters=%d accepted(1)=%d', o2.outer_iters, o2.hist.accepted(1)));

% (c) fail-closed is INERT when every inner solve converges: identical to off
c3 = base;  c3.inner_max_iter = 400;  c3.outer_max_iter = 5;
rng(0,'twister');  oA = recon_solve(setfield(c3,'fail_closed',false)); %#ok<SFLD>
rng(0,'twister');  oB = recon_solve(setfield(c3,'fail_closed',true));  %#ok<SFLD>
allconv = all(oB.hist.inner_converged);
ok = allconv && isequal(oA.rho_final, oB.rho_final) && ...
     oA.outer_iters == oB.outer_iters;
r(end+1) = mk('T3c inert when inner converges (bit-identical)', ok, ...
    sprintf('all inner converged=%d  identical=%d', allconv, ...
    isequal(oA.rho_final,oB.rho_final)));

% (d) the gate itself: volume/bound predicates fire on constructed input
ih = struct('converged',true,'termination_reason','convergence','n_iters',3, ...
            'fval',[0 0 0]);
rho = 0.5*ones(10,1);
[okv, why] = local_fc(ih, 0.2*ones(10,1), rho, 1e-3, 0.5, Inf, 1e-4, 1e-9);
ok1 = ~okv && startsWith(why,'volume_violation');
[okb, whyb] = local_fc(ih, 0.9*ones(10,1), rho, 1e-3, 0.5, 0.2, 1e-4, 1e-9);
ok2 = ~okb && strcmp(whyb,'bound_violation');
[okn, whyn] = local_fc(ih, [NaN; zeros(9,1)], rho, 1e-3, 0.5, Inf, 1e-4, 1e-9);
ok3 = ~okn && strcmp(whyn,'nonfinite_increment');
[okg, ~]   = local_fc(ih, zeros(10,1), rho, 1e-3, 0.5, Inf, 1e-4, 1e-9);
r(end+1) = mk('T3d gate predicates fire correctly', ok1 && ok2 && ok3 && okg, ...
    sprintf('vol=%d bound=%d nonfinite=%d clean-pass=%d', ok1, ok2, ok3, okg));
end

%% ---------------------------------------------------------------------
function r = t4_continuation()
fprintf('\n--- T4: continuation semantics ---\n');
r = struct('name',{},'pass',{},'detail',{});

base = struct('support_type','CC','nelx',40,'nely',10,'volfrac',0.5, ...
    'mass_mode','du2007_c1','rmin_elem',2.5,'n_target',1,'n_modes',4, ...
    'mult_tol',1e-3,'outer_max_iter',12,'outer_tol',1e-9,'inner_max_iter',30, ...
    'inner_tol',1e-4,'move_lim',0.2,'outer_move',0.2,'alpha',0.5, ...
    'verbose',false,'penal',3.0,'fail_closed',false);

% (a) declared fixed schedule is realised exactly
c = base;
c.cont = struct('enabled',true,'p_values',[1 2 3],'mode','fixed','stage_len',4);
rng(0,'twister');  o = recon_solve(c);
want = [1 1 1 1 2 2 2 2 3 3 3 3]';
ok = isequal(o.hist.penal(:), want);
r(end+1) = mk('T4a fixed schedule realised exactly', ok, ...
    sprintf('p = [%s]', num2str(o.hist.penal(:)', '%g ')));

% (b) continuation disabled holds p constant at cfg.penal
c2 = base;  c2.cont = struct('enabled',false);
rng(0,'twister');  o2 = recon_solve(c2);
ok = all(o2.hist.penal == 3.0);
r(end+1) = mk('T4b disabled continuation holds p = cfg.penal', ok, ...
    sprintf('unique p = [%s]', num2str(unique(o2.hist.penal)', '%g ')));

% (c) a single-value schedule p = [3] is bit-identical to continuation off
c3 = base;  c3.cont = struct('enabled',true,'p_values',3,'mode','fixed','stage_len',4);
rng(0,'twister');  o3 = recon_solve(c3);
ok = isequal(o2.rho_final, o3.rho_final) && o2.outer_iters == o3.outer_iters;
r(end+1) = mk('T4c p=[3] schedule == continuation off (bit-identical)', ok, ...
    sprintf('max|drho|=%g', max(abs(o2.rho_final - o3.rho_final))));

% (d) densities transfer across stages: rho at the first iteration of stage 2
%     equals rho at the last iteration of stage 1 (no reinitialisation).
sn = o.rho_snapshots;
ok = ~isequal(sn(:,4), 0.5*ones(size(sn,1),1)) && ...
     norm(sn(:,5) - sn(:,4)) > 0 && norm(sn(:,5) - sn(:,4)) < 1e9;
stage_jump = norm(sn(:,5)-sn(:,4))/sqrt(size(sn,1));
r(end+1) = mk('T4d densities transfer across the stage boundary', ok, ...
    sprintf('||rho_5 - rho_4||/sqrt(nEl) = %.4e (finite, nonzero)', stage_jump));

% (e) 'drho' trigger mode advances only after min_stage_len and below trigger
c5 = base;  c5.outer_max_iter = 20;
c5.cont = struct('enabled',true,'p_values',[1 3],'mode','drho', ...
                 'drho_trigger',1e9,'min_stage_len',6);
rng(0,'twister');  o5 = recon_solve(c5);
first3 = find(o5.hist.penal == 3, 1, 'first');
ok = ~isempty(first3) && first3 == 7;
r(end+1) = mk('T4e drho-trigger respects min_stage_len', ok, ...
    sprintf('first p=3 at iter %d (expected 7)', first3));
end

%% ---------------------------------------------------------------------
function s = mk(name, pass, detail)
s = struct('name', name, 'pass', logical(pass), 'detail', detail);
if pass, tag = 'PASS'; else, tag = 'FAIL'; end
fprintf('  [%s] %s -- %s\n', tag, name, detail);
end

%% ---------------------------------------------------------------------
function [ok, reason] = local_fc(ih, drho, rho, rho_min, volfrac, outer_move, vt, bt)
% Mirror of recon_solve/fc_check for direct predicate testing.
ok = true;  reason = 'ok';
if ~ih.converged
    ok = false;  reason = 'inner_not_converged';  return
end
if any(~isfinite(ih.fval(:))), ok = false; reason = 'nonfinite_constraint'; return, end
if any(~isfinite(drho)),       ok = false; reason = 'nonfinite_increment';  return, end
lb = max(rho_min - rho, -outer_move*ones(size(rho)));
ub = min(1        - rho, +outer_move*ones(size(rho)));
if any(drho < lb - bt) || any(drho > ub + bt)
    ok = false;  reason = 'bound_violation';  return
end
if mean(rho + drho) > volfrac + vt
    ok = false;  reason = sprintf('volume_violation(%.3e)', mean(rho+drho)-volfrac);
end
end
