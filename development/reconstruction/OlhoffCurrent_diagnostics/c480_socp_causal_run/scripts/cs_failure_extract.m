function F = cs_failure_extract()
%CS_FAILURE_EXTRACT  READ-ONLY: pull the rejected iteration's full SOCP record
%   (attempts, certificate candidates, bars) out of the saved treatment evidence.
S = cs_setup();
T = load(fullfile(S.evDir,'C480x60_socp_trajectory.mat'),'socp','hist','log','status','termination');
k = find(~cellfun(@isempty, T.socp), 1, 'last');
r = T.socp{k};
F = struct('outer', r.outer, 'accepted', r.accepted, 'termination', r.termination, 'reason', r.reason, ...
    'move', r.move, 'lam', r.lam, 'lamJ', r.lamJ, 'dOff', r.dOff, 'lamref', r.lamref, ...
    'omega', sqrt(r.lam), 'relative_lambda_gap', (r.lam(2)-r.lam(1))/r.lam(1), ...
    'eligibility', r.eligibility, 'equivZero', r.equivZero, 'tAssembly', r.tAssembly);
for a = 1:numel(r.attempts)
    t = r.attempts(a); c = t.cert;
    A = struct('solver', t.solver, 'exitflag', t.exitflag, 'iterations', t.iterations, 'message', t.message, ...
        'tSolve', t.tSolve, 'tCertificate', t.tCertificate, 'certified', t.certified, 'reason', t.reason, ...
        'apex', c.apex, 'coneNorm', c.coneNorm, 'predicted_separation_e2_minus_e1', 2*c.coneNorm*r.lamref, ...
        'maxRow', c.maxRow, 'fval', c.fval, 'rawBoxViolation', c.rawBoxViolation, 'bs', c.bs, ...
        'predicted_gain_beta_minus_lam1', (c.bs-1)*r.lamref, 'sRow0', c.sRow0, ...
        'bestGap', c.bestGap, 'bestGapCandidate', c.bestGapCandidate, 'candidates', c.candidates);
    F.attempts(a) = A;
end
% previous accepted iterations: separation trajectory approaching the apex
sep = nan(1, k-1); gap = nan(1, k-1); cand = cell(1, k-1);
for j = 1:k-1
    q = T.socp{j};
    sep(j) = q.separationPred; gap(j) = q.gap; cand{j} = q.candidate;
end
F.previous_predicted_separation = sep;
F.previous_relative_predicted_separation = sep ./ cellfun(@(q) q.lamref, T.socp(1:k-1));
F.previous_gap = gap; F.previous_candidate = cand;
F.hist_gap12 = T.hist.gap12; F.hist_omega = T.hist.omega;
F.log = T.log;
cs_json(fullfile(S.study,'evaluations','termination_record.json'), F);
fprintf('outer %d: %s\n', F.outer, F.reason);
for a = 1:numel(F.attempts)
    A = F.attempts(a);
    fprintf(' %s ef=%d it=%d apex=%d ||s||=%.3e sep=%.3e maxRow=%.2e bs-1=%.4e bestGap=%.3e\n', A.solver, A.exitflag, ...
        A.iterations, A.apex, A.coneNorm, A.predicted_separation_e2_minus_e1, A.maxRow, A.bs-1, A.bestGap);
    for c = A.candidates
        fprintf('    %-18s gap=%.3e box=%.3e row=%.2e bs=%.2e rms=%.2e fails=%s\n', c.name, c.gap, c.boxComp, c.rowComp, c.bsStat, c.statRms, c.fails);
    end
end
disp(F.previous_relative_predicted_separation);
disp(F.hist_gap12);
end
