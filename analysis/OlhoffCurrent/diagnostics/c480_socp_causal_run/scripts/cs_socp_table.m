function T = cs_socp_table(recs)
%CS_SOCP_TABLE  Flatten the per-iteration SOCP records into one scalar table.
names = {'outer','accepted','acceptedAttempt','exitflag1','iters1','tSolve1','tCert1', ...
    'exitflag2','iters2','tSolve2','tCert2','move','lam1','lam2','lamJ','dOff2','lamref','bs', ...
    'beta','predGain','dlam1','dlam2','fJJdrho','predLam1','predLam2','predLamJ','maxAbsDrho', ...
    'norm2Drho','sumDrho','nLowerDensity','nLowerMove','nUpperMove','nUpperDensity','nInterior', ...
    'nGray','nGrayFullMove','primalResidual','gap','dualBound','bestGap','rowComp','boxComp', ...
    'bsStat','statRms','statMax','mu','nu1','nu2','sRow0','coneNorm','separationPred','apex', ...
    'rawBoxViolation','gapSolverDuals','gapComplementarySlackness','gapDualboundGeneral', ...
    'gapDualboundAligned','candidateIndex','tEligibility','tAssembly','tSolve','tCertificate','tTotal', ...
    'cross_d2','cross_dinf','cross_dbs','cross_sign','cross_bound','cross_tSolve', ...
    'cross_nDiffGt0p1Move','cross_certified','cross_bestGap','cross_iterations','cross_tCertificate', ...
    'eqZero_cone_err','eqAcc_cone_err','eqAcc_grad_relerr','eqAcc_redundancy','eqAcc_row_nextmode_err', ...
    'eqAcc_row_volume_err'};
idx = find(~cellfun(@isempty, recs));
M = nan(numel(idx), numel(names));
cand = {'solverDuals','complementarySlackness','dualboundGeneral','dualboundAligned'};
for i = 1:numel(idx)
    r = recs{idx(i)};
    v = nan(1, numel(names));
    v(1) = r.outer; v(2) = r.accepted;
    if isfield(r,'attempts')
        for a = 1:numel(r.attempts)
            t = r.attempts(a); o = (a-1)*4;
            v(4+o) = t.exitflag; v(5+o) = t.iterations; v(6+o) = t.tSolve; v(7+o) = t.tCertificate;
        end
    end
    v(12) = r.move; v(13) = r.lam(1); v(14) = r.lam(end); v(15) = r.lamJ;
    if numel(r.dOff) >= 2, v(16) = r.dOff(2); end
    if r.accepted
        v(3) = r.acceptedAttempt;
        f = {'lamref','bs','beta','predGain'}; for j = 1:numel(f), v(strcmp(names,f{j})) = r.(f{j}); end
        v(strcmp(names,'dlam1')) = r.dlamPred(1); v(strcmp(names,'dlam2')) = r.dlamPred(2);
        v(strcmp(names,'fJJdrho')) = r.fJJdrho;
        v(strcmp(names,'predLam1')) = r.predLam(1); v(strcmp(names,'predLam2')) = r.predLam(2);
        v(strcmp(names,'predLamJ')) = r.predLamJ;
        f = {'maxAbsDrho','norm2Drho','sumDrho','nLowerDensity','nLowerMove','nUpperMove', ...
             'nUpperDensity','nInterior','nGray','nGrayFullMove','primalResidual','gap','dualBound', ...
             'bestGap','rowComp','boxComp','bsStat','statRms','statMax','mu','sRow0','coneNorm', ...
             'separationPred','apex','rawBoxViolation','tSolve','tCertificate'};
        for j = 1:numel(f), v(strcmp(names,f{j})) = double(r.(f{j})); end
        v(strcmp(names,'nu1')) = r.nu(1); v(strcmp(names,'nu2')) = r.nu(2);
        for j = 1:numel(r.candidateNames)
            k = find(strcmp(cand, r.candidateNames{j}));
            v(strcmp(names, ['gap' upper(cand{k}(1)) cand{k}(2:end)])) = r.candidateGaps(j);
        end
        v(strcmp(names,'candidateIndex')) = find(strcmp(cand, r.candidate));
        if ~isempty(r.cross) && isfield(r.cross,'d2')
            v(strcmp(names,'cross_d2')) = r.cross.d2; v(strcmp(names,'cross_dinf')) = r.cross.dinf;
            v(strcmp(names,'cross_dbs')) = r.cross.dbs; v(strcmp(names,'cross_sign')) = r.cross.signAgreement;
            v(strcmp(names,'cross_bound')) = r.cross.boundAgreement;
            v(strcmp(names,'cross_nDiffGt0p1Move')) = r.cross.nDiffGt0p1Move;
            v(strcmp(names,'cross_certified')) = r.cross.certified;
            v(strcmp(names,'cross_bestGap')) = r.cross.bestGap;
            v(strcmp(names,'cross_iterations')) = r.cross.iterations;
            v(strcmp(names,'cross_tCertificate')) = r.cross.tCertificate;
        end
        if ~isempty(r.cross), v(strcmp(names,'cross_tSolve')) = r.cross.tSolve; end
        v(strcmp(names,'eqAcc_cone_err')) = r.equivAccepted.cone_err;
        v(strcmp(names,'eqAcc_grad_relerr')) = r.equivAccepted.grad_relerr;
        v(strcmp(names,'eqAcc_redundancy')) = r.equivAccepted.redundancy;
        v(strcmp(names,'eqAcc_row_nextmode_err')) = r.equivAccepted.row_nextmode_err;
        v(strcmp(names,'eqAcc_row_volume_err')) = r.equivAccepted.row_volume_err;
    end
    if isfield(r,'equivZero'), v(strcmp(names,'eqZero_cone_err')) = r.equivZero.cone_err; end
    f = {'tEligibility','tAssembly','tTotal'};
    for j = 1:numel(f), if isfield(r, f{j}), v(strcmp(names,f{j})) = r.(f{j}); end, end
    M(i,:) = v;
end
T = array2table(M, 'VariableNames', names);
end
