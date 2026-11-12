function check_sign_invariance()
%CHECK_SIGN_INVARIANCE  Section 4.2: the weighted sum of two load-case compliances,
%   Eq. (14), does not depend on the signs of the reference eigenvectors.  alpha = 0.5,
%   clamped beam and building; each eigenvector reversed in turn.  About 1 min.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
for prob = {problem_clamped_beam(), problem_building()}
    p = prob{1};
    A = proposed_freq(p, struct('alpha', 0.5));
    for s = [-1 1; 1 -1]'
        ref = struct('omega0', A.omega0, 'phi0', A.phi0 .* s');
        B = proposed_freq(p, struct('alpha', 0.5, 'reference', ref));
        fprintf('%-13s signs (%+d,%+d): max|x - x_(+,+)| = %g\n', p.name, s, max(abs(B.x - A.x)));
    end
end
end
