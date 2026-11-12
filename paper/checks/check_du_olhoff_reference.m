function check_du_olhoff_reference()
%CHECK_DU_OLHOFF_REFERENCE  Section 3.1: the reference implementation against the
%   simply supported beam of Du & Olhoff (2007), Fig. 3a: uniform start 68.7 rad/s,
%   bimodal optimum omega1 = omega2 = 174.7 rad/s (mesh and settings not published),
%   and a finite-difference check of a simple-eigenvalue sensitivity.  About 3 min.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
prob = problem_ss_beam(160, 20);
D = du_olhoff(prob, struct('maxOuter', 1));
fprintf('uniform start x = 0.5, 160x20: omega1 = %.2f rad/s (Du & Olhoff: 68.7)\n', D.hist.omega(1,1));
D = du_olhoff(prob, struct('maxOuter', 400, 'stopRule', 'native'));
fprintf('optimized 160x20 (native stop): omega1/omega2/omega3 = %.2f / %.2f / %.2f rad/s, %d outer iterations (Du & Olhoff: 174.7 / 174.7 / 284.9)\n', ...
    D.omega(1:3), D.nOuter);
end
