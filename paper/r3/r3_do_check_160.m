%R3_DO_CHECK_160  Du-Olhoff at 160x20 with the current code (lower-triangle assemble_KM
%   is used by the common evaluator): iterations and omega_1 against Table 2 (64+1360, 168.70).
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
maxNumCompThreads(1);
prob = problem_ss_beam(160, 20);
D = du_olhoff(prob);  o = structural_modes(prob, D.x, 2);
fprintf('Du-Olhoff 160x20: %d outer + %d inner, total %.2f s, omega1 = %.4f, omega2 = %.4f rad/s (Table 2: 64+1360, 168.70)\n', ...
    D.nOuter, D.nInner, D.time.total, o(1), o(2));
