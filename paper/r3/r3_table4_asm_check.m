function r3_table4_asm_check(label)
%R3_TABLE4_ASM_CHECK  One timed run of the proposed method with the matched settings of
%   Table 4 at 800x100 (after a 160x20 warm-up), with whichever assemble_KM is first on the
%   path (the caller puts the old 64-triplet version in front for the "old" run).
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
if nargin >= 1 && strcmp(label, 'old')
    addpath(getenv('OLDASM'), '-begin');
end
maxNumCompThreads(1);
fprintf('assemble_KM used: %s\n', which('assemble_KM'));
matched = struct('filterRadius', 0.06, 'move', 0.1, 'stiffness', 'pedersen', 'Emin', 1e-5*1e7, 'rhoMin', 1e-3);
proposed_freq(problem_ss_beam(160, 20), matched);            % warm-up
prob = problem_ss_beam(800, 100);
t0 = tic;  P = proposed_freq(prob, matched);
o = structural_modes(prob, P.x, 1);  tw = toc(t0);
fprintf('%s assembly, proposed matched 800x100: solver total %.2f s (ref %.2f, loop %.2f), with final modal %.2f s, %d it, omega1 %.2f\n', ...
    label, P.time.total, P.time.reference, P.time.loop, tw, P.iterations, o);
end
