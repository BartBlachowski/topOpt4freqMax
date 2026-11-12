function run_all(quick)
%RUN_ALL  Reproduce every table and figure of the paper.
%
%   run_all(true)    quick check of every script on coarse meshes (about 25 min)
%   run_all          full reproduction (many hours; the Du-Olhoff runs of Tables 2 and 4
%                    dominate).  Results go to results/<name>/.
%
%   Each script can also be run on its own; see README.md for the map
%   paper item -> script -> run time.
if nargin < 1, quick = false; end
here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(here), 'setup_paths.m'));
addpath(fullfile(here, 'checks'));
steps = {
    'Figures 1-4 (method steps)',             @() fig1_4_method_steps()
    'Section 2.3 (spectral check, m_v)',      @() sec2_3_spectral_check()
    'Table 1 (gradient ablation)',            @() table1_gradient_ablation(quick)
    'Table 2, Figures 5-8 (benchmark)',       @() table2_fig8_benchmark(quick)
    'Tables 3-4 (matched settings)',          @() tables3_4_matched_settings(quick)
    'Figure 9 (frequency histories)',         @() fig9_frequency_histories()
    'Tables 5-8, Figures 10-11 (alpha sweep)', @() tables5_8_fig10_11_alpha_sweep(quick)
    'Table 9, Figures 12-13 (direct)',        @() table9_fig12_13_direct_comparison(quick)
    'Check: mode selection',                  @() check_mode_selection(quick)
    'Check: sign invariance',                 @() check_sign_invariance()
    'Check: full derivative (FD)',            @() check_full_derivative_fd()
    'Check: Du-Olhoff reference',             @() check_du_olhoff_reference()};
if ~quick
    steps(end+1, :) = {'Section 4.1 (tolerance 1e-3)', @() sec4_1_tolerance_check()};
    steps(end+1, :) = {'Section 4.3 (reference mode)', @() sec4_3_reference_mode_study()};
end
for k = 1:size(steps, 1)
    fprintf('\n===== %s =====\n', steps{k,1});
    t = tic;  steps{k,2}();  fprintf('(%.0f s)\n', toc(t));
end
end
