function nFail = run_all_tests()
%RUN_ALL_TESTS  The architecture-level suite.
root = '/Users/piotrek/Programming/Matlab/Olhoff';
cd(root); setpaths(); addpath(fullfile(root,'architecture','tests'));
maxNumCompThreads(1);
suites = {'test_config','test_mass','test_modules','test_continuation', ...
           'test_legacy_roundtrip','test_presets_match_history'};
nFail = 0;  summary = {};
for k = 1:numel(suites)
    fprintf('\n===== %s =====\n', suites{k});
    n = feval(suites{k});
    summary(end+1,:) = {suites{k}, n}; %#ok<AGROW>
    nFail = nFail + n;
end
fprintf('\n================ SUMMARY ================\n');
for k = 1:size(summary,1)
    fprintf('  %-32s %s\n', summary{k,1}, local_v(summary{k,2}));
end
fprintf('  %-32s %d\n', 'TOTAL FAILURES', nFail);
end
function s = local_v(n)
if n==0, s = 'PASS'; else, s = sprintf('%d FAILURE(S)', n); end
end
