function out = tr_tests()
%TR_TESTS  Phase 15 software validation, under this task's ZERO-RUN lock.
%
%   Runs the repository suite EXCEPT test_preset_equivalence, which performs a
%   160x20 optimization and is therefore refused here (see SKIPPED below), plus
%   the controller-mechanics suite from the completed validation retry, whose
%   only end-to-end solves are 48x6 (NE = 288) -- software mechanics, far below
%   the 160x20 scientific floor, never interpreted.
%
%   No scientific optimization is executed by this function.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
r1    = fullfile(root,'diagnostics','three_rung_promotion_validation_retry1');

RUN  = {'test_currentness','test_source_integrity','test_path_isolation', ...
        'test_evidence_retention','test_finalization_gate'};
SKIP = {'test_preset_equivalence'};
SKIP_REASON = ['runs a 160x20 optimization (production preset vs frozen ' ...
    'conference fixture); this task authorizes ZERO scientific optimization runs'];

out = struct('run',{{}},'nFail',[],'skipped',{SKIP},'skipReason',SKIP_REASON);
tot = 0; rows = {};
for i = 1:numel(RUN)
    addpath(root); addpath(fullfile(root,'tests'));
    try
        nf = feval(RUN{i});
    catch ME
        fprintf('\n!! %s ERRORED: %s\n', RUN{i}, ME.message); nf = -1;
    end
    rows{end+1} = sprintf('%-28s nFail=%d', RUN{i}, nf); %#ok<AGROW>
    out.run{end+1} = RUN{i}; out.nFail(end+1) = nf; %#ok<AGROW>
    tot = tot + max(nf,0);
end

% ---- controller mechanics, reused from the validation retry -------------
addpath(root); addpath(fullfile(r1,'scripts')); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
guard = olhoffcurrent_paths(); %#ok<NASGU>
mech = [];
try
    mech = tr_tests_retry_shim(r1);
catch ME
    fprintf('\n!! controller mechanics ERRORED: %s\n', ME.message);
end
out.mechanics = mech;

fprintf('\n\n===== PHASE 15 SUMMARY =====\n');
for i=1:numel(rows), fprintf('  %s\n', rows{i}); end
fprintf('  %-28s SKIPPED -- %s\n', SKIP{1}, SKIP_REASON);
if ~isempty(mech)
    fprintf('  %-28s %d/%d passed  (%s)\n', 'controller mechanics', ...
        mech.nPass, mech.nTests, mech.verdict);
    tot = tot + (mech.nTests - mech.nPass);
end
fprintf('  TOTAL FAILURES = %d\n', tot);
out.totalFailures = tot;

fid = fopen(fullfile(study,'evidence','software_tests.json'),'w');
c = onCleanup(@() fclose(fid)); fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));
end

function m = tr_tests_retry_shim(~)
% The retry's controller-mechanics suite, copied verbatim under a distinct
% function name (scripts/tr_tests_retry.m) so it can be invoked alongside this
% study's own tr_tests without a name collision.  Its content is unmodified:
% same 29 assertions, same 48x6 software-only solves.
m = tr_tests_retry();
end
