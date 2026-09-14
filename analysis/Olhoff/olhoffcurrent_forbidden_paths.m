function [repoRel, absolute] = olhoffcurrent_forbidden_paths()
%OLHOFFCURRENT_FORBIDDEN_PATHS  Olhoff trees that must NEVER execute in production.
%
%   [repoRel, absolute] = OLHOFFCURRENT_FORBIDDEN_PATHS()
%
%     repoRel   repository-relative prefixes, matched against resolved files
%               and against MATLAB path entries
%     absolute  absolute prefixes outside this repository
%
%   analysis/Olhoff is the ONLY Olhoff implementation production may execute.
%   Everything below is historical evidence, experimental code, audit material
%   or development upstream.  Since the 2026-09-14 repository cleanup every
%   historical tree lives under development/, which is forbidden as a whole;
%   the pre-cleanup locations stay listed so a tree re-created there still
%   fails closed.  Classification: development/repository_cleanup/
%   MIGRATION_MANIFEST.tsv (the 2026-09-07 map is archived under
%   development/migration_history/olhoff_current_promotion/).
%
%   THIS BLACKLIST IS NOT THE WHOLE GATE.  It is the part that names what we
%   already know about.  olhoffcurrent_assert_dispatch ALSO refuses any second
%   candidate for an owned symbol from anywhere outside this tree, so a NEW
%   Olhoff implementation nobody has added here still fails closed.
%
%   See also OLHOFFCURRENT_ASSERT_DISPATCH, OLHOFFCURRENT_PATHS.

repoRel = { ...
    fullfile('development'), ...                        % the archive: every historical tree since 2026-09-14
    fullfile('analysis', 'OlhoffM4Reconstruction'), ...   % pre-2026-09-14 locations, kept as guards
    fullfile('analysis', 'OlhoffExperiments'), ...        % future experimental tree
    fullfile('analysis', 'OlhoffApproach'), ...
    fullfile('analysis', 'OlhoffApproachExact'), ...
    fullfile('analysis', 'OlhoffApproachExactOpus'), ...
    fullfile('analysis', 'OlhoffRegularized'), ...
    fullfile('analysis', 'OlhoffReproduced2007'), ...
    fullfile('analysis', 'olhoff_stabilization_audit'), ...
    fullfile('analysis', 'olhoff_fixed_budget_audit'), ...
    fullfile('analysis', 'olhoff_native_convergence'), ...
    fullfile('analysis', 'olhoff_nested_mma_route_audit'), ...
    fullfile('analysis', 'olhoff_practical_convergence_audit'), ...
    fullfile('Matlab',   'reproduction2007') };

% The development/research upstream.  Production must never execute from it:
% it is where experiments happen, it is not pinned by this repository's git
% history, and a run that reached it could not be reproduced from this
% repository alone.  See analysis/OlhoffCurrent/PROVENANCE.md.
absolute = { '/Users/piotrek/Programming/Matlab/Olhoff' };
end
