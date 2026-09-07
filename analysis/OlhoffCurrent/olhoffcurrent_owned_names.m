function names = olhoffcurrent_owned_names()
%OLHOFFCURRENT_OWNED_NAMES  Every bare function name this implementation owns.
%
%   DERIVED FROM THE DIRECTORY, never restated as a literal list, so a file
%   added to +impl/ cannot slip past the dispatch gate unchecked.  This is the
%   "derive the definitive symbol set from the actual canonical source"
%   requirement of the promotion brief.
%
%   Package functions under architecture/+olh/ are deliberately EXCLUDED: they
%   are addressed as olh.config.*, olh.presets.* and cannot be shadowed by a
%   bare .m file on the path.  What can be shadowed is exactly what is listed
%   here -- and it includes every helper, not merely the entry point:
%   olhoffOpt, olhoffSolve, innerLoop, genGrad, deltaLambda, massScale,
%   moveControl, multRule, mmasub, subsolv, prepFilter, applyFilter, eigSolve,
%   model2D, assemble2D, elemMats2D, classifyModes, useMMA, ...
%
%   See also OLHOFFCURRENT_ASSERT_DISPATCH.

dirs = olhoffcurrent_impl_dirs();
fn = fieldnames(dirs);
names = {};
for i = 1:numel(fn)
    listing = dir(fullfile(dirs.(fn{i}), '*.m'));
    for k = 1:numel(listing)
        [~, base] = fileparts(listing(k).name);
        names{end+1} = base; %#ok<AGROW>
    end
end
names = unique(names);
end
