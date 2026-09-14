function report = repro2007_assert_identity(verbose)
%REPRO2007_ASSERT_IDENTITY  Prove which implementation is about to execute.
%
%   REPRO2007_ASSERT_IDENTITY() errors unless every function owned by the
%   clean-room reproduction resolves to a file inside REPRO2007_ROOT.
%
%   report = REPRO2007_ASSERT_IDENTITY() additionally returns a struct:
%     .root          the implementation root that was asserted
%     .n_checked     number of owned functions checked
%     .shadowed      struct array of owned functions that ALSO exist elsewhere
%                    on the current path (name, ours, other, identical)
%     .ok            true (the function errors rather than returning false)
%
%   REPRO2007_ASSERT_IDENTITY(true) prints the report.
%
%   REQUIRES the implementation path to be installed first.  Called on its own
%   it will (correctly) error, because none of algo/, fem/, filter/, mma/ or
%   runs/ is on a default path:
%
%       guard = repro2007_paths();   %#ok<NASGU>
%       repro2007_assert_identity(true);
%
%   RUN_REPRO2007 does this for you and returns the report in
%   info.path_identity.
%
%   This is the WP6 diagnostic: three implementations of the same algorithm
%   family live in this repository and share function names.  A benchmark run
%   must be able to state, on the record, which implementation root it
%   executed -- not assume it.
%
%   A name that resolves outside this root is a hard error.  A name that
%   resolves HERE but also exists elsewhere is reported, not fatal: that is
%   ordinary shadowing, and the report says whether the two files are
%   byte-identical so a reviewer can judge it.
%
%   See also REPRO2007_ROOT, REPRO2007_PATHS.

if nargin < 1 || isempty(verbose)
    verbose = false;
end

root = repro2007_root();
subdirs = {'algo', 'fem', 'filter', 'mma', 'runs', 'runner'};

names = {};
owners = {};
for i = 1:numel(subdirs)
    d = fullfile(root, subdirs{i});
    listing = dir(fullfile(d, '*.m'));
    for k = 1:numel(listing)
        [~, base] = fileparts(listing(k).name);
        names{end+1}  = base;                          %#ok<AGROW>
        owners{end+1} = fullfile(d, listing(k).name);  %#ok<AGROW>
    end
end

if isempty(names)
    error('repro2007_assert_identity:NoFunctions', ...
        'No .m files found under %s -- the implementation is missing.', root);
end

wrong = {};
shadowed = struct('name', {}, 'ours', {}, 'other', {}, 'identical', {});

for i = 1:numel(names)
    resolved = which(names{i});
    if isempty(resolved)
        wrong{end+1} = sprintf('%-24s -> NOT ON PATH (expected %s)', ...
            names{i}, owners{i});                                      %#ok<AGROW>
        continue
    end
    if ~localIsInside(resolved, root)
        wrong{end+1} = sprintf('%-24s -> %s\n%26s(expected inside %s)', ...
            names{i}, resolved, '', root);                             %#ok<AGROW>
        continue
    end

    % Resolves correctly.  Does the name exist anywhere else on the path?
    others = which(names{i}, '-all');
    for k = 2:numel(others)
        o = others{k};
        if localIsInside(o, root) || ~endsWith(o, '.m')
            continue    % another copy inside our own tree, or a built-in note
        end
        shadowed(end+1) = struct('name', names{i}, 'ours', resolved, ...
            'other', o, 'identical', localSameBytes(resolved, o));     %#ok<AGROW>
    end
end

if ~isempty(wrong)
    error('repro2007_assert_identity:WrongImplementation', ...
        ['MATLAB path resolves %d function(s) outside the clean-room ' ...
         'reproduction root.\nRefusing to run: the executing implementation ' ...
         'is not the one this runner claims.\nRoot: %s\n\n%s\n'], ...
        numel(wrong), root, strjoin(wrong, newline));
end

report = struct('root', root, 'n_checked', numel(names), ...
    'shadowed', shadowed, 'ok', true);

if verbose
    fprintf('[repro2007] implementation root : %s\n', root);
    fprintf('[repro2007] functions verified   : %d\n', numel(names));
    if isempty(shadowed)
        fprintf('[repro2007] name collisions      : none\n');
    else
        fprintf('[repro2007] name collisions      : %d (ours wins)\n', numel(shadowed));
        for i = 1:numel(shadowed)
            if shadowed(i).identical
                tag = 'byte-identical';
            else
                tag = '*** DIFFERENT CONTENT ***';
            end
            fprintf('    %-16s also at %s  [%s]\n', ...
                shadowed(i).name, shadowed(i).other, tag);
        end
    end
end
end

% -------------------------------------------------------------------------
function tf = localIsInside(file, root)
f = localCanonical(file);
r = localCanonical(root);
if ~endsWith(r, filesep)
    r = [r filesep];
end
tf = strncmp(f, r, numel(r));
end

function p = localCanonical(p)
% Resolve symlinks and relative segments so that comparisons are meaningful on
% macOS, where /tmp and /private/tmp denote the same directory.
try
    if exist(p, 'dir') == 7
        d = dir(p);
        p = d(1).folder;
    else
        d = dir(p);
        if ~isempty(d)
            p = fullfile(d(1).folder, d(1).name);
        end
    end
catch
    % keep p as given
end
end

function tf = localSameBytes(a, b)
tf = false;
try
    fa = dir(a);
    fb = dir(b);
    if isempty(fa) || isempty(fb) || fa(1).bytes ~= fb(1).bytes
        return
    end
    ha = fopen(a, 'r'); da = fread(ha, Inf, '*uint8'); fclose(ha);
    hb = fopen(b, 'r'); db = fread(hb, Inf, '*uint8'); fclose(hb);
    tf = isequal(da, db);
catch
    tf = false;
end
end
