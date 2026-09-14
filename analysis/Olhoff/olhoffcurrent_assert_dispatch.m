function report = olhoffcurrent_assert_dispatch(varargin)
%OLHOFFCURRENT_ASSERT_DISPATCH  Fail-closed proof that ONE Olhoff is visible.
%
%   report = OLHOFFCURRENT_ASSERT_DISPATCH() checks every function name this
%   implementation owns and REFUSES TO RETURN unless, for each of them:
%
%     (1) the name resolves at all;
%     (2) the WINNING resolution lies inside analysis/OlhoffCurrent/+impl;
%     (3) there is NO OTHER CANDIDATE anywhere outside this tree.
%
%   Options:
%     'Throw'   (default true)   false -> populate the report and return
%                                without raising, for the negative tests
%
%   WHY (3) EXISTS
%   --------------
%   The gate this replaces checked resolution with which(name) -- the FIRST hit
%   only.  That proves "the winner is right"; it does not prove "there is only
%   one candidate".  A second copy of innerLoop, massScale or mmasub sitting
%   further down the path is invisible to that test for as long as the first
%   hit happens to be correct, and becomes the executed code the moment any
%   addpath(...,'-begin') or genpath sweep reorders the path.  Helper shadowing
%   is the documented failure mode in this project (49 bare names are shared
%   between the upstream tree and Matlab/reproduction2007), and it must be
%   detected even when the top-level solver resolves correctly.
%
%   So this function uses which(name,'-all') and treats a SECOND candidate as a
%   blocker in its own right, whether or not the blacklist knows about it.
%
%   report fields:
%     ok            logical, true only when nothing was found wrong
%     resolved      struct array (name, file) -- write straight into a manifest
%     blockers      cellstr, human-readable, empty when ok
%     warnings      cellstr, non-fatal name collisions (MATLAB's own install,
%                   and files declared in olhoffcurrent_known_collisions)
%     declaredCollisions  struct array of declared collisions actually observed,
%                   with the colliding file's current hash, for the manifest
%     pathEntries   forbidden directories found on the MATLAB path
%
%   See also OLHOFFCURRENT_PATHS, OLHOFFCURRENT_FORBIDDEN_PATHS.

p = inputParser();
p.addParameter('Throw', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
doThrow = p.Results.Throw;

root     = olhoffcurrent_root();
core     = fullfile(root, '+impl');
repoRoot = fileparts(fileparts(root));            % <repo>/analysis/OlhoffCurrent
mlroot   = matlabroot();
names    = olhoffcurrent_owned_names();
[repoRel, absForbidden] = olhoffcurrent_forbidden_paths();

% Absolute forbidden prefixes, repository-relative ones made absolute.
forbidden = absForbidden(:).';
for k = 1:numel(repoRel)
    forbidden{end+1} = fullfile(repoRoot, repoRel{k}); %#ok<AGROW>
end

under = @(f, d) ~isempty(f) && strncmp(f, [d filesep], numel(d)+1);

% ---- 1. no forbidden directory may sit on the MATLAB path ---------------
onPath = strsplit(path, pathsep);
pathEntries = {};
for i = 1:numel(forbidden)
    for k = 1:numel(onPath)
        if strcmp(onPath{k}, forbidden{i}) || under(onPath{k}, forbidden{i})
            pathEntries{end+1} = onPath{k}; %#ok<AGROW>
        end
    end
end

blockers = {};
warnings = {};
collisions = struct('symbol', {}, 'file', {}, 'sha256', {}, 'declaredSha256', {});
known = olhoffcurrent_known_collisions();
for k = 1:numel(pathEntries)
    blockers{end+1} = sprintf(...
        'FORBIDDEN DIRECTORY ON PATH: %s', pathEntries{k}); %#ok<AGROW>
end

% ---- 2 & 3. per-symbol resolution and shadow detection ------------------
resolved = struct('name', {}, 'file', {});
for i = 1:numel(names)
    nm  = names{i};
    all = which(nm, '-all');
    if ischar(all), all = {all}; end
    if isempty(all)
        blockers{end+1} = sprintf('%s -> UNRESOLVED', nm); %#ok<AGROW>
        resolved(end+1) = struct('name', nm, 'file', ''); %#ok<AGROW>
        continue
    end

    win = all{1};
    resolved(end+1) = struct('name', nm, 'file', win); %#ok<AGROW>

    if ~under(win, core)
        blockers{end+1} = sprintf(...
            '%s RESOLVES OUTSIDE PRODUCTION: %s', nm, win); %#ok<AGROW>
    end

    % Every further candidate is a potential shadow.  Ours are fine; the rest
    % are classified, because "any second hit is fatal" would be unusable.
    for k = 1:numel(all)
        c = all{k};
        if under(c, core); continue; end

        % A class method (@cls/f.m) or a package member (+pkg/f.m) CANNOT
        % shadow a bare-name call: MATLAB dispatches those on the object type
        % or the qualified name, never by path order against a bare function.
        % Skipping them is not a weakening of the gate -- they are not
        % candidates for the resolution being tested.
        if local_isScoped(c); continue; end

        inForbidden = '';
        for f = 1:numel(forbidden)
            if under(c, forbidden{f}); inForbidden = forbidden{f}; break; end
        end

        if ~isempty(inForbidden)
            % A competing Olhoff tree we already know about.  Always fatal.
            blockers{end+1} = sprintf('%s SHADOWED BY: %s [%s]', ...
                nm, c, inForbidden); %#ok<AGROW>
        elseif under(c, mlroot) || ~isempty(strfind(c, 'built-in')) %#ok<STREMP>
            % MATLAB's own installation is not an Olhoff implementation.  Our
            % copy wins and that is the intended outcome, but the collision is
            % recorded so a future release adding a bare function with one of
            % our names cannot pass unnoticed.
            warnings{end+1} = sprintf('%s also exists in the MATLAB installation: %s', ...
                nm, c); %#ok<AGROW>
        else
            % Is this a DECLARED non-Olhoff collision?  Declaring one does not
            % permit it to win -- only to exist.
            di = local_declared(known, nm, c, repoRoot);
            if di > 0
                if k == 1
                    % The declared file is the WINNER.  Hard blocker: this is
                    % the silent-wrong-variant failure the registry exists to
                    % prevent, not the benign coexistence it tolerates.
                    blockers{end+1} = sprintf(...
                        '%s RESOLVES TO A DECLARED NON-PRODUCTION COPY: %s -- %s', ...
                        nm, c, known(di).reason); %#ok<AGROW>
                else
                    try
                        h = olhoffcurrent_sha256_file(c);
                    catch
                        h = '<unreadable>';
                    end
                    tagH = '';
                    if ~isempty(known(di).sha256) && ~strcmp(h, known(di).sha256)
                        tagH = sprintf(' [CONTENT CHANGED: %s, declared %s]', ...
                            h, known(di).sha256);
                    end
                    warnings{end+1} = sprintf(...
                        '%s also exists at %s (declared, production copy wins)%s', ...
                        nm, c, tagH); %#ok<AGROW>
                    collisions(end+1) = struct('symbol', nm, 'file', c, ...
                        'sha256', h, 'declaredSha256', known(di).sha256); %#ok<AGROW>
                end
            else
                % A bare .m file with one of our names, outside this tree,
                % outside MATLAB, and declared nowhere.  This is exactly the
                % case the blacklist cannot cover: an Olhoff tree nobody has
                % told us about.  Fail closed.
                blockers{end+1} = sprintf(...
                    '%s SHADOWED BY: %s [UNDECLARED Olhoff tree or stray copy]', ...
                    nm, c); %#ok<AGROW>
            end
        end
    end
end

report = struct('ok', isempty(blockers), 'resolved', resolved, ...
                'blockers', {blockers}, 'warnings', {warnings}, ...
                'declaredCollisions', collisions, 'pathEntries', {pathEntries});

if ~report.ok && doThrow
    error('olhoffcurrent_assert_dispatch:PathContaminated', ...
        ['PRODUCTION OLHOFF PATH IS CONTAMINATED -- refusing to run.\n' ...
         'Exactly one Olhoff implementation may be visible: %s\n\n  %s\n\n' ...
         'See analysis/OLHOFF_IMPLEMENTATION_MAP.md.'], ...
        core, strjoin(blockers, sprintf('\n  ')));
end
end

function i = local_declared(known, nm, file, repoRoot)
%LOCAL_DECLARED  Index of the registry entry matching this symbol and file, or 0.
i = 0;
for k = 1:numel(known)
    if strcmp(known(k).symbol, nm) && ...
       strcmp(file, fullfile(repoRoot, known(k).relativePath))
        i = k; return
    end
end
end

function tf = local_isScoped(f)
%LOCAL_ISSCOPED  True for a class method or package member, which cannot take
%   part in bare-name path resolution.
parts = strsplit(f, filesep);
tf = any(cellfun(@(p) ~isempty(p) && (p(1) == '@' || p(1) == '+'), parts(1:end-1)));
end
