function [tf, reason] = olhoffcurrent_is_artifact(name)
%OLHOFFCURRENT_IS_ARTIFACT  Is this a filesystem/editor artifact, not source?
%
%   [tf, reason] = OLHOFFCURRENT_IS_ARTIFACT(name) classifies ONE file by its
%   name (a bare name or any path; only the last component is examined).
%
%   THE SINGLE ARTIFACT POLICY.  Every integrity check in this tree asks this
%   function.  There is deliberately no second ignore list anywhere: two lists
%   drift, and a drifted ignore list is how a real source file gets skipped.
%
%   WHY THIS EXISTS
%   ---------------
%   The source-integrity manifest hashes every file under +impl/ and treats an
%   unrecorded file as evidence that production source changed.  macOS writes
%   .DS_Store into any directory a Finder window or a Spotlight pass touches,
%   and MATLAB writes .asv/.m~ backups beside an edited file.  Before this
%   policy existed, two .DS_Store files were enough to make
%   olhoffcurrent_currentness report LOCAL_MODIFIED -- its most serious state,
%   meaning "production source has been edited in place" -- and to fail the
%   production preflight, while all 74 recorded source files were byte
%   identical.  Filesystem metadata is not executable source, and confusing the
%   two blocks production for no scientific reason.
%
%   THIS DOES NOT WEAKEN INTEGRITY CHECKING.
%   ----------------------------------------
%   The rule below is ALLOW-LIST-FIRST and fails safe:
%
%     1. Anything MATLAB can EXECUTE is never an artifact, whatever it is
%        called -- .m, .mlx, .mlapp, .p, .mex*, .slx, .fig.  This is tested
%        FIRST and overrides every other rule, so no wildcard can ever hide a
%        function, a class file or a package member.
%     2. Only the EXACT names and EXACT extensions enumerated below are
%        artifacts.  There is no broad wildcard and no "starts with a dot"
%        rule: an unrecognised file stays source, and therefore still blocks.
%
%   An unexpected .m file under +impl/ is a HARD BLOCK, by construction.
%
%   See also OLHOFFCURRENT_SOURCE_MANIFEST, OLHOFFCURRENT_CURRENTNESS.

[~, base, ext] = fileparts(char(name));
leaf = [base ext];
lext = lower(ext);

% ---- 1. EXECUTABLE / MATLAB SOURCE: never an artifact -------------------
% Checked first and unconditionally.  Note that '.m~' and '.asv' are NOT in
% this list -- fileparts('foo.m~') yields ext '.m~', which is not '.m'.
executable = {'.m','.mlx','.mlapp','.p','.slx','.mdl','.fig','.mat','.json','.md','.txt','.csv'};
if any(strcmp(lext, executable)) || strncmp(lext, '.mex', 4)
    tf = false; reason = 'source/data/documentation - never ignored';
    return
end

% ---- 2. exact artifact names -------------------------------------------
names = {'.DS_Store', 'Thumbs.db', 'desktop.ini', '.localized'};
if any(strcmpi(leaf, names))
    tf = true; reason = 'platform filesystem metadata';
    return
end

% macOS AppleDouble resource fork written onto non-native filesystems.
if strncmp(leaf, '._', 2)
    tf = true; reason = 'macOS AppleDouble resource fork';
    return
end

% ---- 3. exact editor-backup extensions ---------------------------------
backups = {'.asv', '.m~', '.autosave', '.orig', '.rej', '.swp', '.swo', '.bak'};
if any(strcmp(lext, backups))
    tf = true; reason = 'editor backup file';
    return
end

% ---- 4. everything else is source until proved otherwise ---------------
tf = false; reason = 'unrecognised - treated as source, so it still blocks';
end
