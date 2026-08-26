function root = repro2007_root()
%REPRO2007_ROOT  Absolute path of the clean-room reproduction implementation root.
%
%   root = REPRO2007_ROOT() returns the directory that contains algo/, fem/,
%   filter/, mma/, runs/ and runner/ for the Du-Olhoff 2007 clean-room
%   benchmark reproduction (Eq. 22 LP route).
%
%   Every path and identity check in this implementation is expressed relative
%   to this one function, so that moving the tree cannot silently split the
%   implementation across two roots.
%
%   See also REPRO2007_PATHS, REPRO2007_ASSERT_IDENTITY.

root = fileparts(fileparts(mfilename('fullpath')));
end
