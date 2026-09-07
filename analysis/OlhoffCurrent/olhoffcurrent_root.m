function r = olhoffcurrent_root()
%OLHOFFCURRENT_ROOT  Absolute path of the canonical production Olhoff tree.
%
%   Derived from this file's own location, never from a stored absolute path,
%   so the repository can be cloned or moved without editing anything.
r = fileparts(mfilename('fullpath'));
end
