function root = olhoffm4_root()
%OLHOFFM4_ROOT  Absolute path of the imported Du-Olhoff reconstruction (M4).
%
%   This namespace holds the CONFERENCE-ACTIVE Du-Olhoff reconstruction,
%   imported from /Users/piotrek/Programming/Matlab/Olhoff.  It is not one of
%   the superseded historical Olhoff implementations under analysis/Olhoff*.
%   See analysis/OLHOFF_IMPLEMENTATION_STATUS.md and IMPORT_MANIFEST.json.
%
%   See also OLHOFFM4_PATHS, OLHOFFM4_CONFIG, OLHOFFM4_RUN.

root = fileparts(mfilename('fullpath'));
end
