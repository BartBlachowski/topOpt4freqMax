function flat = olhoffcurrent_legacy_view(cfg)
%OLHOFFCURRENT_LEGACY_VIEW  Flat rendering of an effective configuration.
%
%   flat = OLHOFFCURRENT_LEGACY_VIEW(cfg) returns olh.config.toLegacy(cfg): the
%   same formulation expressed in the historical flat field vocabulary
%   (innerSolver, multRule, moveFamily, s2Levels, tolInner, outerGuard, ...).
%
%   WHAT THIS IS FOR, AND WHAT IT IS NOT
%   ------------------------------------
%   It exists so that checks, manifests and comparisons written against the
%   historical vocabulary keep working WITHOUT anyone re-deriving the
%   production realization by hand.  It is a READ-ONLY VIEW.
%
%   It is NOT a second configuration.  Nothing may be built from it and fed to
%   a solver in production: the canonical cfg is the configuration, and
%   olhoffcurrent_config is the only sanctioned way to obtain one.
%
%   flat.diag is normalized to a logical here.  toLegacy omits the field
%   entirely when diagnostics are off, because the pre-canonical solver treated
%   an absent field as false; a checker that reads flat.diag should not have to
%   know that.
%
%   See also OLHOFFCURRENT_CONFIG, OLH.CONFIG.TOLEGACY.

flat = olh.config.toLegacy(cfg);
if ~isfield(flat, 'diag') || isempty(flat.diag)
    flat.diag = false;
end
flat.diag = logical(flat.diag);
end
