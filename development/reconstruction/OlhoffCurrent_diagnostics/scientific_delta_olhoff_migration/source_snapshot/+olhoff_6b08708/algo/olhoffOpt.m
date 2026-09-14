function res = olhoffOpt(cfg)
%OLHOFFOPT  Compatibility entry point for the Du & Olhoff (2007) solver.
%
%   res = OLHOFFOPT(cfg) accepts EITHER form of configuration:
%
%     * a CANONICAL cfg (olh.config.resolve) -- forwarded straight to
%       olhoffSolve;
%     * a LEGACY FLAT cfg -- the form built by algo/defaultCfg.m, by
%       TMA_FROZEN_CFGS.mat and by every audit runner.  It is translated by
%       olh.config.fromLegacy and then solved by the same olhoffSolve.
%
%   THERE IS ONE SOLVER.  This file contains no mathematics.  Its only job is
%   to preserve the legacy calling contract so that every historical runner
%   keeps working unmodified -- which is what allows the audit trees to stay
%   frozen (Phase 23).
%
%   The legacy contract this shim reproduces:
%     res.cfg   is the FLAT config, with the three mutations the old solver
%               performed on its own input: the five isfield defaults, the
%               resolved cfg.mmasubPath, and cfg.rminEl recomputed from
%               cfg.rminPhys.  Audit code reads res.cfg.rminEl.
%   Additions (purely additive, nothing reads them in legacy code):
%     res.cfgCanonical   the effective canonical configuration
%     res.cfgWarnings    validation warnings, collected rather than raised
%     res.status         explicit status, see olhoffSolve
%
%   The pre-canonical implementation is preserved verbatim at
%   architecture/legacy/olhoffOpt_PRE_CANONICAL.m.

if isfield(cfg,'domain') && isfield(cfg,'material')
    res = olhoffSolve(cfg);
    return
end

canon = olh.config.fromLegacy(cfg);
[canon, warnings] = olh.config.validate(canon);

res = olhoffSolve(canon);

res.cfgCanonical = canon;
res.cfgWarnings  = warnings;
res.cfg          = local_legacyEcho(cfg);
end

% =========================================================================
function cfg = local_legacyEcho(cfg)
%LOCAL_LEGACYECHO  Reproduce, for res.cfg only, the three mutations the
%   pre-canonical solver applied to its own input.  The solver itself no longer
%   writes to its configuration; this exists so that res.cfg is byte-for-byte
%   what audit code has always seen.
if ~isfield(cfg,'mmaVariant'), cfg.mmaVariant = 'published'; end
if ~isfield(cfg,'outerNorm'),  cfg.outerNorm  = 'l2';        end
if ~isfield(cfg,'innerVar'),   cfg.innerVar   = 'drho';      end
if ~isfield(cfg,'moveFamily'), cfg.moveFamily = 'S0';        end
if ~isfield(cfg,'outerGuard'), cfg.outerGuard = 'none';      end
cfg.mmasubPath = which('mmasub');
if isfield(cfg,'rminPhys') && ~isempty(cfg.rminPhys) && cfg.rminPhys > 0
    dyEl = cfg.b/cfg.nely;
    cfg.rminEl = cfg.rminPhys/dyEl;
end
end
