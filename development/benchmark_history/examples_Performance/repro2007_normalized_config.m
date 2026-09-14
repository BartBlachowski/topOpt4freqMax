function [nc, hash, text, hashedFields] = repro2007_normalized_config(cfg)
%REPRO2007_NORMALIZED_CONFIG  Canonical, hashable form of the effective
%   configuration at the OLHOFFOPT boundary.
%
%   [nc, hash, text] = REPRO2007_NORMALIZED_CONFIG(cfg)
%
%     nc    ordered struct: every field of CFG that can affect numerical
%           behaviour, renamed to the repository's vocabulary, plus the derived
%           quantities the raw struct only implies (element size, aspect ratio,
%           effective filter radius, mode window).
%     hash  SHA-256 of TEXT.
%     text  the canonical serialization -- one `key = value` line per field, in
%           a FIXED order, doubles printed with %.17g so the text round-trips
%           to the same bits.
%     hashedFields
%           the field names that entered TEXT, i.e. the ones declared
%           numerical.  Returned so that a caller comparing two configurations
%           can ask which differences matter instead of keeping its own copy of
%           the answer.
%
%   WHY THIS EXISTS
%   ---------------
%   The performance benchmark reaches OLHOFFOPT through
%
%       run_topopt_from_json -> run_repro2007 -> repro2007_config -> olhoffOpt
%
%   and every one of those hops can rename, default or override a field.  A
%   previous defect mapped the benchmark's void MATERIAL density floor onto the
%   reproduction's DESIGN VARIABLE bound (DIAGNOSTIC_REPRO2007_BENCHMARK.md):
%   both source JSONs were correct, both wrappers ran without error, and the
%   trajectory silently diverged at outer iteration 101.  Comparing the source
%   JSON would not have caught it.  Comparing the struct that OLHOFFOPT
%   actually received would have caught it immediately.
%
%   FAILS CLOSED ON UNKNOWN FIELDS
%   ------------------------------
%   A field of CFG that this function does not know about is an ERROR, not a
%   silent omission.  If the reproduction's configuration gains a knob, the
%   equivalence gate must be taught whether it is numerical before it can pass
%   again -- otherwise the first thing a new knob would do is disappear from
%   the hash that is supposed to police it.
%
%   NON-NUMERICAL FIELDS
%   --------------------
%   `verbose` and `name` are carried in NC (so a comparison can show them) but
%   are excluded from TEXT and therefore from HASH: neither changes a single
%   floating-point operation.  Everything else is hashed.
%
%   See also SHA256_HEX, VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE, OLHOFFOPT.

if ~isstruct(cfg) || numel(cfg) ~= 1
    error('repro2007_normalized_config:InvalidInput', 'cfg must be a scalar struct.');
end

% ---- every field the reproduction's configuration is allowed to carry ----
% numeric-relevant (hashed)                      | reporting-only (not hashed)
numericalFields = { ...
    'a','b','t','E','nu','rhom', ...
    'nelx','nely','bc','support','axial','elemType','massType', ...
    'p','massInterp','rhomin','rho0','volfrac', ...
    'n','Nmax','rminEl','rminPhys','tolMult','move','offDiag', ...
    'innerSolver','filterMode','maxInner','tolInner','minInner', ...
    'maxOuter','tolOuter','solver','threads'};
reportingFields = {'verbose','name'};

known = [numericalFields, reportingFields];
present = fieldnames(cfg);
unknown = setdiff(present, known);
if ~isempty(unknown)
    error('repro2007_normalized_config:UnknownField', ...
        ['Configuration field(s) not classified by this normalizer: %s.\n' ...
         'Refusing to hash a configuration whose effect on the numerics is ' ...
         'undeclared -- add each field to numericalFields or reportingFields ' ...
         'in %s and re-run the equivalence gate.'], ...
        strjoin(unknown(:)', ', '), mfilename('fullpath'));
end
missing = setdiff(numericalFields, present);
if ~isempty(missing)
    error('repro2007_normalized_config:MissingField', ...
        'Configuration is missing required field(s): %s.', strjoin(missing(:)', ', '));
end

g = @(f) cfg.(f);

% ---- geometry and mesh ---------------------------------------------------
nc = struct();
nc.nelx           = g('nelx');
nc.nely           = g('nely');
nc.n_elements     = g('nelx') * g('nely');
nc.length_a       = g('a');
nc.height_b       = g('b');
nc.thickness_t    = g('t');
nc.aspect_ratio   = g('a') / g('b');
nc.dx             = g('a') / g('nelx');
nc.dy             = g('b') / g('nely');
nc.element_aspect = (g('a') / g('nelx')) / (g('b') / g('nely'));

% ---- supports and element ------------------------------------------------
nc.bc             = localChar(g('bc'));
nc.support        = localChar(g('support'));
nc.axial          = localChar(g('axial'));
nc.support_type   = localSupportType(g('bc'));
nc.elem_type      = localChar(g('elemType'));
nc.mass_type      = localChar(g('massType'));

% ---- material and interpolation -----------------------------------------
nc.E0             = g('E');
nc.nu             = g('nu');
nc.rho_material   = g('rhom');
nc.penal_p        = g('p');
nc.mass_interp    = localChar(g('massInterp'));
nc.rhomin         = g('rhomin');
nc.rho0           = g('rho0');
nc.volfrac        = g('volfrac');

% ---- problem -------------------------------------------------------------
nc.target_mode_n  = g('n');
nc.Nmax           = g('Nmax');
nc.J_calc         = g('n') + g('Nmax');   % modes OLHOFFOPT requests each iteration

% ---- filter --------------------------------------------------------------
% OLHOFFOPT overrides rminEl from rminPhys when the latter is set, so the
% radius that is actually used is a derived quantity, not a stored one.
nc.rmin_el_declared = g('rminEl');
nc.rmin_phys        = localEmptyToNaN(g('rminPhys'));
if ~isempty(g('rminPhys')) && g('rminPhys') > 0
    nc.rmin_el_effective = g('rminPhys') / (g('b') / g('nely'));
    nc.rmin_source       = 'physical';
else
    nc.rmin_el_effective = g('rminEl');
    nc.rmin_source       = 'element';
end
nc.filter_mode      = localChar(g('filterMode'));

% ---- algorithm -----------------------------------------------------------
nc.tol_mult       = g('tolMult');
nc.move           = g('move');
nc.off_diag       = logical(g('offDiag'));
nc.inner_solver   = lower(localChar(g('innerSolver')));
nc.max_inner      = g('maxInner');
nc.tol_inner      = g('tolInner');
nc.min_inner      = g('minInner');
nc.max_outer      = g('maxOuter');
nc.tol_outer      = g('tolOuter');

% ---- numerics ------------------------------------------------------------
nc.eig_solver     = localChar(g('solver'));
nc.threads        = g('threads');

% ---- reporting-only (carried, not hashed) -------------------------------
nc.verbose        = localLogical(cfg, 'verbose');
nc.config_name    = localChar(localGet(cfg, 'name', ''));

% Reporting-only fields are carried in NC but excluded here: neither `verbose`
% nor `config_name` can change a floating-point operation.
hashedFields = setdiff(fieldnames(nc), {'verbose', 'config_name'}, 'stable');
lines = cell(numel(hashedFields), 1);
for i = 1:numel(hashedFields)
    lines{i} = sprintf('%s = %s', hashedFields{i}, localFormat(nc.(hashedFields{i})));
end
text = strjoin(lines, newline);
hash = sha256_hex(text);
end

% -------------------------------------------------------------------------
function s = localFormat(v)
if ischar(v)
    s = ['"' v '"'];
elseif islogical(v)
    if v, s = 'true'; else, s = 'false'; end
elseif isempty(v)
    s = '[]';
elseif isnumeric(v) && isscalar(v)
    if isnan(v)
        s = 'NaN';
    else
        s = sprintf('%.17g', v);   % round-trips a double exactly
    end
else
    parts = arrayfun(@(x) sprintf('%.17g', x), v(:).', 'UniformOutput', false);
    s = ['[' strjoin(parts, ',') ']'];
end
end

function c = localChar(v)
if isempty(v)
    c = '';
else
    c = char(string(v));
end
end

function v = localEmptyToNaN(v)
if isempty(v)
    v = NaN;
end
end

function tf = localLogical(s, f)
if isfield(s, f) && ~isempty(s.(f))
    tf = logical(s.(f));
else
    tf = false;
end
end

function v = localGet(s, f, d)
if isfield(s, f) && ~isempty(s.(f))
    v = s.(f);
else
    v = d;
end
end

function code = localSupportType(bc)
switch lower(localChar(bc))
    case 'a', code = 'SS';
    case 'b', code = 'CS';
    case 'c', code = 'CC';
    otherwise, code = ['UNKNOWN(' localChar(bc) ')'];
end
end
