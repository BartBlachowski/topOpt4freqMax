function d = olhoffcurrent_test_digest(res)
%OLHOFFCURRENT_TEST_DIGEST  Bitwise digest of a solver result, timing excluded.
%
%   d = OLHOFFCURRENT_TEST_DIGEST(res) returns a struct of SHA-256 digests over
%   the SCIENTIFIC content of an olhoffSolve result, so that a compact tracked
%   fixture can stand for a result bitwise:
%
%     rho, omega, lambda              class + size + raw little-endian bytes
%     hist.<every field>              same, EXCEPT the nondeterministic timing
%                                     fields tEig, tGrad, tInner, tOuter
%     aux.<every field>               when present (adaptive-box / Pedersen runs)
%     exhaustion                      when present (stage-exhaustion runs)
%     log                             the text, one digest over all lines
%     nOuter, innerTotal, status      plain values
%
%   Two results with equal digests are equal under isequal on every digested
%   array (a SHA-256 collision aside).  Also returned, for human comparison with
%   historical records: rho_sha256_bytes (SHA-256 of the float64 bytes alone,
%   the convention of the two_branch_controller_validation records) and omega
%   printed with %.17g.
%
%   Test helper only; never called by production code.

TIMING = {'tEig', 'tGrad', 'tInner', 'tOuter'};
d = struct();
d.rho    = local_hash(res.rho);
d.omega  = local_hash(res.omega);
d.lambda = local_hash(res.lambda);
d.nOuter = numel(res.hist.N);
d.innerTotal = sum(double(res.hist.nInner));
d.status = char(res.status);
d.log    = local_hash(strjoin(cellfun(@char, res.log(:).', 'UniformOutput', false), char(10)));
d.logLines = res.log(:).';
f = sort(setdiff(fieldnames(res.hist), TIMING));
d.histFields = f(:).';
d.hist = struct();
for k = 1:numel(f), d.hist.(f{k}) = local_hash(res.hist.(f{k})); end
if isfield(res, 'aux') && isstruct(res.aux)
    a = sort(fieldnames(res.aux));
    d.aux = struct();
    for k = 1:numel(a), d.aux.(a{k}) = local_hash(res.aux.(a{k})); end
end
if isfield(res, 'exhaustion')
    d.exhaustion = local_hash(res.exhaustion);
end
d.rho_sha256_bytes = local_sha(typecast(double(res.rho(:)), 'uint8'));
d.omega_17g = arrayfun(@(v) sprintf('%.17g', v), double(res.omega(:)).', 'UniformOutput', false);
end

% =========================================================================
function h = local_hash(v)
h = local_sha(local_bytes(v));
end

function b = local_bytes(v)
cls = class(v);
hdr = uint8(sprintf('%s|%s|', cls, mat2str(size(v))));
if isnumeric(v)
    if ~isreal(v), v = [real(v(:)); imag(v(:))]; hdr = [hdr uint8('complex|')]; end
    body = typecast(v(:), 'uint8');
elseif islogical(v)
    body = uint8(v(:));
elseif ischar(v)
    body = typecast(uint16(v(:)), 'uint8');
elseif isstring(v)
    body = local_bytes(cellstr(v));
elseif iscell(v)
    parts = cellfun(@local_bytes, v(:).', 'UniformOutput', false);
    body = [parts{:}];
    if isempty(body), body = uint8([]); end
elseif isstruct(v)
    fn = sort(fieldnames(v));
    parts = cell(1, numel(fn)*numel(v));
    q = 0;
    for e = 1:numel(v)
        for k = 1:numel(fn)
            q = q + 1;
            parts{q} = [uint8([fn{k} '=']) local_bytes(v(e).(fn{k}))];
        end
    end
    body = [parts{:}];
    if isempty(body), body = uint8([]); end
else
    error('olhoffcurrent_test_digest:Unsupported', 'cannot digest class %s', cls);
end
b = [hdr reshape(uint8(body), 1, [])];
end

function h = local_sha(bytes)
md = java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(bytes), md.update(bytes(:)); end
x = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(x, 2).', 1, []));
end
