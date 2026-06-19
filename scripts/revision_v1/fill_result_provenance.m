function r = fill_result_provenance(r, cfg)
%FILL_RESULT_PROVENANCE  Populate r.provenance from the current system state.
%
%   r = fill_result_provenance(r)        system info only
%   r = fill_result_provenance(r, cfg)   also stores a hex fingerprint of cfg
%
%   Reads MATLAB version, platform, CPU brand, RAM, current git HEAD, and
%   UTC timestamp.  Does not throw on failure; fields stay '' or NaN when
%   a query cannot be completed.
%
%   Call this once at the start of an experiment (before the timed region)
%   so that commit and platform metadata are captured even if the run fails.
%
%   See also MAKE_EXPERIMENT_RESULT, CHECK_EXPERIMENT_RESULT.

if nargin < 2, cfg = []; end

r.provenance.matlab_version = version();
r.provenance.platform       = computer();
r.provenance.timestamp_utc  = localUtcTimestamp();
r.provenance.cpu_info       = localCpuInfo();
r.provenance.ram_GB         = localRamGB();
r.provenance.commit         = localGitCommit();

if ~isempty(cfg)
    r.provenance.config_hash = localHashStruct(cfg);
end
end

% =========================================================================
function ts = localUtcTimestamp()
ts = '';
try
    d  = datetime('now', 'TimeZone', 'UTC');
    ts = char(d, "yyyy-MM-dd'T'HH:mm:ss'Z'");
catch
    try, ts = datestr(now, 'yyyy-mm-ddTHH:MM:SSZ'); catch, end %#ok<TNOW1,DATST>
end
end

% =========================================================================
function info = localCpuInfo()
info = '';
try
    [s, out] = system('sysctl -n machdep.cpu.brand_string 2>/dev/null');
    if s == 0 && ~isempty(strtrim(out))
        info = strtrim(out);
        return;
    end
catch, end
try
    [s, out] = system('grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2');
    if s == 0, info = strtrim(out); end
catch, end
end

% =========================================================================
function gb = localRamGB()
gb = NaN;
try
    [~, hw] = memory();
    gb = hw.PhysicalMemory.Total / 1e9;
    return;
catch, end
try
    [s, out] = system('sysctl -n hw.memsize 2>/dev/null');
    if s == 0
        v = str2double(strtrim(out));
        if ~isnan(v), gb = v / 1e9; end
    end
catch, end
end

% =========================================================================
function sha = localGitCommit()
sha = 'unknown';
try
    [s, out] = system('git rev-parse HEAD 2>/dev/null');
    if s == 0 && ~isempty(strtrim(out))
        sha = strtrim(out);
    end
catch, end
end

% =========================================================================
function h = localHashStruct(s)
% 8-char hex fingerprint of jsonencode(s) via polynomial rolling hash.
% Not cryptographic; changes whenever the serialised config changes.
h = '';
try
    str   = jsonencode(s);
    bytes = double(uint8(str));   % safe for ASCII/Latin-1 field names
    M     = 2^32;
    P     = 1000003;              % large prime avoids clustering
    acc   = 0;
    for k = 1:numel(bytes)
        acc = mod(acc * P + bytes(k), M);
    end
    h = lower(dec2hex(acc, 8));
catch
    h = '';
end
end
