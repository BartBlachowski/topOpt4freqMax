function ctx = nmp_ctx()
%NMP_CTX  Observation-hook context: the lock file named by the NMP_LOCK
%   environment variable.  Read fresh on every call (no persistent state), so
%   the `clear` at the top of performance_comparison.m cannot affect it.
%   The hash the launcher expects is carried in NMP_LOCK_SHA256 and compared by
%   NMP_IDENTITY_SNAPSHOT.
lockPath = getenv('NMP_LOCK');
if isempty(lockPath)
    error('nmp_ctx:NoLock', 'NMP_LOCK is not set; the hooks cannot identify the campaign.');
end
ctx = jsondecode(fileread(lockPath));
ctx.lock_path = lockPath;
ctx.lock_sha256_expected = getenv('NMP_LOCK_SHA256');
end
