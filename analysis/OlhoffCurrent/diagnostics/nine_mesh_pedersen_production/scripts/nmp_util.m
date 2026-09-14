function varargout = nmp_util(op, varargin)
%NMP_UTIL  Small shared helpers for the observation hooks (no MATLAB-path side effects).
%   nmp_util('json', path, S)        pretty JSON
%   nmp_util('text', path, txt)      plain text
%   nmp_util('event', ctx, msg)      append a timestamped line to <run_root>/HOOK_EVENTS.log
%   h = nmp_util('sha256double', x)  SHA-256 of little-endian IEEE bytes, column order
%                                    (identical to sd_sha256_double of the upstream sweep audit)
switch op
    case 'json'
        fid = fopen(varargin{1}, 'w');
        fprintf(fid, '%s\n', jsonencode(varargin{2}, 'PrettyPrint', true));
        fclose(fid);
    case 'text'
        fid = fopen(varargin{1}, 'w');
        fprintf(fid, '%s\n', varargin{2});
        fclose(fid);
    case 'event'
        ctx = varargin{1};
        if exist(ctx.run_root_abs, 'dir') ~= 7; mkdir(ctx.run_root_abs); end
        fid = fopen(fullfile(ctx.run_root_abs, 'HOOK_EVENTS.log'), 'a');
        fprintf(fid, '%s  %s\n', nmp_now(), varargin{2});
        fclose(fid);
    case 'sha256double'
        b = typecast(double(varargin{1}(:)).', 'uint8');
        md = java.security.MessageDigest.getInstance('SHA-256');
        md.update(b);
        d = typecast(md.digest(), 'uint8');
        varargout{1} = lower(reshape(dec2hex(d, 2).', 1, []));
    otherwise
        error('nmp_util:op', 'unknown op %s', op);
end
end
