function sd_config_dump()
%SD_CONFIG_DUMP  Flatten the four effective 480x60 configurations to JSON:
%   S480 (stored), M1 (stored in the M1 result), C480 canary (stored), and target
%   canonical production resolved now through olhoffcurrent_config (read-only).
[P, guard] = sd_use_target(); %#ok<ASGLU>
[cProd, ~] = olhoffcurrent_config(480, 60);
S  = load(P.s480, 'cfg');
M1 = load(fullfile(P.m1dir, 'M1_480x60_res.mat'), 'cfg');
C  = load(P.c480traj, 'cfg');
cfgs = {'S480_source_native', S.cfg; 'M1_source_simp4b', M1.cfg; 'C480_target_threerung', C.cfg; 'PROD_target_canonical', cProd};
out = struct();
for i = 1:size(cfgs,1)
    L = sd_flatten(cfgs{i,2});
    rec = struct('path', L(:,1), 'value', cellfun(@sd_show, L(:,2), 'UniformOutput', false));
    out.(cfgs{i,1}) = rec;
end
fid = fopen(fullfile(P.eval, 'effective_configs_480.json'), 'w');
fprintf(fid, '%s\n', jsonencode(out, 'PrettyPrint', true)); fclose(fid);
fprintf('wrote %d configs\n', size(cfgs,1));
end

function s = sd_show(v)
if ischar(v) || isstring(v), s = char(v);
elseif isnumeric(v) || islogical(v), s = mat2str(v, 17);
elseif iscell(v), s = strjoin(cellfun(@sd_show, v, 'UniformOutput', false), ' | ');
else, s = class(v);
end
end
