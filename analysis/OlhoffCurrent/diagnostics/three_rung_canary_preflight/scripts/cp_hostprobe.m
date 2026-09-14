function h = cp_hostprobe(label, study)
%CP_HOSTPROBE  Part G: machine-load and a short non-scientific timing calibration.
%
%   Recorded immediately before and after each canary.  It makes gross host-load
%   differences between the two canaries visible, so a wall-time comparison is
%   not silently a comparison of two different machine states.
%
%   NO thermal or cache control is attempted, and none is claimed.  Small
%   wall-time differences are NOT evidence of exact reproducibility.
%
%   The calibration kernel is a dense matrix product and a sparse solve of
%   FIXED size, unrelated to any mesh: it measures the host, not the science.

if nargin < 2, study = fileparts(fileparts(mfilename('fullpath'))); end

h = struct();
h.label = label;
h.when  = char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ss'));
h.loadavg  = local_sh('sysctl -n vm.loadavg');
h.swap     = local_sh('sysctl -n vm.swapusage');
h.vm_stat  = local_sh('vm_stat | head -6');
h.freeRAM_pages = local_sh('vm_stat | awk ''/Pages free/{print $3}''');
h.matlabProcs = local_sh('ps -eo pid,pcpu,rss,comm | grep -i -c "[M]ATLAB"');
h.topCPU   = local_sh('ps -eo pcpu,comm -r | head -6');
h.threads  = maxNumCompThreads;

% ---- short, fixed, non-scientific calibration --------------------------
% DETERMINISTIC BY CONSTRUCTION.  No rand, and deliberately no rng() call: the
% probe runs immediately before a scientific solve and must not touch global
% state the solve could depend on.  eigSolve already supplies ARPACK a fixed
% start vector, so the RNG is not in fact a determinism dependency -- but a
% probe that mutates it anyway would be an unnecessary risk to take.
nq = 1200;
A = cos((1:nq)'*(1:nq)*1e-4);
B = sin((1:nq)'*(1:nq)*1e-4);
A*B;                                      %#ok<VUNUS>  warm-up, discarded
t = zeros(3,1);
for k = 1:3, tc = tic; C = A*B; t(k) = toc(tc); end %#ok<NASGU>
h.calib_dgemm_s = median(t);

nn = 200000;
S = spdiags([-ones(nn,1) 2.2*ones(nn,1) -ones(nn,1)], -1:1, nn, nn);
b = ones(nn,1);
S\b;                                      %#ok<VUNUS>  warm-up, discarded
t = zeros(3,1);
for k = 1:3, tc = tic; x = S\b; t(k) = toc(tc); end %#ok<NASGU>
h.calib_spsolve_s = median(t);

outFile = fullfile(study,'evidence',sprintf('hostprobe_%s.json', label));
if ~isfolder(fileparts(outFile)), mkdir(fileparts(outFile)); end
fid = fopen(outFile,'w'); fprintf(fid,'%s', jsonencode(h,'PrettyPrint',true)); fclose(fid);
fprintf('[cp_hostprobe] %s  load=%s  dgemm=%.3fs  spsolve=%.3fs\n', ...
    label, h.loadavg, h.calib_dgemm_s, h.calib_spsolve_s);
end

function s = local_sh(cmd)
[st, out] = system(cmd);
if st ~= 0, s = ''; else, s = strtrim(out); end
end
