function cp_supplement(nelx, nely)
%CP_SUPPLEMENT  The per-iteration columns the frozen CSV schema does not carry.
%
%   cv_export's 55-column schema is FROZEN: it is the validated C320 study's
%   own schema, and this study reuses it unchanged so the canary CSVs are
%   column-for-column comparable to that oracle.  It does not carry omega3+,
%   gap23, or the three timing channels tEig / tGrad / tInner.
%
%   Part B requires all of them.  They ARE retained -- hist.omega is Jcalc x n
%   and hist.tEig/tGrad/tInner are written every iteration -- so rather than
%   edit another study's frozen exporter, this function reads the trajectory
%   .mat the canary already wrote and emits the missing columns alongside.
%
%   Runs AFTER a canary, reads only, and writes no scientific state.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));

tag = sprintf('C%dx%d_three_rung', nelx, nely);
f = fullfile(root,'evidence','three_rung_canary_preflight', ...
             sprintf('%s_trajectory.mat', tag));
assert(isfile(f), 'cp_supplement:NoTrajectory', 'no trajectory for %s', tag);
S = load(f, 'hist');
h = S.hist;
n = numel(h.N);

om = h.omega;                       % Jcalc x n
J  = size(om,1);
tOther = h.tOuter(:) - h.tEig(:) - h.tGrad(:) - h.tInner(:);
gap23 = nan(n,1);
if J >= 3, gap23 = (om(3,:).' - om(2,:).')./om(2,:).'; end

cols = {'outer'};
M = (1:n).';
for j = 1:J
    cols{end+1} = sprintf('omega%d', j); %#ok<AGROW>
    M = [M, om(j,:).']; %#ok<AGROW>
end
extra = {'gap23', gap23; 'tOuter', h.tOuter(:); 'tEig', h.tEig(:); ...
         'tGrad', h.tGrad(:); 'tInner', h.tInner(:); 'tOther', tOther; ...
         'multN', h.N(:); 'multJ', double(h.multJ(:)); 'degen', h.degen(:); ...
         'nInner', h.nInner(:); 'dxOuter', h.dxOuter(:); 'dxNorm2', h.dxNorm2(:); ...
         'move', h.move(:); 'stage', h.stage(:); 'vol', h.vol(:)};
for k = 1:size(extra,1)
    cols{end+1} = extra{k,1}; %#ok<AGROW>
    M = [M, double(extra{k,2})]; %#ok<AGROW>
end

out = fullfile(study,'runs',sprintf('%s_supplement.csv', tag));
fid = fopen(out,'w');
fprintf(fid, '%s\n', strjoin(cols, ','));
fmt = [repmat('%.17g,',1,numel(cols)-1) '%.17g\n'];
fprintf(fid, fmt, M.');
fclose(fid);
fprintf('[cp_supplement] wrote %s  (%d rows x %d cols)\n', out, n, numel(cols));
end
