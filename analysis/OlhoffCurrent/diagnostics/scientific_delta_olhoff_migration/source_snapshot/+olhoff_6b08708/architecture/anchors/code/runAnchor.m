function runAnchor(label, stage)
%RUNANCHOR  Produce one behavioural-anchor artifact.
%
%   runAnchor(label,'reference')  run with the CURRENT solver and store the
%                                 pre-refactor reference.
%   runAnchor(label,'candidate')  run again and compare bitwise against it.

root = '/Users/piotrek/Programming/Matlab/Olhoff';
here = fullfile(root,'architecture','anchors');
addpath(fullfile(here,'code'));
cd(root); setpaths(); maxNumCompThreads(1);

[cfg, meta] = anchorCfg(label);
t = tic;
res = olhoffOpt(cfg);
wall = toc(t);

rec = anchorRecord(res, cfg);
rec.meta = meta;
dig = anchorDigests(rec);

outDir = fullfile(here, stage);
if ~isfolder(outDir), mkdir(outDir); end
save(fullfile(outDir,[label '.mat']), 'rec','cfg','meta','dig','-v7.3');

fprintf('ANCHOR %-22s stage=%-9s outer=%4d inner=%6d status=%-14s omega1=%.15g vol=%.15g wall=%.1fs\n', ...
    label, stage, rec.nOuter, rec.innerTotal, rec.status, rec.omega(1), rec.volume, wall);
fprintf('DIGEST %-22s science=%s\n', label, dig.science);

if strcmp(stage,'candidate')
    R = load(fullfile(here,'reference',[label '.mat']));
    ref = anchorDigests(R.rec);          % recomputed, so the standard can evolve
    okSci   = strcmp(dig.science,      ref.science);
    okShape = strcmp(dig.logShape,     ref.logShape);
    okText  = strcmp(dig.presentation, ref.presentation);
    fprintf('COMPARE %-22s science=%s logShape=%s logText=%s\n', label, ...
        ternary(okSci,'IDENTICAL','*** DIFFERS ***'), ...
        ternary(okShape,'IDENTICAL','*** DIFFERS ***'), ...
        ternary(okText,'identical','reworded'));
    if ~okSci
        anchorExplain(R.rec, rec);
    end
end
end

function s = ternary(c,a,b), if c, s=a; else, s=b; end, end
