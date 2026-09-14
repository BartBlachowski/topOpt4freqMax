function nFail = anchorReport()
%ANCHORREPORT  Compare every stored candidate anchor against its reference.
%
%   Recomputes both digests from the stored records, so the equality standard
%   can be re-evaluated without re-running a single solver iteration.
root = '/Users/piotrek/Programming/Matlab/Olhoff';
here = fullfile(root,'architecture','anchors');
addpath(fullfile(here,'code'));
L = {'A1_frozen160','A2_mature160','A3_r1ladder160','A4_nodescent160', ...
     'A5_pcont160','A6_pdecoupled160','A7_massp160','A8_projidentity160', ...
     'A9_projection160','A10_binarydiag160','A11_maxnorm160','A12_rhovar160'};
nFail = 0;
fprintf('%-22s %6s %7s %-14s %-18s %-10s %-9s\n', ...
    'anchor','outer','inner','status','science','logShape','logText');
fprintf('%s\n', repmat('-',1,92));
for k = 1:numel(L)
    R = load(fullfile(here,'reference',[L{k} '.mat']));
    C = load(fullfile(here,'candidate',[L{k} '.mat']));
    dr = anchorDigests(R.rec);  dc = anchorDigests(C.rec);
    okS = strcmp(dr.science, dc.science);
    okH = strcmp(dr.logShape, dc.logShape);
    okT = strcmp(dr.presentation, dc.presentation);
    if ~(okS && okH), nFail = nFail + 1; end
    fprintf('%-22s %6d %7d %-14s %-18s %-10s %-9s\n', L{k}, ...
        C.rec.nOuter, C.rec.innerTotal, C.rec.status, ...
        tern(okS,'BITWISE IDENTICAL','*** DIFFERS ***'), ...
        tern(okH,'identical','*** DIFFERS ***'), ...
        tern(okT,'identical','reworded'));
    if ~okS, anchorExplain(R.rec, C.rec); end
end
fprintf('%s\n', repmat('-',1,92));
fprintf('anchors failing the equality standard: %d of %d\n', nFail, numel(L));
end
function s = tern(c,a,b), if c, s=a; else, s=b; end, end
