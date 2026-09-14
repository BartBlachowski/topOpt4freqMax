function R = dr_prefix(RHOnew, Pnew, refFile, refVar, nCmp, tag)
%DR_PREFIX  Phase D common-prefix reproduction check.
%   Bitwise identity is the preferred standard; the fallback standard
%   (event structure + archived precision) is evaluated and reported too.

S = load(refFile);
RHOref = S.RHO;
n = min([nCmp, size(RHOnew,2), size(RHOref,2)]);

dmax = 0; firstDiff = NaN;
for k = 1:n
    d = max(abs(RHOnew(:,k)-RHOref(:,k)));
    if d > 0 && isnan(firstDiff), firstDiff = k; end
    dmax = max(dmax,d);
end
bitwise = (dmax == 0);

% archived scalar comparison
pr = S.out.per;
om1d = max(abs(Pnew.omega1(1:n) - pr.omega1(1:n)));
mndd = max(abs(Pnew.Mnd(1:n)    - pr.Mnd(1:n)));
vold = max(abs(Pnew.volume(1:n) - pr.volume(1:n)));
mvd  = max(abs(Pnew.move(1:n)   - pr.move(1:n)));
evNew = find(Pnew.move(2:n) ~= Pnew.move(1:n-1))+1;
evRef = find(pr.move(2:n)   ~= pr.move(1:n-1))+1;

R = struct('tag',tag,'refFile',refFile,'nCompared',n, ...
    'bitwise',bitwise,'maxAbsRhoDiff',dmax,'firstDifferingIter',firstDiff, ...
    'maxAbsOmega1Diff',om1d,'maxAbsMndDiff',mndd,'maxAbsVolumeDiff',vold, ...
    'maxAbsMoveDiff',mvd,'moveEventsNew',evNew(:).','moveEventsRef',evRef(:).', ...
    'eventStructureIdentical',isequal(evNew(:).',evRef(:).'));
fprintf('[prefix %s] n=%d bitwise=%d maxRhoDiff=%.3g  omega1=%.3g Mnd=%.3g vol=%.3g events=%d\n', ...
    tag, n, bitwise, dmax, om1d, mndd, vold, R.eventStructureIdentical);
end
