function nFail = test_mass()
%TEST_MASS  Mass interpolation: values, derivatives, and piecewise continuity.
%
%   Values are checked against the PRINTED equations, written out here
%   independently of the implementation, so the test cannot pass by agreeing
%   with a shared mistake.

nFail = 0;
mk = @(m) struct('model',m,'q',1,'lowDensityExponent',6,'cutoff',0.1);
rho = [1e-6 1e-3 0.01 0.05 0.0999 0.1 0.1001 0.2 0.5 0.9 1].';

% ---- 1. values against the printed formulas ------------------------------
printed = containers.Map();
printed('eq2')  = @(r) r;                                             % eq. (2), q=1
printed('eq4')  = @(r) (r>0.1).*r + (r<=0.1).*r.^6;                   % eq. (4), r=6
printed('eq4a') = @(r) (r>0.1).*r + (r<=0.1).*(1e5*r.^6);             % eq. (4a), c0=1e5
printed('eq4b') = @(r) (r>0.1).*r + (r<=0.1).*(6e5*r.^6 - 5e6*r.^7);  % eq. (4b)
models = {'eq2','eq4','eq4a','eq4b'};
for k = 1:numel(models)
    m = models{k};
    g = olh.material.massInterpolation(rho, mk(m));
    f = printed(m);  want = f(rho);
    if ~isequal(typecast(g,'uint8'), typecast(want,'uint8'))
        fprintf('  FAIL mass value %s: max diff %.3e\n', m, max(abs(g-want)));  nFail = nFail+1;
    else
        fprintf('  ok   mass value %s (bitwise vs the printed equation)\n', m);
    end
end

% ---- 2. derivatives against central finite differences -------------------
for k = 1:numel(models)
    m = models{k};
    % away from the cut-off, where eq4 is genuinely discontinuous
    x = [0.02 0.05 0.08 0.2 0.5 0.9].';
    h = 1e-7;
    [~, dg] = olh.material.massInterpolation(x, mk(m));
    gp = olh.material.massInterpolation(x+h, mk(m));
    gm = olh.material.massInterpolation(x-h, mk(m));
    fd = (gp-gm)/(2*h);
    rel = abs(dg-fd)./max(abs(fd),1e-12);
    if max(rel) > 1e-5
        fprintf('  FAIL mass derivative %s: max rel err %.3e\n', m, max(rel));  nFail = nFail+1;
    else
        fprintf('  ok   mass derivative %s (max rel err %.2e vs central FD)\n', m, max(rel));
    end
end

% ---- 3. continuity at the cut-off, as each model CLAIMS -------------------
t = 0.1;  d = 1e-9;
claims = {  % model, C0 expected, C1 expected
  'eq2',  true,  true
  'eq4',  false, false     % the paper states (4) is discontinuous at rho=0.1
  'eq4a', true,  false     % c0 enforces C0 only
  'eq4b', true,  true      % c1,c2 ensure C1
};
for k = 1:size(claims,1)
    m = claims{k,1};
    [gL,dL] = olh.material.massInterpolation(t-d, mk(m));
    [gR,dR] = olh.material.massInterpolation(t+d, mk(m));
    isC0 = abs(gL-gR) < 1e-8;
    isC1 = isC0 && abs(dL-dR) < 1e-6;
    ok = (isC0 == claims{k,2}) && (isC1 == claims{k,3});
    if ok
        fprintf('  ok   mass continuity %-5s C0=%d C1=%d  (jump %.2e, slope jump %.2e)\n', ...
            m, isC0, isC1, abs(gL-gR), abs(dL-dR));
    else
        fprintf('  FAIL mass continuity %s: got C0=%d C1=%d, expected C0=%d C1=%d\n', ...
            m, isC0, isC1, claims{k,2}, claims{k,3});  nFail = nFail+1;
    end
end

% ---- 4. the printed constants are refused outside their printed regime ----
for m = {'eq4a','eq4b'}
    bad = mk(m{1});  bad.cutoff = 0.2;
    try
        olh.material.massInterpolation(rho, bad);
        fprintf('  FAIL %s accepted cutoff=0.2; the printed coefficients do not apply there\n', m{1});
        nFail = nFail + 1;
    catch e
        if strcmp(e.identifier,'olh:material:massConstants')
            fprintf('  ok   %s refuses a cut-off its printed coefficients do not fit\n', m{1});
        else
            fprintf('  FAIL %s wrong error id %s\n', m{1}, e.identifier);  nFail = nFail+1;
        end
    end
end

% ---- 5. the legacy name map still reaches the same numbers ---------------
pairs = {'lin','eq2';'4','eq4';'4a','eq4a';'4b','eq4b'};
for k = 1:size(pairs,1)
    [gA,dA] = massScale(rho, pairs{k,1});
    [gB,dB] = olh.material.massInterpolation(rho, mk(pairs{k,2}));
    if isequal(typecast(gA,'uint8'),typecast(gB,'uint8')) && ...
       isequal(typecast(dA,'uint8'),typecast(dB,'uint8'))
        fprintf('  ok   legacy name ''%s'' == %s (bitwise, value and derivative)\n', pairs{k,1}, pairs{k,2});
    else
        fprintf('  FAIL legacy name ''%s'' differs from %s\n', pairs{k,1}, pairs{k,2});  nFail = nFail+1;
    end
end
end
