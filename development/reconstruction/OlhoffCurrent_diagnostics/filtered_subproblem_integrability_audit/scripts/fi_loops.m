function out = fi_loops()
%FI_LOOPS  Part 6: closed-loop line integrals of g_filt, with the mandatory
%   g_phys positive control.
%
%   Rectangle  rho -> rho+a*u -> rho+a*u+b*v -> rho+b*v -> rho, each edge by
%   8-point Gauss-Legendre.  For a conservative field the loop integral
%   vanishes; by Stokes, for a small rectangle it tends to -a*b*(u'Jv - v'Ju).
%
%   No density is updated: every quadrature point is a temporary evaluation.

S = fi_setup();
study = fileparts(fileparts(mfilename('fullpath')));
D = fi_directions(S);
rho = S.rho386(:);

amps = [1e-3 3e-4 1e-4];                      % preregistered
[xg, wg] = local_gauss8();

sym = jsondecode(fileread(fullfile(study,'evaluations','jacobian_symmetry.json')));
res = struct('pair',{},'amp',{},'loop_phys',{},'loop_filt',{}, ...
             'rel_filt',{},'rel_phys',{},'admissible',{},'uJv_filt',{},'vJu_filt',{});
nev = 0; t0 = tic;

for i = 1:size(D.pairs,1)
    un = D.pairs{i,1}; vn = D.pairs{i,2};
    u = D.(un); v = D.(vn);
    pname = sprintf('%s,%s',un,vn);
    % Stokes reference from Part 5 at the matching delta (1e-3)
    k = find(strcmp({sym.results.pair}, pname) & [sym.results.delta] == 1e-3, 1);
    uJv = sym.results(k).uJv_filt; vJu = sym.results(k).vJu_filt;
    scaleF = max(abs(uJv), abs(vJu));
    kP = k; scaleP = max(abs(sym.results(kP).uJv_phys), abs(sym.results(kP).vJu_phys));

    for aI = 1:numel(amps)
        a = amps(aI); b = a;
        corner = rho + a*u + b*v;
        adm = all(corner >= S.rhomin - 1e-15) && all(corner <= 1 + 1e-15);
        [Lp, Lf, ne] = local_loop(S, rho, u, v, a, b, xg, wg);
        nev = nev + ne;
        q = numel(res)+1;
        res(q).pair = pname;      res(q).amp = a;
        res(q).loop_phys = Lp;    res(q).loop_filt = Lf;
        res(q).rel_filt = abs(Lf)/max(a*b*scaleF, realmin);
        res(q).rel_phys = abs(Lp)/max(a*b*scaleP, realmin);
        res(q).admissible = adm;
        res(q).uJv_filt = uJv;    res(q).vJu_filt = vJu;
    end
end

% ---- explicit orientation-reversal check on one pair --------------------
u = D.(D.pairs{3,1}); v = D.(D.pairs{3,2}); a = 1e-3;
[~, Lf_fwd, n1] = local_loop(S, rho, u, v, a, a, xg, wg);
[~, Lf_rev, n2] = local_loop(S, rho, v, u, a, a, xg, wg);   % reversed orientation
nev = nev + n1 + n2;

out = struct();
out.amps = amps;
out.results = res;
out.n_evaluations = nev;
out.wall_s = toc(t0);
out.reversal = struct('pair', sprintf('%s,%s',D.pairs{3,1},D.pairs{3,2}), ...
    'forward', Lf_fwd, 'reversed', Lf_rev, 'sum', Lf_fwd + Lf_rev, ...
    'relative_sum', abs(Lf_fwd + Lf_rev)/max(abs(Lf_fwd), realmin));

% ---- amplitude scaling: a genuine curl gives |loop| ~ a^2 ---------------
sc = struct('pair',{},'exponent_filt',{},'exponent_phys',{});
for i = 1:size(D.pairs,1)
    pname = sprintf('%s,%s',D.pairs{i,1},D.pairs{i,2});
    sel = strcmp({res.pair}, pname);
    A = [res(sel).amp]; Lf = abs([res(sel).loop_filt]); Lp = abs([res(sel).loop_phys]);
    pf = polyfit(log(A), log(max(Lf,realmin)), 1);
    pp = polyfit(log(A), log(max(Lp,realmin)), 1);
    q = numel(sc)+1; sc(q).pair = pname;
    sc(q).exponent_filt = pf(1); sc(q).exponent_phys = pp(1);
end
out.scaling = sc;

med = zeros(numel(amps),3);
for t = 1:numel(amps)
    sel = [res.amp] == amps(t);
    med(t,:) = [amps(t), median([res(sel).rel_filt]), median([res(sel).rel_phys])];
end
out.median_cols = {'amp','median_rel_filt','median_rel_phys'};
out.median_by_amp = med;
out.median_exponent_filt = median([sc.exponent_filt]);
out.median_exponent_phys = median([sc.exponent_phys]);
out.all_admissible = all([res.admissible]);

f = fullfile(study,'evaluations','closed_loops.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);

fprintf('\n[fi_loops] %d evaluations, %.1f s, all corners admissible = %d\n', ...
    nev, out.wall_s, out.all_admissible);
fprintf('  %-10s %-18s %-18s\n','amp','median rel filt','median rel phys');
for t = 1:numel(amps)
    fprintf('  %-10.3g %-18.4e %-18.4e\n', med(t,1), med(t,2), med(t,3));
end
fprintf('  amplitude exponent: filtered median = %.3f   physical median = %.3f\n', ...
    out.median_exponent_filt, out.median_exponent_phys);
fprintf('  reversal: forward=%.6e reversed=%.6e sum=%.3e relative=%.3e\n', ...
    Lf_fwd, Lf_rev, Lf_fwd+Lf_rev, out.reversal.relative_sum);
fprintf('\n  per-pair |loop| (filtered) at the three amplitudes, and exponent:\n');
for i = 1:size(D.pairs,1)
    pname = sprintf('%s,%s',D.pairs{i,1},D.pairs{i,2});
    sel = find(strcmp({res.pair}, pname));
    fprintf('  %-14s', pname);
    for q = sel, fprintf(' %10.3e', abs(res(q).loop_filt)); end
    fprintf('   exp=%5.2f   rel=%8.2e\n', sc(i).exponent_filt, res(sel(1)).rel_filt);
end
end

% =========================================================================
function [Lp, Lf, nev] = local_loop(S, rho, u, v, a, b, xg, wg)
Lp = 0; Lf = 0; nev = 0;
edges = { @(t) rho + (t*a)*u,             a*u;
          @(t) rho + a*u + (t*b)*v,       b*v;
          @(t) rho + ((1-t)*a)*u + b*v,  -a*u;
          @(t) rho + ((1-t)*b)*v,        -b*v };
for e = 1:4
    pathFn = edges{e,1}; dvec = edges{e,2};
    for q = 1:numel(xg)
        E = fi_eval(S, pathFn(xg(q)), 'mode1');
        nev = nev + 1;
        Lp = Lp + wg(q)*(dvec.'*E.gPhys);
        Lf = Lf + wg(q)*(dvec.'*E.gFilt);
    end
end
end

function [x, w] = local_gauss8()
% 8-point Gauss-Legendre mapped to [0,1]
xs = [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, ...
      -0.1834346424956498,  0.1834346424956498,  0.5255324099163290, ...
       0.7966664774136267,  0.9602898564975363];
ws = [ 0.1012285362903763,  0.2223810344533745,  0.3137066458778873, ...
       0.3626837833783620,  0.3626837833783620,  0.3137066458778873, ...
       0.2223810344533745,  0.1012285362903763];
x = 0.5*(xs+1); w = 0.5*ws;
end
