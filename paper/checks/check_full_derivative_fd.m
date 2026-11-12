function T = check_full_derivative_fd()
%CHECK_FULL_DERIVATIVE_FD  Section 3.2: finite-difference check of the full derivative
%   of the compliance, Eq. (13), dc/dx = g + g^F, about 1 min.
%   Clamped beam 160x20 and building 40x120; initial, converged and random designs;
%   alpha = 1 and 0.5; central directional differences along a random direction with
%   steps 1e-3 ... 1e-7.  Also reports ||g^F|| / ||g|| on the converged designs.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
rows = {};
for prob = {problem_clamped_beam(160, 20), problem_building(40, 120)}
    p = prob{1};
    for alpha = [1 0.5]
        P = proposed_freq(p, struct('alpha', alpha));
        mdl = P.mdl;  b = P.phi0 .* (P.omega0'.^2);  w = [alpha; 1-alpha];
        rng(7);  xr = P.x;  xr(mdl.act) = 0.2 + 0.6*rand(numel(mdl.act), 1);
        x0 = p.volfrac*ones(mdl.nEl,1);  x0(mdl.pas) = 1;
        if ~isempty(mdl.pas), x0(mdl.act) = (p.volfrac*mdl.nEl - numel(mdl.pas))/numel(mdl.act); end
        D = {x0, P.x, xr};  names = {'initial', 'converged', 'random'};
        rng(11);  d = zeros(mdl.nEl,1);  d(mdl.act) = randn(numel(mdl.act),1);  d = d/norm(d);
        for id = 1:3
            [~, g, gF] = cfun(mdl, p, D{id}, w, b);
            an = (g + gF)'*d;
            for h = 10.^(-3:-1:-7)
                fd = (cfun(mdl, p, D{id} + h*d, w, b) - cfun(mdl, p, D{id} - h*d, w, b))/(2*h);
                rows(end+1, :) = {p.name, alpha, names{id}, h, abs(fd - an)/abs(an), norm(gF)/norm(g)}; %#ok<AGROW>
            end
        end
    end
end
T = cell2table(rows, 'VariableNames', {'case', 'alpha', 'design', 'h', 'rel_error', 'gF_over_g'});
S = T(T.h >= 1e-4, :);
fprintf('largest relative error for h = 1e-4 ... 1e-3: %.2e (all %d cases)\n', max(S.rel_error), height(S)/2);
C = T(strcmp(T.design, 'converged') & T.h == 1e-3, :);
fprintf('||g^F|| / ||g|| on converged designs: %.2f - %.2f\n', min(C.gF_over_g), max(C.gF_over_g));
end

function [c, g, gF] = cfun(mdl, p, x, w, b)
[K, M] = assemble_KM(mdl, x, p.E, p.Emin, p.rho, p.rhoMin);
f = mdl.free;  F = M*b;  U = zeros(mdl.ndof, 2);  U(f,:) = K(f,f)\F(f,:);
c = 0;  g = zeros(mdl.nEl,1);  gF = g;
dE = 3*x.^2*(p.E - p.Emin);  dR = (p.rho - p.rhoMin);
for l = 1:2
    if w(l) == 0, continue; end
    c = c + w(l)*(U(f,l)'*F(f,l));
    ue = reshape(U(mdl.edofMat, l), mdl.nEl, 8);  be = reshape(b(mdl.edofMat, l), mdl.nEl, 8);
    g  = g  - w(l)*dE.*sum((ue*mdl.KE).*ue, 2);
    gF = gF + w(l)*2*dR*sum((ue*mdl.ME).*be, 2);
end
g(mdl.pas) = 0;  gF(mdl.pas) = 0;
end
