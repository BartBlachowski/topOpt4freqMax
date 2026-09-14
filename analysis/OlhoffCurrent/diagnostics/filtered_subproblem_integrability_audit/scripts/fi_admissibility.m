function out = fi_admissibility()
%FI_ADMISSIBILITY  Quantify how far, if at all, loop corners leave the design box.
S = fi_setup(); D = fi_directions(S); rho = S.rho386(:);
study = fileparts(fileparts(mfilename('fullpath')));
amps = [1e-3 3e-4 1e-4];
rows = struct('pair',{},'amp',{},'minRho',{},'maxRho',{},'belowBy',{},'aboveBy',{}, ...
              'nBelow',{},'nAbove',{},'admissible',{});
for i = 1:size(D.pairs,1)
    u = D.(D.pairs{i,1}); v = D.(D.pairs{i,2});
    for t = 1:numel(amps)
        a = amps(t);
        % worst corner over the four rectangle vertices
        C = [rho+a*u, rho+a*u+a*v, rho+a*v, rho];
        mn = min(C(:)); mx = max(C(:));
        q = numel(rows)+1;
        rows(q).pair = sprintf('%s,%s',D.pairs{i,1},D.pairs{i,2});
        rows(q).amp = a; rows(q).minRho = mn; rows(q).maxRho = mx;
        rows(q).belowBy = max(S.rhomin - mn, 0);
        rows(q).aboveBy = max(mx - 1, 0);
        rows(q).nBelow = sum(C(:) < S.rhomin);
        rows(q).nAbove = sum(C(:) > 1);
        rows(q).admissible = mn >= S.rhomin - 1e-15 && mx <= 1 + 1e-15;
    end
end
out.rows = rows;
out.n_inadmissible = sum(~[rows.admissible]);
out.max_below_by = max([rows.belowBy]);
out.max_above_by = max([rows.aboveBy]);
out.rhomin = S.rhomin;
out.max_relative_excursion = max([rows.belowBy])/S.rhomin;
out.worst = rows(find([rows.belowBy] == max([rows.belowBy]),1));
f = fullfile(study,'evaluations','loop_admissibility.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);
fprintf('[fi_admissibility] %d of %d loop configurations leave the box\n', ...
    out.n_inadmissible, numel(rows));
fprintf('  max excursion below rhomin=%.3g : %.3e  (=%.4f%% of rhomin)\n', ...
    S.rhomin, out.max_below_by, 100*out.max_relative_excursion);
fprintf('  max excursion above 1          : %.3e\n', out.max_above_by);
for q = 1:numel(rows)
    if ~rows(q).admissible
        fprintf('  %-14s a=%-8.3g minRho=%.6e (below by %.2e, %d elems)\n', ...
            rows(q).pair, rows(q).amp, rows(q).minRho, rows(q).belowBy, rows(q).nBelow);
    end
end
end
