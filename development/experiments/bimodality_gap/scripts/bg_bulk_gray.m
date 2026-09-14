function bg_bulk_gray()
%BG_BULK_GRAY  What the bulk-gray elements of P designs are made of (read-only on data/rho_*.mat).
%   bulk gray = 0.1<rho<0.9 and NOT within one filter radius of both a solid (>=0.9) and a void (<=0.1)
%   element.  Classified as
%     coreless member : void within R, no solid within R  (a member with no solid core)
%     gray plateau    : no void within R                   (gray inside or next to solid)
%   and the physical width of coreless members is estimated as 2*(max distance to void) over each
%   connected component of coreless elements.
here = fileparts(mfilename('fullpath')); data = fullfile(fileparts(here), 'data'); addpath(here);
M9 = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100]; M4 = M9(1:4,:);
cases = {};
for i = 1:9, cases(end+1,:) = {'P', M9(i,1), M9(i,2), 0.06}; end %#ok<AGROW>
for i = 1:4, cases(end+1,:) = {'R012', M4(i,1), M4(i,2), 0.12}; end %#ok<AGROW>
for i = 1:4, cases(end+1,:) = {'filterEl3', M4(i,1), M4(i,2), 3/M4(i,2)}; end %#ok<AGROW>
for m = [160 20; 240 30; 320 40; 400 50; 640 80; 800 100]', cases(end+1,:) = {'budget400', m(1), m(2), 0.06}; end %#ok<AGROW>
rows = {};
for c = 1:size(cases,1)
    S = load(fullfile(data, sprintf('rho_%s_%dx%d.mat', cases{c,1}, cases{c,2}, cases{c,3})));
    nelx = cases{c,2}; nely = cases{c,3}; h = 1/nely; Rphys = cases{c,4}; rc = ceil(Rphys*nely);
    R = reshape(S.rho, nely, nelx); NE = nelx*nely;
    G = R > 0.1 & R < 0.9; dS = bg_chebdist(R >= 0.9, 20); dV = bg_chebdist(R <= 0.1, 20);
    band = G & dS <= rc & dV <= rc; bulk = G & ~band;
    coreless = bulk & dV <= rc & dS > rc; plateau = bulk & dV > rc;
    q = 4*R.*(1-R);
    % connected components of coreless elements (4-neighbour flood fill, no toolbox)
    lab = zeros(size(R)); nc = 0; widths = [];
    [ii, jj] = find(coreless);
    for t = 1:numel(ii)
        if lab(ii(t), jj(t)) > 0, continue; end
        nc = nc + 1; stack = [ii(t) jj(t)]; lab(ii(t), jj(t)) = nc; members = [];
        while ~isempty(stack)
            p = stack(end, :); stack(end, :) = []; members(end+1, :) = p; %#ok<AGROW>
            for d = [0 1; 1 0; 0 -1; -1 0]'
                q2 = p + d';
                if all(q2 >= 1) && q2(1) <= nely && q2(2) <= nelx && coreless(q2(1), q2(2)) && lab(q2(1), q2(2)) == 0
                    lab(q2(1), q2(2)) = nc; stack(end+1, :) = q2; %#ok<AGROW>
                end
            end
        end
        if size(members, 1) >= 4
            dv = dV(sub2ind(size(R), members(:,1), members(:,2)));
            widths(end+1) = (2*max(dv) + 1) * h; %#ok<AGROW>
        end
    end
    r = struct('case', sprintf('%s_%dx%d', cases{c,1}, nelx, nely), 'arm', cases{c,1}, 'nelx', nelx, 'nely', nely, 'Rphys', Rphys, 'R_over_h', Rphys*nely, 'Mnd', mean(q(:)), ...
        'Mnd_bulk', sum(q(bulk))/NE, 'Mnd_coreless', sum(q(coreless))/NE, 'Mnd_plateau', sum(q(plateau))/NE, ...
        'frac_bulk', mean(bulk(:)), 'frac_coreless', mean(coreless(:)), 'n_coreless_components', numel(widths), ...
        'coreless_width_median', median([widths NaN], 'omitnan'), 'coreless_width_max', max([widths 0]), 'coreless_width_median_over_R', median([widths NaN], 'omitnan')/Rphys, ...
        'mean_rho_coreless', mean(R(coreless)));
    rows{end+1} = r; %#ok<AGROW>
    fprintf('R/h=%.1f %-20s Mnd=%.4f bulk=%.4f coreless=%.4f plateau=%.4f ncomp=%d width med=%.3f max=%.3f mean rho=%.2f\n', r.R_over_h, r.case, r.Mnd, r.Mnd_bulk, r.Mnd_coreless, r.Mnd_plateau, r.n_coreless_components, r.coreless_width_median, r.coreless_width_max, r.mean_rho_coreless);
end
bg_write_rows(rows, fullfile(data, 'bulk_gray_classification.csv'));
end
