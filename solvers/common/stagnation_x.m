function [met, r] = stagnation_x(d, g, k, W, relTol, grayTol)
%STAGNATION_X  Common stopping criterion of the paper (Section 2.5, Eq. (12)).
%   d(j): relative change ||x_j - x_{j-1}|| / ||x_{j-1}|| of the analysed design,
%   g(j): non-discreteness 4 mean(x_j (1 - x_j)).  Met at iteration k when over the
%   last W changes max d < relTol and the range of g over the last W+1 designs is
%   below grayTol; the initial design is never in the window.
met = false;  r = [NaN NaN];
if k - W < 2, return; end
dw = d(k-W+1:k);  gw = g(k-W:k);
r = [max(dw), max(gw) - min(gw)];
met = r(1) < relTol && r(2) < grayTol;
end
