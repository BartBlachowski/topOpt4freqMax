function [met, r] = stop_stagnation_x(d, g, k, W, relTol, grayTol)
%STOP_STAGNATION_X  Windowed stagnation of the physical density field and of the grayness.
%
%   [met, r] = STOP_STAGNATION_X(d, g, k, W, relTol, grayTol) is the test of
%   stop criterion 'stagnation_x'.  It is 'stagnation' with the objective test
%   replaced by a test on the physical density field, the field that enters the
%   stiffness/mass interpolation and the FE analysis:
%
%     d(j)  relative L2 change between the designs analysed at iterations j-1
%           and j of the current stage,
%             d(j) = ||xPhys_j - xPhys_{j-1}||_2 / ||xPhys_{j-1}||_2,
%           NaN for j = 1 (the stage's starting design has no predecessor
%           within the stage)
%     g(j)  grayness 4*mean(xPhys_j.*(1-xPhys_j)) of the design analysed at j
%
%   The window is the same as for 'stagnation': the W+1 analysed designs
%   j = k-W..k, never reaching the stage's starting design (j = 1), so it is
%   complete only once k-W >= 2.  Its W+1 designs span W consecutive changes,
%   d(k-W+1..k), and
%
%     r   = [max(d(k-W+1:k)), max(g(k-W:k)) - min(g(k-W:k))]
%     met = r(1) < relTol && r(2) < grayTol          (both strict)
%
%   The largest change over the whole window is tested, not the last one.
%   r = [NaN NaN] and met = false while the window is incomplete.  Proposed
%   and Yuksel call this function; olhoffSolve carries an identical local copy
%   (local_stagnation_x) because analysis/Olhoff/+impl is self-contained.
%
%   See also TOPOPT_FREQ, TOP99NEO_INERTIAL_FREQ.

met = false;  r = [NaN NaN];
if k - W < 2, return; end
dw = d(k-W+1:k);  gw = g(k-W:k);
r = [max(dw), max(gw) - min(gw)];
met = r(1) < relTol && r(2) < grayTol;
end
