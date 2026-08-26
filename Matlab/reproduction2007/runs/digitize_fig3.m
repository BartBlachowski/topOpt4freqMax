function [rho, diag] = digitize_fig3(pngfile, nelx, nely, verbose)
%DIGITIZE_FIG3  Read a printed topology out of a figure crop and resample it
%   onto our element grid, with the DOMAIN OUTLINE ERASED.
%
%   The design domain is drawn as a thin rectangle.  Its outer bounds are the
%   outermost rows/cols that are dark along most of their length.  The pen width
%   is measured from the row profile: the outline rows are dark across exactly
%   the domain width, i.e. their coverage equals the domain's width fraction of
%   the crop, whereas structure rows are always less.
%
%   The outline is then ERASED by inpainting each frame band from the first
%   interior row/column.  That is the right rule for a drawn boundary: where the
%   structure touches the frame the neighbour is solid and the band stays solid;
%   where it does not, the neighbour is void and the band becomes void.  No
%   structure is invented and none is lost.
%
%   The element grid is mapped to the outline's CENTRE-LINE rectangle, which is
%   the actual design domain (the pen straddles the boundary).

if nargin<4, verbose = false; end
I = imread(pngfile);
if ndims(I)==3, I = uint8(mean(double(I),3)); end
BW = I < 128;
[H,W] = size(BW);

rowFrac = mean(BW,2);
cand = find(rowFrac > 0.5);
if numel(cand)<2, error('digitize_fig3:rows','domain outline not found'); end
r1 = min(cand); r2 = max(cand);
sub = BW(r1:r2,:);
colFrac = mean(sub,1);
candc = find(colFrac > 0.5);
c1 = min(candc); c2 = max(candc);

% pen width: consecutive rows at the top/bottom whose coverage equals the
% domain width fraction (to within 2%)
dwf = (c2-c1+1)/W;
tTop = 0; while r1+tTop <= r2 && rowFrac(r1+tTop) >= 0.98*dwf, tTop = tTop+1; end
tBot = 0; while r2-tBot >= r1 && rowFrac(r2-tBot) >= 0.98*dwf, tBot = tBot+1; end
t = max([tTop tBot 1]);

% ---- ERASE the outline by inpainting from the first interior line ----------
E = BW;
if r1+t <= r2-t && c1+t <= c2-t
    E(r1:r1+t-1, :) = repmat(BW(r1+t,   :), t, 1);
    E(r2-t+1:r2, :) = repmat(BW(r2-t,   :), t, 1);
    E(:, c1:c1+t-1) = repmat(E(:, c1+t),  1, t);
    E(:, c2-t+1:c2) = repmat(E(:, c2-t),  1, t);
end

% ---- domain = outline centre-line rectangle -------------------------------
rTop = round(r1 + (t-1)/2);  rBot = round(r2 - (t-1)/2);
cLef = round(c1 + (t-1)/2);  cRig = round(c2 - (t-1)/2);
R = E(rTop:rBot, cLef:cRig);

[hh,ww] = size(R);
rowEdges = round(linspace(1,hh+1,nely+1));
colEdges = round(linspace(1,ww+1,nelx+1));
X = zeros(nely,nelx);
for j = 1:nelx
    for i = 1:nely
        blk = R(rowEdges(i):rowEdges(i+1)-1, colEdges(j):colEdges(j+1)-1);
        X(i,j) = mean(blk(:));
    end
end
rho = X(:);

diag = struct('r1',r1,'r2',r2,'c1',c1,'c2',c2,'pen',t, ...
              'aspect',(cRig-cLef+1)/(rBot-rTop+1),'vf',mean(rho), ...
              'erased',E(rTop:rBot, cLef:cRig));
if verbose
    fprintf(['crop %dx%d | outline rows %d..%d cols %d..%d | pen %d px | ' ...
             'domain aspect %.3f (exact 8) | digitized vf %.4f\n'], ...
            H,W,r1,r2,c1,c2,t,diag.aspect,diag.vf);
end
end
