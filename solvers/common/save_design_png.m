function save_design_png(prob, x, file, titleText)
%SAVE_DESIGN_PNG  Grey-scale density plot of a design (black x = 1, white x = 0).
%   x in the common numbering (fe_mesh).  Optional title.
fig = figure('Visible', 'off', 'Color', 'w');
ax = axes(fig);
imagesc(ax, [0 prob.L], [0 prob.H], 1 - reshape(x(:), prob.nely, prob.nelx));
set(ax, 'YDir', 'normal');  colormap(ax, gray);  caxis(ax, [0 1]);  axis(ax, 'image', 'off');
if nargin >= 4 && ~isempty(titleText), title(ax, titleText, 'FontWeight', 'normal'); end
exportgraphics(fig, file, 'Resolution', 200);
close(fig);
end
