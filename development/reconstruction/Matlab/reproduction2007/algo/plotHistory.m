function plotHistory(res, outfile, ttl, xmax)
%PLOTHISTORY  Reproduce the paper's iteration-history figure (2007 Fig. 4a /
%   2014 Fig. 4) in the same style: omega_1 black circles, omega_2 blue '+',
%   omega_3 red triangles, y from 0 to 600, x = iteration number.
h = res.hist;
n = size(h.omega,2);
if nargin<4 || isempty(xmax), xmax = 80; end
if nargin<3 || isempty(ttl),  ttl = ''; end
it = 1:n;

f = figure('Visible','off','Color','w','Position',[100 100 900 760]);
ax = axes(f); hold(ax,'on'); box(ax,'on');

plot(ax, it, h.omega(3,:), '-^', 'Color',[1 0 0], 'MarkerEdgeColor',[1 0 0], ...
     'MarkerFaceColor','none','MarkerSize',7,'LineWidth',1.0);
plot(ax, it, h.omega(2,:), '-+', 'Color',[0.15 0.15 0.6], ...
     'MarkerEdgeColor',[0.15 0.15 0.6],'MarkerSize',8,'LineWidth',1.0);
plot(ax, it, h.omega(1,:), '-o', 'Color',[0 0 0], 'MarkerEdgeColor',[0 0 0], ...
     'MarkerFaceColor','none','MarkerSize',7,'LineWidth',1.0);

xlim(ax,[0 xmax]); ylim(ax,[0 600]);
set(ax,'XTick',0:20:xmax,'YTick',0:100:600,'FontSize',22,'LineWidth',1.2);
xlabel(ax,'Iteration number','FontSize',26);
ylabel(ax,'Eigenfrequencies','FontSize',26);

% curve labels placed as in the paper
xl = 0.62*xmax;
text(ax, xl, h.omega(3,min(n,round(xl)))+45, '\omega_3','FontSize',26,'Color',[1 0 0]);
text(ax, xl, 205, '\omega_2','FontSize',26,'Color',[0.15 0.15 0.6]);
text(ax, 0.30*xmax, 130, '(Maximized) \omega_1','FontSize',24,'Color','k');
if ~isempty(ttl), title(ax, ttl, 'FontSize',18,'FontWeight','normal'); end

exportgraphics(f, outfile, 'Resolution', 150);
close(f);
fprintf('wrote %s\n', outfile);
end
