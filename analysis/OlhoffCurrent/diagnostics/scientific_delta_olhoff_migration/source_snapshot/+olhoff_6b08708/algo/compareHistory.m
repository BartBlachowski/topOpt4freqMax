function compareHistory(res, outfile, label, xmax)
%COMPAREHISTORY  Stack our iteration history above the paper's printed figure.
if nargin<4, xmax = 80; end
tmp = [tempname '.png'];
plotHistory(res, tmp, '', xmax);
A = imread(tmp); if ndims(A)==3, A = uint8(mean(double(A),3)); end
P = imread('docs/figs/paper_fig4_hist.png'); if ndims(P)==3, P = uint8(mean(double(P),3)); end
W = 1100;
A = imresizeNN(A, round(size(A,1)*W/size(A,2)), W);
P = imresizeNN(P, round(size(P,1)*W/size(P,2)), W);
band = 34; pad = @(h) 255*ones(h,W,'uint8');
canvas = [pad(band); A; pad(10); pad(band); P; pad(10)];
canvas = textStamp(canvas, 6, 8, label, 3);
canvas = textStamp(canvas, band+size(A,1)+10+6, 8, 'PAPER FIG 4 DU AND OLHOFF', 3);
imwrite(canvas, outfile);
delete(tmp);
fprintf('wrote %s\n', outfile);
end
