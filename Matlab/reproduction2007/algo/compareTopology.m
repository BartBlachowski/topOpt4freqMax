function compareTopology(rho, nelx, nely, paperPng, outfile, ourLabel, paperLabel)
%COMPARETOPOLOGY  Stack the reproduced topology directly above the paper's
%   printed figure at matched width, so the designs are compared as IMAGES and
%   not only through frequency numbers.
%
%   The paper crop still carries its support symbols and domain outline; only
%   width is matched -- no pixel-for-pixel registration is attempted.

if nargin < 6 || isempty(ourLabel),   ourLabel   = 'REPRODUCTION'; end
if nargin < 7 || isempty(paperLabel), paperLabel = 'PAPER DU AND OLHOFF 2007'; end

ours = topologyImage(rho, nelx, nely, 8);
P = imread(paperPng);
if ndims(P) == 3, P = uint8(mean(double(P),3)); end

W = 1600;
ours = imresizeNN(ours, max(1,round(size(ours,1)*W/size(ours,2))), W);
P    = imresizeNN(P,    max(1,round(size(P,1)   *W/size(P,2))),    W);

band = 34;
pad  = @(h) 255*ones(h, W, 'uint8');
canvas = [pad(band); ours; pad(12); pad(band); P; pad(12)];

canvas = textStamp(canvas, 6, 8, ourLabel, 3);
canvas = textStamp(canvas, band+size(ours,1)+12+6, 8, paperLabel, 3);
imwrite(canvas, outfile);
fprintf('wrote %s\n', outfile);
end
