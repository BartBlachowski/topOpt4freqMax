function diag_frame(pngfile)
I = imread(pngfile); if ndims(I)==3, I = uint8(mean(double(I),3)); end
BW = I < 128;
rowFrac = mean(BW,2);
cand = find(rowFrac > 0.5); r1 = min(cand); r2 = max(cand);
fprintf('rows %d..%d\n',r1,r2);
fprintf('rowFrac near TOP:\n');
for r = r1-2:r1+14, fprintf('  row %3d: %.3f\n', r, rowFrac(r)); end
fprintf('rowFrac near BOTTOM:\n');
for r = r2-14:r2+2, fprintf('  row %3d: %.3f\n', r, rowFrac(r)); end
sub = BW(r1:r2,:); colFrac = mean(sub,1);
candc = find(colFrac > 0.5); c1 = min(candc); c2 = max(candc);
fprintf('cols %d..%d\n',c1,c2);
fprintf('colFrac near LEFT:  '); fprintf('%.2f ',colFrac(c1-1:c1+12)); fprintf('\n');
fprintf('colFrac near RIGHT: '); fprintf('%.2f ',colFrac(c2-12:c2+1)); fprintf('\n');
end
