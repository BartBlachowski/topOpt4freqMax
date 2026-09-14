function B = imresizeNN(A, h, w)
%IMRESIZENN  Nearest-neighbour resize, no Image Processing Toolbox required.
ri = max(1, min(size(A,1), round(linspace(0.5, size(A,1)+0.5, h))));
ci = max(1, min(size(A,2), round(linspace(0.5, size(A,2)+0.5, w))));
B = A(ri, ci);
end
