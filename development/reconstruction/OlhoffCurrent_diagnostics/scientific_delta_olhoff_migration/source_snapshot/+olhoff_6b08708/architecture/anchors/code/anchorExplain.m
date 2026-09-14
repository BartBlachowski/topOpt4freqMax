function anchorExplain(ref, got)
%ANCHOREXPLAIN  Report the first field at which two anchor records diverge.
f = union(fieldnames(ref), fieldnames(got));
for k = 1:numel(f)
    n = f{k};
    if ~isfield(ref,n), fprintf('  ONLY-IN-CANDIDATE %s\n',n); continue; end
    if ~isfield(got,n), fprintf('  MISSING-IN-CANDIDATE %s\n',n); continue; end
    a = ref.(n); b = got.(n);
    if isequaln(a,b), continue; end
    if isnumeric(a) && isnumeric(b) && isequal(size(a),size(b))
        d = max(abs(double(a(:))-double(b(:))));
        fprintf('  DIFFERS %-16s maxabs=%.3e  first idx=%d\n', n, d, ...
            find(double(a(:))~=double(b(:)),1));
    else
        fprintf('  DIFFERS %-16s (non-numeric or shape change)\n', n);
    end
end
end
