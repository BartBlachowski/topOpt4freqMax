function out = scan_persistence(passMatrix, P)
%SCAN_PERSISTENCE Find earliest complete all-pass window without look-ahead.
arguments
    passMatrix (:,:) logical
    P (1,1) double {mustBeInteger,mustBePositive}
end
n=size(passMatrix,1); m=size(passMatrix,2);
out=struct('k_enter',nan(1,m),'k_cert',nan(1,m),'instantaneous_first',nan(1,m), ...
    'P',P,'n_observed',n,'tail_incomplete',false(1,m));
for j=1:m
    a=find(passMatrix(:,j),1,'first'); if ~isempty(a), out.instantaneous_first(j)=a; end
    run=0;
    for k=1:n
        if passMatrix(k,j), run=run+1; else, run=0; end
        if run==P
            out.k_cert(j)=k; out.k_enter(j)=k-P+1; break
        end
    end
    if isnan(out.k_enter(j))
        suffix=0; for k=n:-1:1, if passMatrix(k,j), suffix=suffix+1; else, break; end, end
        out.tail_incomplete(j)=suffix>0 && suffix<P;
    end
end
end
