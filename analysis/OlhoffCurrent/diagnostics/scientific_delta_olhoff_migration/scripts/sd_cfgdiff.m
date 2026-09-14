function D = sd_cfgdiff(a, b)
%SD_CFGDIFF  Leaf-by-leaf difference of two configuration structs.
%   D is a struct array with fields path, a, b, kind ('diff','onlyA','onlyB').
La = sd_flatten(a); Lb = sd_flatten(b);
ka = La(:,1); kb = Lb(:,1);
D = struct('path',{},'a',{},'b',{},'kind',{});
allk = union(ka, kb, 'stable');
for i = 1:numel(allk)
    k = allk{i};
    ia = find(strcmp(ka,k),1); ib = find(strcmp(kb,k),1);
    if isempty(ia)
        D(end+1) = struct('path',k,'a',[],'b',{Lb{ib,2}},'kind','onlyB'); %#ok<AGROW>
    elseif isempty(ib)
        D(end+1) = struct('path',k,'a',{La{ia,2}},'b',[],'kind','onlyA'); %#ok<AGROW>
    elseif ~isequaln(La{ia,2}, Lb{ib,2})
        D(end+1) = struct('path',k,'a',{La{ia,2}},'b',{Lb{ib,2}},'kind','diff'); %#ok<AGROW>
    end
end
end
