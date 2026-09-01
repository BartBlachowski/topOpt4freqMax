function audit_main_table(tags)
%AUDIT_MAIN_TABLE  Emit the report's headline table rows from saved artifacts.
repo='/Users/piotrek/Programming/topOpt4freqMax';
base=fullfile(repo,'analysis','OlhoffRegularized','audit','results');
fprintf(['| Mesh | Route | Status | Outer | Accepted | CAP? | Native CONVERGED? | ' ...
    'Indep. stationarity certified? | max feasible ascent | omega1 |\n']);
fprintf('|---|---|---|---|---|---|---|---|---|---|\n');
for i=1:numel(tags)
    t=tags{i};f=fullfile(base,t,'run.mat');
    if exist(f,'file')~=2,fprintf('| *(%s: no run.mat)* |||||||||\n',t);continue,end
    L=load(f);m=L.meta;it=L.iterations;
    cap=strcmp(L.status,'CAP_HIT');conv=strcmp(L.status,'CONVERGED');
    asc='n/a';cert='n/a';
    sf=fullfile(base,t,'stationarity.mat');
    if exist(sf,'file')==2
        S=load(sf);S=S.S;
        a=S.selfConsistentAscent;
        asc=sprintf('%.2e',a);
        j=find([S.phys.N]==S.selfConsistentN & abs([S.phys.t]-1e-5)<1e-12,1);
        r=NaN;if ~isempty(j),r=S.phys(j).ratio;end
        if conv
            if a<=1e-5 && abs(r-1)<0.1
                cert=sprintf('**YES** (act/pred %.4f)',r);
            else
                cert=sprintf('**NO** (act/pred %.4f)',r);
            end
        else
            cert='n/a -- correctly refused';
        end
    end
    route=sprintf('%s + %s',localCap(m.formulation),upper(m.optimizer));
    fprintf('| %dx%d | %s | %s | %d | %d | %s | %s | %s | %s | %.6f |\n', ...
        m.nelx,m.nely,route,L.status,it.outer,it.accepted_updates,localYN(cap), ...
        localYN(conv),cert,asc,L.omega(1));
end
end
function s=localCap(x),s=[upper(x(1)) x(2:end)];end
function s=localYN(t),if t,s='yes';else,s='no';end,end
