function audit_compare(tags)
%AUDIT_COMPARE  Quantitative comparison of terminal designs across routes.
%   Writes a PNG per design and prints a pairwise distance table, so "same
%   physical basin?" is answered numerically rather than by eye.
repo='/Users/piotrek/Programming/topOpt4freqMax';
base=fullfile(repo,'analysis','OlhoffRegularized','audit','results');
R={};keep={};
for i=1:numel(tags)
    f=fullfile(base,tags{i},'run.mat');
    if exist(f,'file')~=2,fprintf('skip %s (no run.mat)\n',tags{i});continue,end
    L=load(f);R{end+1}=L; %#ok<AGROW>
    keep{end+1}=tags{i}; %#ok<AGROW>
    img=uint8(255*(1-reshape(L.rho,L.meta.nely,L.meta.nelx)));
    imwrite(flipud(img),fullfile(base,[tags{i} '_topology.png']));
end
fprintf('\n%-26s %10s %10s %10s %8s %8s\n','tag','omega1','omega2','omega3','gray','status');
for i=1:numel(R)
    fprintf('%-26s %10.5f %10.5f %10.5f %8.4f  %s\n',keep{i},R{i}.omega(1),R{i}.omega(2), ...
        R{i}.omega(3),4*mean(R{i}.rho.*(1-R{i}.rho)),R{i}.status);
end
fprintf('\npairwise terminal-density distance (same mesh only)\n');
fprintf('%-26s %-26s %10s %10s %10s\n','A','B','mean|dRho|','max|dRho|','corr');
for i=1:numel(R)
    for j=i+1:numel(R)
        if R{i}.meta.nelx~=R{j}.meta.nelx||R{i}.meta.nely~=R{j}.meta.nely,continue,end
        % never compare across boundary conditions: a different problem is not
        % a different basin of the same problem.
        if ~strcmpi(R{i}.meta.bcType,R{j}.meta.bcType),continue,end
        a=R{i}.rho(:);b=R{j}.rho(:);d=a-b;
        c=corr(a,b);
        fprintf('%-26s %-26s %10.4f %10.4f %10.5f\n',keep{i},keep{j},mean(abs(d)),max(abs(d)),c);
    end
end
end
