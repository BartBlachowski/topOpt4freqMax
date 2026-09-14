function [R, A] = ar_collect()
%AR_COLLECT  Load unstopped trajectories, verify the baseline anchor, evaluate
%   the preregistered candidates, write METRICS.json / CSVs / figures.
repo = '/Users/piotrek/Programming/topOpt4freqMax';
base = fullfile(repo,'analysis','OlhoffCurrent','diagnostics','admission_rule');
addpath(fullfile(base,'code')); addpath(fullfile(repo,'analysis','OlhoffCurrent'));
guard = olhoffcurrent_paths(); %#ok<NASGU>

meshes = [160 20; 320 40];
R = [];
for i = 1:size(meshes,1)
    f = fullfile(base,'runs', sprintf('unstopped_%dx%d.mat', meshes(i,1), meshes(i,2)));
    if exist(f,'file')~=2; fprintf('  MISSING %s\n', f); continue; end
    S = load(f,'out'); if isempty(R); R = S.out; else; R(end+1)=S.out; end %#ok<AGROW>
end
fprintf('loaded %d unstopped trajectory(ies)\n', numel(R));

cands = ar_candidates();
A = struct('mesh',{},'NE',{},'prodStop',{},'prodTol',{},'anchor',{},'cands',{});

% archived conference record (the ultimate anchor)
Z = load(fullfile(repo,'examples','Performance','conference_benchmark', ...
                  'campaign_9mesh_r2','benchmark_records.mat'));
rec = Z.records(strcmp({Z.records.method_key},'olhoff'));

for i = 1:numel(R)
    r = R(i); P = r.per; nelx=r.mesh(1); nely=r.mesh(2);

    % ---- reproduce the PRODUCTION admission on this trajectory ----------
    % eps is the production value (r.prodTol); settledMove requires the level
    % unchanged from the previous iteration.
    settled = P.itersSinceMoveChange >= 1;
    prodAdmit = (P.l2 < r.prodTol) & settled;
    kProd = find(prodAdmit,1);

    % ---- BASELINE GATE: does the trajectory reproduce the archive? ------
    j = find(arrayfun(@(s) isequal(s.mesh(:).',[nelx nely]), rec),1);
    anchor = struct('archivedOuter',NaN,'archivedOmega1',NaN,'prodStopIter',kProd, ...
        'outerMatch',false,'omega1Bitwise',false,'rhoBitwise',false, ...
        'histBitwise',false,'histDetail','');
    if ~isempty(j)
        anchor.archivedOuter  = double(rec(j).counts.outer_iterations);
        anchor.archivedOmega1 = double(rec(j).omega(1));
        anchor.outerMatch     = isequal(kProd, anchor.archivedOuter);
        anchor.omega1Bitwise  = isequaln(P.omega1(kProd), anchor.archivedOmega1);
        Sr = load(fullfile(base,'runs',sprintf('unstopped_%dx%d.mat',nelx,nely)),'RHO');
        anchor.rhoBitwise     = isequaln(Sr.RHO(:,kProd), double(rec(j).x(:)));
    end
    % per-iteration bitwise vs the previous diagnostic's baseline run
    bf = fullfile(repo,'analysis','OlhoffCurrent','diagnostics','move_stop','runs', ...
                  sprintf('baseline_%dx%d.mat',nelx,nely));
    if exist(bf,'file')==2
        B = load(bf,'out'); bp = B.out.per; nb = B.out.nOuter;
        flds = {'omega1','omega2','gap12','volume','move','stage','beta','l2', ...
                'maxAbs','nInner','multN'};
        bad = {};
        for q = 1:numel(flds)
            if ~isequaln(P.(flds{q})(1:nb), bp.(flds{q})(1:nb)); bad{end+1}=flds{q}; end %#ok<AGROW>
        end
        anchor.histBitwise = isempty(bad);
        anchor.histDetail  = strjoin(bad,', ');
        anchor.baselineOuter = nb;
    end

    % ---- candidates ------------------------------------------------------
    CE = struct('name',{},'role',{},'status',{},'stopIter',{},'gapToLastDescent',{}, ...
        'lastDescentBefore',{},'state',{},'objRelRangeAtStop',{});
    for c = 1:numel(cands)
        E = ar_predicate(P, cands(c));
        st = ar_state_at(P, E.stopIter, cands(c).name);
        orr = NaN; if ~isempty(E.stopIter); orr = E.objRelRange(E.stopIter); end
        CE(end+1) = struct('name',cands(c).name,'role',cands(c).role, ...
            'status',E.status,'stopIter',ternEmpty(E.stopIter),'gapToLastDescent',E.gapToLastDescent, ...
            'lastDescentBefore',E.lastDescentBefore,'state',st,'objRelRangeAtStop',orr); %#ok<AGROW>
    end

    A(end+1) = struct('mesh',[nelx nely],'NE',r.NE, ...
        'prodStop', ar_state_at(P,kProd,'production'), 'prodTol', r.prodTol, ...
        'anchor', anchor, 'cands', CE); %#ok<AGROW>
end
end

function v = ternEmpty(x)
if isempty(x); v = NaN; else; v = x; end
end
