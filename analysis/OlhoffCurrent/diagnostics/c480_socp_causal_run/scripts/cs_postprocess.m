function out = cs_postprocess(res, cfg, tag, evDir, runDir, meta)
%CS_POSTPROCESS  Durable evidence FIRST, then CSVs and the scalar record.
%   Shared by the treatment runner and the toy smoke test so the exact save
%   path used by the one treatment run has been exercised before launch.
%   Reads res only; writes no scientific state back.
if ~isfolder(evDir), mkdir(evDir); end
if ~isfolder(runDir), mkdir(runDir); end
NE = numel(res.rho);
n = numel(res.hist.N);
rho0 = cfg.design.initial*ones(NE,1);
rhomin = cfg.design.minimum;
RHO = zeros(NE, n); DRHO = zeros(NE, n);
r = rho0;
for k = 1:n
    d = res.diag.drho{k};
    DRHO(:,k) = d;
    r = min(1, max(rhomin, r + d));
    RHO(:,k) = r;
end
out = struct('tag',tag,'NE',NE,'nOuter',n,'status',res.status,'termination',res.termination);
out.rebuildExact = (n == 0) || isequal(RHO(:,end), res.rho);
if n > 0
    out.clampDisplacementMax = max(max(abs(diff([rho0 RHO],1,2) - DRHO)));
else
    out.clampDisplacementMax = NaN;
end

hist = res.hist; log = res.log; move = hist.move(:); socp = res.socp; %#ok<NASGU>
if isfield(res,'exhaustion'), exh = res.exhaustion; else, exh = struct(); end %#ok<NASGU>
omegaFinal = res.omega; lambdaFinal = res.lambda; status = res.status; %#ok<NASGU>
termination = res.termination; treat = res.treat; dg = res.diag; %#ok<NASGU>
trajFile = fullfile(evDir, sprintf('%s_trajectory.mat', tag));
save(trajFile, 'RHO','DRHO','move','hist','cfg','meta','exh','log','socp','omegaFinal', ...
    'lambdaFinal','status','termination','treat','-v7.3');
stateFile = fullfile(evDir, sprintf('%s_state.mat', tag));
state = struct('rho', res.rho, 'cfg', cfg, 'NE', NE, 'nOuter', n, 'tag', tag, ...
               'omega', res.omega, 'lambda', res.lambda); %#ok<NASGU>
save(stateFile, 'state', '-v7.3');
diagFile = fullfile(evDir, sprintf('%s_diag.mat', tag));
save(diagFile, 'dg', '-v7.3');
out.files = struct('trajectory', trajFile, 'state', stateFile, 'diag', diagFile);

% ---- SOCP per-iteration table --------------------------------------------
try
    T = cs_socp_table(res.socp);
    socpCsv = fullfile(runDir, sprintf('%s_socp_iterations.csv', tag));
    writetable(T, socpCsv);
    out.files.socpCsv = socpCsv;
catch ME
    out.socpTableError = ME.message;
end

% ---- frozen cv_export schema (column-for-column the control's) -----------
try
    per = cv_telemetry(res, RHO, NE, cfg.design.initial);
    csvFile = fullfile(runDir, sprintf('%s_iterations.csv', tag));
    cv_export(per, csvFile);
    out.files.csv = csvFile;
    out.Mnd_final = per.Mnd(end); out.gray_final = per.gray(end); out.mid_final = per.mid(end);
catch ME
    out.cvExportError = ME.message;
end

% ---- supplement (as cp_supplement) ---------------------------------------
try
    h = res.hist; om = h.omega; J = size(om,1);
    tOther = h.tOuter(1:n).' - h.tEig(:) - h.tGrad(:) - h.tInner(:);
    gap23 = nan(n,1);
    if J >= 3, gap23 = (om(3,:).' - om(2,:).')./om(2,:).'; end
    cols = {'outer'}; M = (1:n).';
    for j = 1:J, cols{end+1} = sprintf('omega%d', j); M = [M, om(j,:).']; end %#ok<AGROW>
    extra = {'gap23', gap23; 'tOuter', h.tOuter(1:n).'; 'tEig', h.tEig(:); 'tGrad', h.tGrad(:); ...
             'tInner', h.tInner(:); 'tOther', tOther; 'multN', h.N(:); 'multJ', double(h.multJ(:)); ...
             'degen', h.degen(:); 'nInner', h.nInner(:); 'dxOuter', h.dxOuter(:); ...
             'dxNorm2', h.dxNorm2(:); 'move', h.move(:); 'stage', h.stage(:); 'vol', h.vol(:)};
    for k = 1:size(extra,1)
        cols{end+1} = extra{k,1}; M = [M, double(extra{k,2})]; %#ok<AGROW>
    end
    supFile = fullfile(runDir, sprintf('%s_supplement.csv', tag));
    fid = fopen(supFile,'w'); fprintf(fid, '%s\n', strjoin(cols, ','));
    fprintf(fid, [repmat('%.17g,',1,numel(cols)-1) '%.17g\n'], M.'); fclose(fid);
    out.files.supplement = supFile;
catch ME
    out.supplementError = ME.message;
end

% ---- scalar record --------------------------------------------------------
out.omega = res.omega(:).'; out.omega1 = res.omega(1); out.omega2 = res.omega(2);
if numel(res.omega) >= 3, out.omega3 = res.omega(3); out.gap23 = (res.omega(3)-res.omega(2))/res.omega(2); end
out.gap12 = (res.omega(2)-res.omega(1))/res.omega(1);
out.volume_final = mean(res.rho);
out.rho_sha256 = cs_hash(res.rho);
if n > 0
    out.move_final = hist.move(end); out.stage_final = hist.stage(end);
    out.N_unique = unique(hist.N); out.multJ_count = sum(hist.multJ);
    out.terminal_l2 = hist.dxNorm2(end); out.terminal_max = hist.dxOuter(end);
    out.t = struct('eig', sum(hist.tEig), 'grad', sum(hist.tGrad), 'inner', sum(hist.tInner), ...
                   'outer', sum(hist.tOuter(1:n)));
end
if isfield(res,'exhaustion')
    out.stageStarts = res.exhaustion.stageStarts; out.descents = res.exhaustion.descents;
    out.terminalDeclared = res.exhaustion.terminalDeclared;
    out.terminalDeclIter = res.exhaustion.terminalDeclIter;
    out.terminalDeclBegin = res.exhaustion.terminalDeclBegin;
    out.terminalBranch = res.exhaustion.terminalBranch;
    out.eventBranch = res.exhaustion.eventBranch;
end
out.log = res.log;
out.meta = meta;
out.wallclock_solver = res.wallclock;
cs_json(fullfile(runDir, sprintf('%s_record.json', tag)), out);
end
