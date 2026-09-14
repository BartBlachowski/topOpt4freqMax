function sd_same_state_run(side, stateSet)
%SD_SAME_STATE_RUN  Part 15 driver.  side = 'T' (target +impl) or 'S' (source
%   snapshot, evaluated under S1 = SIMP/eq4b and S0 = Pedersen/eq2).
%   stateSet = 'retained' (rho0, C480 k=10/20/100/386, S480 final) or 'm1'
%   (preregistered M1 states, available after the M1 run).
%   Zero rho updates: every state is frozen; outputs go to evaluations/same_state.
if nargin < 2, stateSet = 'retained'; end
switch side
    case 'T', [P, guard] = sd_use_target(); %#ok<ASGLU> keep the gate's path alive
    case 'S', P = sd_use_source();
    otherwise, error('side must be T or S');
end
maxNumCompThreads(1);
outDir = fullfile(P.eval, 'same_state'); if ~isfolder(outDir), mkdir(outDir); end
states = sd_states(P, stateSet);

% ---- evaluator configurations -------------------------------------------
cfgs = struct('name',{},'cfg',{});
if side == 'T'
    L = load(P.c480traj, 'cfg');                      % the C480 canary's own cfg
    cfgs(1) = struct('name','T','cfg',L.cfg);
else
    L = load(P.s480, 'cfg');                          % S480's own cfg (Pedersen/eq2)
    cM = sd_m1_config();                              % M1 cfg (SIMP/eq4b)
    cfgs(1) = struct('name','S1','cfg',cM);
    cfgs(2) = struct('name','S0','cfg',L.cfg);
end

summary = {};
for s = 1:numel(states)
    st = states(s);
    for c = 1:numel(cfgs)
        boxes = {0.04};
        if strcmp(cfgs(c).name,'T') && ~isempty(st.boxT) && ~isequal(st.boxT, 0.04), boxes{end+1} = st.boxT; end %#ok<AGROW>
        if cfgs(c).name(1) == 'S' && ~isempty(st.boxS), boxes{end+1} = st.boxS; end %#ok<AGROW>
        t0 = tic;
        R = sd_eval_state(cfgs(c).cfg, st.rho, boxes); %#ok<NASGU>
        f = fullfile(outDir, sprintf('%s__%s.mat', st.name, cfgs(c).name));
        save(f, 'R', '-v7');
        summary{end+1} = sprintf('%s %s omega1=%.15g lam1=%.15g N=%d nInner=%s t=%.1fs', ...
            st.name, cfgs(c).name, R.omega(1), R.lam(1), R.N, mat2str([R.inner.nInner]), toc(t0)); %#ok<AGROW>
        fprintf('%s\n', summary{end});
    end
end
fid = fopen(fullfile(outDir, sprintf('summary_%s_%s.txt', side, stateSet)), 'w');
fprintf(fid, '%s\n', summary{:}); fclose(fid);
end

function S = sd_states(P, set)
S = struct('name',{},'rho',{},'boxT',{},'boxS',{});
NE = 480*60;
switch set
    case 'retained'
        S(end+1) = struct('name','rho0','rho',0.5*ones(NE,1),'boxT',0.04,'boxS',0.10);
        mf = matfile(P.c480traj);
        H = load(P.c480traj, 'hist'); h = H.hist;
        for k = [10 20 100 386]
            r = mf.RHO(:, k);
            if k < numel(h.move), bT = h.move(k+1); else, bT = h.move(k); end
            S(end+1) = struct('name', sprintf('C480_k%03d', k), 'rho', r, 'boxT', bT, 'boxS', []); %#ok<AGROW>
        end
        L = load(P.s480, 'res');
        S(end+1) = struct('name','S480_final','rho',L.res.rho,'boxT',[],'boxS',[]);
    case 'm1'
        L = load(fullfile(P.eval, 'm1_states.mat'));  % written by the M1 post-processor
        for i = 1:numel(L.names)
            S(end+1) = struct('name', L.names{i}, 'rho', L.rhos(:,i), 'boxT', [], 'boxS', L.boxes(:,i)); %#ok<AGROW>
        end
end
end
