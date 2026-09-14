function Tsummary = run_terminal_direction_audit()
% RUN_TERMINAL_DIRECTION_AUDIT  Fixed-direction audit at saved terminal states.
%
% This is a diagnostic experiment, not an optimization campaign.  For each
% saved paper-example result it:
%   1. reconstructs the next filtered LP subproblem at the exhausted radius;
%   2. solves it once for a direction d;
%   3. holds d fixed and evaluates the physical eigenproblem at rho + t*d;
%   4. compares actual lambda-objective changes with filtered and raw linear
%      predictions; and
%   5. plots lambda_j(t), omega_j(t), and model agreement.
%
% The test cannot certify full stationarity.  It answers the narrower,
% conclusion-critical question: is the filtered direction rejected because it
% is not an ascent direction for the true physical objective, or does a true
% improving small step exist below the imposed trust-region floor?

here = fileparts(mfilename('fullpath'));
matlab_root = fullfile(here, '..', '..', 'Matlab');
paper_root  = fullfile(here, '..', 'paper_examples');
out_root    = fullfile(here, 'results');
addpath(matlab_root);
addpath(fullfile(matlab_root, '..', '..', '..', 'tools', 'Matlab'));
if ~exist(out_root, 'dir'), mkdir(out_root); end

cases = {'ss_n1','cs_n1','cc_n1','ss_n2','cs_n2','cc_n2','cc_gap23'};
% Powers of two make the approach to t=0 explicit without tuning a step.
tpos = 2.^(-(0:14));
tgrid = sort([0, tpos]);

rows = cell(numel(cases),1);
for ic = 1:numel(cases)
    name = cases{ic};
    src = fullfile(paper_root, name, 'result.mat');
    if ~exist(src, 'file')
        error('terminal_direction_audit:MissingResult', ...
            'Missing saved result for %s: %s', name, src);
    end
    S = load(src, 'res');
    r = S.res;
    cfg = audit_defaults(r.cfg);
    rho0 = r.rho(:);

    model = make_model(cfg);
    [lam0, omega0, Phi0, M0] = eig_state(rho0, model, cfg); %#ok<ASGLU>
    sp = make_subproblem(rho0, lam0, Phi0, r.hist, model, cfg);
    sp.move = cfg.move_min;
    sol = subproblem_lp(sp, cfg.inner);
    d = sol.drho(:);

    [filtered_dlambda, raw_dlambda] = predict_lambda_objective(sp, d, cfg);

    ns = numel(tgrid);
    nm = cfg.n_modes;
    lam_t = nan(ns,nm);
    omega_t = nan(ns,nm);
    obj_lam = nan(ns,1);
    obj_omega = nan(ns,1);
    vol = nan(ns,1);
    for it = 1:ns
        rho_t = rho0 + tgrid(it)*d;
        [lt, ot] = eig_state(rho_t, model, cfg);
        lam_t(it,:) = lt(:)';
        omega_t(it,:) = ot(:)';
        [obj_lam(it), obj_omega(it)] = physical_objectives( ...
            lam_t(it,:)', omega_t(it,:)', cfg);
        vol(it) = mean(rho_t);
    end

    obj_lam0 = obj_lam(tgrid == 0);
    obj_omega0 = obj_omega(tgrid == 0);
    actual_dlambda = obj_lam - obj_lam0;
    actual_domega = obj_omega - obj_omega0;
    pred_filtered = tgrid(:) * filtered_dlambda;
    pred_raw = tgrid(:) * raw_dlambda;

    % Estimate the t->0 directional slope over a fixed, predeclared window.
    fit_mask = tgrid >= 2^-10 & tgrid <= 2^-5;
    tf = tgrid(fit_mask)';
    yf = actual_dlambda(fit_mask);
    fd_slope = (tf' * yf) / (tf' * tf);

    scale = max(abs(obj_lam0), 1);
    sign_tol = 1e-10 * scale;
    if filtered_dlambda > sign_tol && fd_slope > sign_tol && raw_dlambda <= sign_tol
        classification = 'TRUE_ASCENT_BELOW_FLOOR_RAW_MODEL_MISMATCH';
    elseif filtered_dlambda > sign_tol && fd_slope > sign_tol
        classification = 'TRUE_ASCENT_EXISTS_BELOW_FLOOR';
    elseif filtered_dlambda > sign_tol && raw_dlambda < -sign_tol && fd_slope < -sign_tol
        classification = 'FILTERED_DIRECTION_IS_TRUE_DESCENT';
    elseif filtered_dlambda > sign_tol && fd_slope <= sign_tol
        classification = 'NO_TRUE_ASCENT_ALONG_FILTERED_DIRECTION';
    else
        classification = 'INDETERMINATE_DIRECTION';
    end

    case_dir = fullfile(out_root, name);
    if ~exist(case_dir, 'dir'), mkdir(case_dir); end
    writematrix(d, fullfile(case_dir, 'direction_drho.csv'));

    Tout = table(tgrid(:), obj_lam, obj_omega, actual_dlambda, actual_domega, ...
        pred_filtered, pred_raw, vol, ...
        'VariableNames', {'t','lambda_objective','omega_objective', ...
        'actual_delta_lambda_objective','actual_delta_omega_objective', ...
        'filtered_predicted_delta_lambda','raw_predicted_delta_lambda','volume'});
    for j = 1:nm
        Tout.(sprintf('lambda%d',j)) = lam_t(:,j);
        Tout.(sprintf('omega%d',j)) = omega_t(:,j);
    end
    writetable(Tout, fullfile(case_dir, 'line_samples.csv'));
    make_plot(fullfile(case_dir, 'lambda_omega_line_audit.png'), name, ...
        tgrid, lam_t, omega_t, actual_dlambda, pred_filtered, pred_raw, cfg);

    row = struct();
    row.case = string(name);
    row.n_target = cfg.n_target;
    row.cluster_N = numel(sp.up.L);
    if strcmpi(cfg.objective,'gap'), row.cluster_R = numel(sp.lo.L); else, row.cluster_R = 0; end
    row.move = sp.move;
    row.drho_inf = max(abs(d));
    row.frac_at_bound = sol.frac_at_bound;
    row.inner_stop = string(sol.stop_reason);
    row.lmi_violation = sol.lmi_violation;
    row.filtered_dlambda = filtered_dlambda;
    row.raw_dlambda = raw_dlambda;
    row.fd_slope_dlambda = fd_slope;
    row.actual_dlambda_t1 = actual_dlambda(tgrid == 1);
    row.actual_domega_t1 = actual_domega(tgrid == 1);
    row.raw_slope_rel_error = abs(fd_slope-raw_dlambda) / ...
        max([abs(fd_slope),abs(raw_dlambda),sign_tol]);
    row.classification = string(classification);
    rows{ic} = row;

    fprintf('%-9s filtered %+12.5e  raw %+12.5e  FD %+12.5e  %s\n', ...
        name, filtered_dlambda, raw_dlambda, fd_slope, classification);
end

Tsummary = struct2table(cell2mat(rows));
writetable(Tsummary, fullfile(out_root, 'summary.csv'));
write_report(fullfile(out_root, 'REPORT.md'), Tsummary, tgrid);
fprintf('\nTerminal-direction audit written to %s\n', out_root);
end

% Saved runner contracts predate expansion of defaults inside the solver and
% therefore omit fields whose values were supplied by TOPOPT_FREQ_EXACT.
function cfg = audit_defaults(cfg)
if ~isfield(cfg,'move_min') || isempty(cfg.move_min), cfg.move_min=1e-4; end
if ~isfield(cfg,'inner') || isempty(cfg.inner), cfg.inner=struct(); end
if ~isfield(cfg,'lumped_mass_frac'), cfg.lumped_mass_frac=[]; end
if ~isfield(cfg,'lumped_mass'), cfg.lumped_mass=[]; end
end

% -------------------------------------------------------------------------
function model = make_model(cfg)
nelx = cfg.nelx; nely = cfg.nely;
dx = cfg.L/nelx; dy = cfg.H/nely;
nEl = nelx*nely;
nDof = 2*(nelx+1)*(nely+1);
[Ke_star, Me_star] = fe_q4_exact(cfg.nu, cfg.t, dx, dy);
model.Ke = cfg.E0*Ke_star;
model.Me = cfg.rho0*Me_star;
nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
model.cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
              cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
[Il,Jl] = find(tril(ones(8)));
model.iK = reshape(model.cMat(:,Il)',[],1);
model.jK = reshape(model.cMat(:,Jl)',[],1);
model.Kl = model.Ke(sub2ind([8,8],Il,Jl));
model.Ml = model.Me(sub2ind([8,8],Il,Jl));
model.nDof = nDof;
model.free = setdiff(1:nDof, build_supports_exact(cfg.support_type,nodeNrs));
if cfg.sensitivity_filter
    [model.h, model.Hs] = build_filter(nelx,nely,cfg.rmin_elem);
else
    model.h = []; model.Hs = [];
end

model.Mc = sparse(nDof,nDof);
has_frac = isfield(cfg,'lumped_mass_frac') && ~isempty(cfg.lumped_mass_frac);
has_mass = isfield(cfg,'lumped_mass') && ~isempty(cfg.lumped_mass);
if has_frac
    rho_initial = cfg.volfrac*ones(nEl,1);
    mb = sum(mass_interp(rho_initial,cfg.mass_mode,cfg.mass_q))* ...
         cfg.rho0*dx*dy*cfg.t;
    mval = cfg.lumped_mass_frac*mb;
    [model.Mc,~] = lumped_mass(nodeNrs,nDof,'bottom_mid',mval);
elseif has_mass
    [model.Mc,~] = lumped_mass(nodeNrs,nDof, ...
        cfg.lumped_mass.where,cfg.lumped_mass.mass);
end
end

% -------------------------------------------------------------------------
function [lam,omega,Phi,M] = eig_state(rho,model,cfg)
[K,M] = assemble_KM_exact(rho,model.Kl,model.Ml,model.iK,model.jK, ...
    model.nDof,cfg.penal,cfg.mass_mode,cfg.mass_q);
M = M + model.Mc;
Kf = K(model.free,model.free); Mf = M(model.free,model.free);
opts = struct('tol',1e-12,'maxit',800);
opts.v0 = ones(numel(model.free),1);
opts.v0(2:2:end) = -1; opts.v0 = opts.v0/norm(opts.v0);
[V,D,flag] = eigs(Kf,Mf,cfg.n_modes,'SM',opts);
if flag ~= 0, error('terminal_direction_audit:EigsFailure','eigs failed'); end
[lam,ix] = sort(real(diag(D))); V = real(V(:,ix));
for j=1:cfg.n_modes
    s = sqrt(abs(V(:,j)'*(Mf*V(:,j))));
    V(:,j) = V(:,j)/s;
end
omega = sqrt(max(lam,0));
Phi = zeros(model.nDof,cfg.n_modes); Phi(model.free,:) = V;
lam = lam(:); omega = omega(:);
end

% -------------------------------------------------------------------------
function sp = make_subproblem(rho,lam,Phi,hist,model,cfg)
n = cfg.n_target;
Nprev = hist.N(end);
[N,~,cl] = detect_multiplicity(lam,n,cfg.mult_tol_join,cfg.mult_tol_leave,Nprev);
N = min(N,cfg.N_max); cl = n:n+N-1;
J = n+N; if J>cfg.n_modes, J=0; end
sp = struct('mode',cfg.objective,'rho',rho,'volfrac',cfg.volfrac, ...
    'rho_min',cfg.rho_min,'move',cfg.move_min);
sp.up = make_block(cl,J,rho,lam,Phi,model,cfg);
if strcmpi(cfg.objective,'gap')
    Rprev = hist.R(end);
    [R,~,cllo] = detect_multiplicity_below(lam,n,cfg.mult_tol_join,cfg.mult_tol_leave,Rprev);
    R = min(R,cfg.N_max); cllo=(n-R):(n-1);
    Jm=n-R-1; if Jm<1, Jm=0; end
    sp.lo = make_block(cllo,Jm,rho,lam,Phi,model,cfg);
end
end

% -------------------------------------------------------------------------
function blk = make_block(cl,guard,rho,lam,Phi,model,cfg)
switch lower(cfg.lam_ref_rule)
    case 'lowest', lam_ref = lam(cl(1));
    case 'mean',   lam_ref = mean(lam(cl));
    otherwise, error('terminal_direction_audit:BadReference','bad lam_ref_rule');
end
Fe_raw = generalized_gradients(rho,lam_ref,Phi(:,cl),model.cMat, ...
    model.Ke,model.Me,cfg.penal,cfg.mass_mode,cfg.mass_q);
Fe = Fe_raw;
if cfg.sensitivity_filter
    Nb = numel(cl);
    for s=1:Nb
        for k=s:Nb
            f = apply_sensitivity_filter(Fe_raw(:,s,k),rho,model.h,model.Hs, ...
                cfg.nely,cfg.nelx);
            Fe(:,s,k)=f; Fe(:,k,s)=f;
        end
    end
end
switch upper(cfg.cluster_model)
    case 'CA', L=lam_ref*ones(numel(cl),1);
    case 'CC', L=lam(cl);
    otherwise, error('terminal_direction_audit:BadCluster','bad cluster_model');
end
blk = struct('L',L(:),'Fe',Fe,'Fe_raw',Fe_raw,'guard',[]);
if guard>0
    g = compute_elem_sensitivity(rho,lam(guard),Phi(:,guard),model.cMat, ...
        model.Ke,model.Me,model.free,model.nDof,cfg.penal,cfg.mass_mode,cfg.mass_q);
    if cfg.sensitivity_filter
        g=apply_sensitivity_filter(g,rho,model.h,model.Hs,cfg.nely,cfg.nelx);
    end
    blk.guard=struct('lam',lam(guard),'grad',g);
end
end

% -------------------------------------------------------------------------
function [dfilt,draw] = predict_lambda_objective(sp,d,cfg)
uf = predict_block(sp.up,d,'Fe'); ur = predict_block(sp.up,d,'Fe_raw');
if strcmpi(cfg.objective,'gap')
    lf = predict_block(sp.lo,d,'Fe'); lr = predict_block(sp.lo,d,'Fe_raw');
    base = min(sp.up.L)-max(sp.lo.L);
    dfilt = (min(uf)-max(lf))-base;
    draw  = (min(ur)-max(lr))-base;
else
    base = min(sp.up.L);
    dfilt = min(uf)-base;
    draw  = min(ur)-base;
end
end

function p = predict_block(blk,d,field)
N=numel(blk.L); F=reshape(blk.(field),numel(d),N*N);
G=diag(blk.L)+reshape(F'*d,N,N);
p=sort(real(eig((G+G')/2)));
end

function [ol,oo] = physical_objectives(lam,omega,cfg)
n=cfg.n_target;
if strcmpi(cfg.objective,'gap')
    ol=lam(n)-lam(n-1); oo=omega(n)-omega(n-1);
else
    ol=lam(n); oo=omega(n);
end
end

% -------------------------------------------------------------------------
function make_plot(fn,name,t,lam,omega,dactual,pred_filt,pred_raw,cfg)
nm=min(5,size(lam,2));
fh=figure('Visible','off','Color','w','Position',[100 100 1450 430]);
subplot(1,3,1); hold on
for j=1:nm, plot(t,lam(:,j),'-o','MarkerSize',3,'DisplayName',sprintf('lambda_%d',j)); end
xlabel('t'); ylabel('lambda_j(t)'); title(sprintf('%s: eigenvalues',upper(name)),'Interpreter','none');
grid on; legend('Location','best');
subplot(1,3,2); hold on
for j=1:nm, plot(t,omega(:,j),'-o','MarkerSize',3,'DisplayName',sprintf('omega_%d',j)); end
xlabel('t'); ylabel('omega_j(t)'); title('eigenfrequencies'); grid on; legend('Location','best');
subplot(1,3,3); hold on
tp=t(t>0); ya=dactual(t>0);
semilogx(tp,ya,'k-o','MarkerSize',3,'DisplayName','actual');
semilogx(tp,pred_filt(t>0),'b--','DisplayName','filtered linear');
semilogx(tp,pred_raw(t>0),'r-.','DisplayName','raw linear');
yline(0,'Color',[0.4 0.4 0.4],'HandleVisibility','off'); set(gca,'XDir','reverse');
xlabel('t (toward 0 to the right)'); ylabel('Delta lambda objective');
title(sprintf('fixed d, move=%g',cfg.move_min)); grid on; legend('Location','best');
sgtitle(sprintf('%s terminal-direction audit',upper(name)),'Interpreter','none');
exportgraphics(fh,fn,'Resolution',170); close(fh);

% A change-scale companion makes the small but sign-critical variations
% visible without replacing the requested absolute lambda(t), omega(t) plot.
[dpath,base,~]=fileparts(fn);
fh=figure('Visible','off','Color','w','Position',[100 100 1100 430]);
subplot(1,2,1); hold on
for j=1:nm
    plot(t,lam(:,j)-lam(1,j),'-o','MarkerSize',3, ...
        'DisplayName',sprintf('Delta lambda_%d',j));
end
yline(0,'Color',[0.4 0.4 0.4],'HandleVisibility','off');
xlabel('t'); ylabel('lambda_j(t)-lambda_j(0)'); title('eigenvalue changes');
grid on; legend('Location','best');
subplot(1,2,2); hold on
for j=1:nm
    plot(t,omega(:,j)-omega(1,j),'-o','MarkerSize',3, ...
        'DisplayName',sprintf('Delta omega_%d',j));
end
yline(0,'Color',[0.4 0.4 0.4],'HandleVisibility','off');
xlabel('t'); ylabel('omega_j(t)-omega_j(0)'); title('eigenfrequency changes');
grid on; legend('Location','best');
sgtitle(sprintf('%s fixed-direction changes',upper(name)),'Interpreter','none');
exportgraphics(fh,fullfile(dpath,[base '_changes.png']),'Resolution',170); close(fh);
end

% -------------------------------------------------------------------------
function write_report(fn,T,tgrid)
fid=fopen(fn,'w');
fprintf(fid,'# Terminal-direction audit\n\n');
fprintf(fid,'Generated %s with MATLAB. The filtered LP direction is solved once at ',datestr(now,31));
fprintf(fid,'the saved terminal density and move floor, then held fixed over ');
fprintf(fid,'`t = %s`. This is a directional diagnostic, not an optimization sweep.\n\n',mat2str(tgrid,5));
fprintf(fid,'| case | filtered dLambda | raw dLambda | FD slope | rel. raw error | classification |\n');
fprintf(fid,'|---|---:|---:|---:|---:|---|\n');
for i=1:height(T)
    fprintf(fid,'| %s | %+.6e | %+.6e | %+.6e | %.3e | %s |\n', ...
        T.case(i),T.filtered_dlambda(i),T.raw_dlambda(i),T.fd_slope_dlambda(i), ...
        T.raw_slope_rel_error(i),T.classification(i));
end
nascent=sum(contains(T.classification,'TRUE_ASCENT'));
ndescent=sum(T.classification=='FILTERED_DIRECTION_IS_TRUE_DESCENT');
fprintf(fid,'\n## Result\n\n');
fprintf(fid,'%d of %d terminal states have a true improving physical step at the nominal ',nascent,height(T));
fprintf(fid,'move floor. The solver therefore stopped before testing an available ascent step. ');
fprintf(fid,'%d of %d filtered LP directions are physical descent directions as `t -> 0`, ',ndescent,height(T));
fprintf(fid,'confirming that the filtered subproblem is not a consistent local model of the physical objective.\n\n');
fprintf(fid,'Interpretation is limited to the tested filtered direction. A non-ascent ');
fprintf(fid,'result does not certify KKT stationarity or exclude other feasible ascent directions.\n');
fclose(fid);
end
