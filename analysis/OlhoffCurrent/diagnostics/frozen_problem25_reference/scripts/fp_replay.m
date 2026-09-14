function out = fp_replay(maxIter)
%FP_REPLAY  FROZEN-SUBPROBLEM REPLAY of the repeated-MMA inner iteration.
%
%   AUDIT-ONLY mirror of +impl/algo/innerLoop.m: every numeric expression is
%   copied character-for-character; the only differences are (1) the mmasub
%   outputs production discards are captured at checkpoints, (2) the stopping
%   tolerance is 1e-12 so the iteration runs to maxIter, (3) cheap scalars are
%   logged every iteration.  The returned drho is NEVER applied to a density.
if nargin < 1, maxIter = 5000; end
S = fp_setup();                                   %#ok<NASGU> keeps the guard alive
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'), 'ctx','xP19','xM500');
ctx = L.ctx;
tolInner = 1e-12;

checkpoints = unique([1:50, 60:10:100, 150:50:1000, 1100:100:maxIter, 19, 500, maxIter]);
checkpoints = checkpoints(checkpoints <= maxIter);
asyKeep = [19 500 1000 maxIter];

NE = numel(ctx.rho);
N  = numel(ctx.lam);
lamref = ctx.lam(1);
Vtot   = ctx.volfrac*NE;

nvar = NE + 1;
lo = max(ctx.rhomin - ctx.rho, -ctx.move);
hi = min(1          - ctx.rho,  ctx.move);
xmin = [lo; 0];
xmax = [hi; 5];

x = [zeros(NE,1); 1];
xold1 = x; xold2 = x;
low = xmin; upp = xmax;

if ctx.offDiag, m = N + 2;
else,           m = N + 2 + N*(N-1); end
a0 = 1; aMMA = zeros(m,1); cMMA = 1000*ones(m,1); dMMA = zeros(m,1);
if isfield(ctx,'dOff'), dOff = ctx.dOff; else, dOff = []; end

nCk = numel(checkpoints);
CK = struct('iter',num2cell(checkpoints), 'x',cell(1,nCk), 'lam',cell(1,nCk), ...
            'xsi',cell(1,nCk), 'eta',cell(1,nCk), 'ymma',cell(1,nCk), 'zmma',cell(1,nCk), ...
            'fval',cell(1,nCk), 'low',cell(1,nCk), 'upp',cell(1,nCk));
H = struct('beta',zeros(maxIter,1),'maxAbsDrho',zeros(maxIter,1),'relStep',zeros(maxIter,1), ...
           'dx',zeros(maxIter,1),'fval',zeros(maxIter,m),'lam',zeros(maxIter,m), ...
           'ymax',zeros(maxIter,1),'zmma',zeros(maxIter,1),'e1',zeros(maxIter,1),'e2',zeros(maxIter,1));
bit19 = NaN; bit500 = NaN;
t0 = tic;
for it = 1:maxIter
    drho = x(1:NE);
    bs   = x(end);
    if ctx.offDiag
        [dlam, ddlam, ~, ~] = deltaLambda(ctx.F, drho, dOff);
    else
        ddlam = zeros(NE,N); dlam = zeros(N,1);
        for j = 1:N
            ddlam(:,j) = ctx.F(:,j,j);
            dlam(j)    = ctx.F(:,j,j).'*drho;
        end
    end
    fval = zeros(m,1);
    dfdx = zeros(m,nvar);
    for j = 1:N
        fval(j)          = bs - (ctx.lam(j) + dlam(j))/lamref;
        dfdx(j,1:NE)     = -ddlam(:,j).'/lamref;
        dfdx(j,nvar)     = 1;
    end
    fval(N+1)        = bs - (ctx.lamJ + ctx.fJJ.'*drho)/lamref;
    dfdx(N+1,1:NE)   = -ctx.fJJ.'/lamref;
    dfdx(N+1,nvar)   = 1;
    fval(N+2)        = (sum(ctx.rho + drho) - Vtot)/Vtot;
    dfdx(N+2,1:NE)   = 1/Vtot;
    f0val  = -bs;
    df0dx  = zeros(nvar,1); df0dx(nvar) = -1;

    [xmma,ymma,zmma,lamD,xsi,eta,~,~,~,low,upp] = mmasub(m,nvar,it,x,xmin,xmax, ...
        xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aMMA,cMMA,dMMA);

    dx = max(abs(xmma(1:NE)-x(1:NE)));
    relStep = dx / max(max(abs(xmma(1:NE))), 1e-12);
    xold2 = xold1; xold1 = x; x = xmma;

    % logs (the iterate AFTER this MMA step, i.e. what innerLoop would return
    % if it stopped here)
    H.beta(it) = x(end)*lamref; H.maxAbsDrho(it) = max(abs(x(1:NE)));
    H.relStep(it) = relStep; H.dx(it) = dx;
    H.fval(it,:) = fval.'; H.lam(it,:) = lamD(:).';   % fval at the PRE-step point
    H.ymax(it) = max(ymma); H.zmma(it) = zmma;
    H.e1(it) = dlam(1)+dOff(1); H.e2(it) = dlam(2)+dOff(2);
    k = find(checkpoints == it, 1);
    if ~isempty(k)
        CK(k).x = x; CK(k).lam = lamD(:); CK(k).xsi = xsi; CK(k).eta = eta;
        CK(k).ymma = ymma; CK(k).zmma = zmma; CK(k).fval = fval;
        if any(asyKeep == it), CK(k).low = low; CK(k).upp = upp; end
    end
    if it == 19,  bit19  = isequal(x, L.xP19);  fprintf('[fp_replay] it 19  bitwise==P19 : %d\n', bit19); end
    if it == 500 && ~isempty(L.xM500), bit500 = isequal(x, L.xM500); fprintf('[fp_replay] it 500 bitwise==M500: %d\n', bit500); end
    if mod(it,100) == 0
        fprintf('[fp_replay] it %5d  beta=%.6f  max|drho|/move=%.4f  relStep=%.3e  t=%.0fs\n', ...
            it, H.beta(it), H.maxAbsDrho(it)/ctx.move, relStep, toc(t0));
    end
    if it >= ctx.minInner && relStep < tolInner, break, end
end
wall = toc(t0);
xM5000 = x;
out = struct('label','FROZEN-SUBPROBLEM REPLAY -- drho discarded, never applied', ...
    'maxIter',maxIter,'nIter',it,'wall_s',wall,'bitwise19',bit19,'bitwise500',bit500, ...
    'checkpoints',checkpoints,'tolInner',tolInner);
save(fullfile(ev,'mma_replay.mat'), 'CK','H','out','xM5000','checkpoints','-v7.3');
fid = fopen(fullfile(ev,'mma_replay_summary.json'),'w');
fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);
fprintf('[fp_replay] done: %d iterations in %.0fs; bitwise19=%d bitwise500=%d\n', it, wall, bit19, bit500);
clear x xM5000 drho
end
