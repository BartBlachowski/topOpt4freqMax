function A = dr_analyze()
%DR_ANALYZE  Phases D-N: prefix checks, regime classification, event alignment.
%   Purely post hoc.  No solver is invoked.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
DIAG  = fileparts(study);
A = struct();

% ---------- the two new runs -------------------------------------------
RA = load(fullfile(study,'runs','runA_400x50.mat'));   % 400x50 production
RB = load(fullfile(study,'runs','runB_320x40.mat'));   % 320x40 fixed move

% ---------- retained references (read-only) ----------------------------
U160 = load(fullfile(DIAG,'move_transition','runs','armU_160x20.mat')); % fixed 0.04 throughout
P160 = load(fullfile(DIAG,'move_transition','runs','armP_160x20.mat')); % production
P320 = load(fullfile(DIAG,'move_transition','runs','armP_320x40.mat')); % production

rho0 = 0.5;
try, rho0 = olh.config.getPath(RA.cfg,'design.initial'); catch, end

% ---------- Phase D: RUN B common-prefix reproduction ------------------
A.prefix = struct();
A.prefix.vs_move_stop_fixedmove = dr_prefix(RB.RHO, RB.out.per, ...
    fullfile(DIAG,'move_stop','runs','fixedmove_320x40.mat'), 'RHO', 216, 'fixedmove216');
A.prefix.vs_move_transition_armU = dr_prefix(RB.RHO, RB.out.per, ...
    fullfile(DIAG,'move_transition','runs','armU_320x40.mat'), 'RHO', 213, 'armU213');

% ---------- dynamics for the retained trajectories ---------------------
D160u = dr_dyn(U160.RHO, U160.out.per.move, rho0);
D160p = dr_dyn(P160.RHO, P160.out.per.move, rho0);
D320p = dr_dyn(P320.RHO, P320.out.per.move, rho0);

% ---------- assemble the five trajectories we reason about -------------
T = struct();
T.f160 = pack('160x20 fixed move 0.04 (retained ARM U)', U160.out.per, D160u, U160.RHO);
T.p160 = pack('160x20 production (retained ARM P)',      P160.out.per, D160p, P160.RHO);
T.p320 = pack('320x40 production (retained ARM P)',      P320.out.per, D320p, P320.RHO);
T.f320 = pack('320x40 fixed move 0.04 EXTENDED (RUN B)', RB.out.per,   RB.out.per, RB.RHO);
T.p400 = pack('400x50 production (RUN A)',               RA.out.per,   RA.out.per, RA.RHO);

% ---------- Phase E-H: classify -----------------------------------------
fn = fieldnames(T);
for i = 1:numel(fn)
    T.(fn{i}).C = dr_classify(T.(fn{i}).dyn, 20);
end

% ---------- Phase J: events ---------------------------------------------
for i = 1:numel(fn)
    t = T.(fn{i}); p = t.per; n = numel(p.move);
    ev = struct();
    ev.nOuter        = n;
    ev.firstBetaStall = firstTrue(p.betaStallFires);
    d = find([false; p.move(2:end) < p.move(1:end-1)]);
    ev.moveDescents  = d(:).';
    ev.firstDescent  = pick(d,1);
    ev.period2Onset  = t.C.onset.PERIOD2;
    ev.coherentOnset = t.C.onset.COHERENT;
    ev.finalLabel    = t.C.labelFinal;
    T.(fn{i}).ev = ev;
end

A.T = T;
A.rho0 = rho0;
A.runA = struct('status',RA.out.status,'nOuter',RA.out.nOuter,'wall_s',RA.out.wall_s, ...
    'innerTotal',RA.out.innerTotal,'omega1',RA.out.omega(1),'Mnd_final',RA.out.Mnd_final, ...
    'cfgHash',RA.out.cfgHash,'volume_final',RA.out.volume_final);
A.runB = struct('status',RB.out.status,'nOuter',RB.out.nOuter,'wall_s',RB.out.wall_s, ...
    'innerTotal',RB.out.innerTotal,'omega1',RB.out.omega(1),'Mnd_final',RB.out.Mnd_final, ...
    'cfgHash',RB.out.cfgHash,'volume_final',RB.out.volume_final);

save(fullfile(study,'evidence','dr_analysis.mat'),'A','-v7.3');
fprintf('\n[dr_analyze] saved evidence/dr_analysis.mat\n');
end

function s = pack(label, per, dyn, RHO)
s = struct('label',label,'per',per,'dyn',dyn,'RHO',RHO);
end
function k = firstTrue(v), k = find(v,1); if isempty(k), k = NaN; end, end
function v = pick(a,i), if numel(a)>=i, v=a(i); else, v=NaN; end, end
