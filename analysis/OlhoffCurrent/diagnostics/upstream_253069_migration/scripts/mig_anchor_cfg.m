function [cfg, meta] = mig_anchor_cfg(label, root)
% mig_anchor_cfg: byte copy of the 253069 snapshot's architecture/anchors/code/anchorCfg.m with only
% the function name changed and the hard-coded upstream root replaced by the argument ROOT.
%ANCHORCFG  The behavioural-anchor configurations, in LEGACY flat form.
%
%   These are the pre-refactor reference configurations.  Every one of them is
%   derived from the SAME frozen source of truth used by every historical
%   audit -- CFG(k).cfg of
%       audit_m4_topology_restoration/baseline/tree/
%           audit_termination_mesh_admission/runs/TMA_FROZEN_CFGS.mat
%   -- read from the preserved snapshot, never rebuilt from defaultCfg.
%
%   CFG(1) is 160x20, CFG(2) 240x30, CFG(3) 320x40.  They differ only in
%   nelx, nely, tolOuter and name; every scientific field is identical.
%
%   meta.exercises records WHICH code path the anchor is there to pin, so a
%   later reader can tell why the anchor exists.

L = load(fullfile(root,'audit_m4_topology_restoration','baseline','tree', ...
        'audit_termination_mesh_admission','runs','TMA_FROZEN_CFGS.mat'));
cfg = L.CFG(1).cfg;          % 160x20 frozen realization
cfg.verbose = false;
cfg.diag    = true;          % needed for drho-level bitwise comparison
meta = struct('label',label,'mesh',[160 20],'capped',false);

switch label
    case 'A1_frozen160'
        % The frozen conference realization verbatim: S2 ladder on the beta
        % signal, settledmove guard, no restoration guard, no continuation.
        meta.exercises = {'mass eq4b','filterMode all','multRule subspace', ...
                          'S2 ladder / beta signal','outerGuard settledmove','outerNorm l2'};

    case 'A2_mature160'
        cfg.restorationGuard = 'R2';  cfg.maxOuter = 400;
        meta.exercises = {'restorationGuard R2 (max|drho| vs epsRMS)'};

    case 'A3_r1ladder160'
        cfg.restorationGuard = 'R1';  cfg.maxOuter = 400;
        meta.exercises = {'restorationGuard R1 (remaining ladder levels vs epsRMS)'};

    case 'A4_nodescent160'
        % audit_s2_final_conference_restoration 'nodescent': the S0 family with
        % the move pinned at the ladder's own first level.
        cfg.moveFamily = 'S0';  cfg.move = 0.04;  cfg.maxOuter = 400;
        meta.exercises = {'moveFamily S0 (no move descent)'};

    case 'A5_pcont160'
        cfg.restorationGuard = 'R2';  cfg.pSchedule = [1 2 3];  cfg.maxOuter = 250;
        meta.capped = true;
        meta.exercises = {'p continuation COUPLED to the S2 ladder stage','stop blocked while p<pEnd'};

    case 'A6_pdecoupled160'
        cfg.restorationGuard = 'R2';  cfg.pSchedule = [1 2 3];
        cfg.pDecouple = true;  cfg.maxOuter = 250;
        meta.capped = true;
        meta.exercises = {'p continuation DECOUPLED','pEvent stall interception','ladder reset'};

    case 'A7_massp160'
        cfg.restorationGuard = 'R2';  cfg.pSchedule = [1 2 3];
        cfg.pDecouple = true;  cfg.massLowP = 'lin';  cfg.maxOuter = 250;
        meta.capped = true;
        meta.exercises = {'printed-mass continuation (massLowP) tied to the p schedule'};

    case 'A8_projidentity160'
        cfg.restorationGuard = 'R2';  cfg.maxOuter = 400;
        cfg.projection = struct('on',true,'betaSchedule',0,'eta',0.5);
        meta.exercises = {'projection ON at betaProj=0','density filter + exact identity projection','projChain sensitivities','volFun'};

    case 'A9_projection160'
        cfg.restorationGuard = 'R2';  cfg.maxOuter = 600;
        cfg.projection = struct('on',true,'betaSchedule',[1 2 4 8],'eta',0.5);
        meta.capped = true;
        meta.exercises = {'full tanh projection','projection continuation (projEvent)','beta schedule'};

    case 'A10_binarydiag160'
        % The OTHER end of the option space: the pre-M4 reconstruction style.
        cfg.multRule = 'binary';  cfg.tolMult = 0.02;
        cfg.filterMode = 'diag';  cfg.moveFamily = 'S0';
        cfg.outerGuard = 'none';  cfg.move = 0.05;  cfg.maxOuter = 400;
        meta.exercises = {'multRule binary (greedy detector)','filterMode diag','outerGuard none'};

    case 'A11_maxnorm160'
        % The max-norm branch at the frozen tolerance is degenerate: eps = 0.05
        % exceeds the ladder's own first move level 0.04, so max|drho| <= 0.04
        % is satisfied on the second iteration and the anchor would pin almost
        % nothing.  Tighten eps to the smallest ladder level so that the test
        % can only be met once the ladder has fully descended.
        cfg.outerNorm = 'max';  cfg.tolOuter = 0.005;  cfg.maxOuter = 400;
        meta.exercises = {'outerNorm max (tightened so the branch is exercised)'};

    case 'A12_rhovar160'
        cfg.innerVar = 'rho';  cfg.maxOuter = 120;
        meta.capped = true;
        meta.exercises = {'innerVar rho (innerLoopRho, persistent MMA state)'};

    otherwise
        error('anchorCfg:label','Unknown anchor label %s',label);
end
cfg.name = label;
end
