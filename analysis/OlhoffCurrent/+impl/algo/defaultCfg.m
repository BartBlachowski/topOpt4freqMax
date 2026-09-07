function cfg = defaultCfg()
%DEFAULTCFG  Baseline configuration.  Every entry marked UNSTATED is a swept
%   control parameter (CLAUDE.md sec.4) -- it is recorded per run, not buried.
cfg = struct( ...
  ... % ---- domain and material (settled, sec.2) ----
  'a',8,'b',1,'t',1,'E',1e7,'nu',0.3,'rhom',1, ...
  'nelx',160,'nely',20, ...
  'bc','a', ...
  ... % ---- idealizations resolved by the Fig. 4a read (see NOTES) ----
  'support','mid','axial','both','elemType','Q4','massType','consistent', ...
  ... % ---- SIMP (settled) ----
  'p',3,'massInterp','4','rhomin',1e-3,'rho0',0.5,'volfrac',0.5, ...
  ... % ---- problem ----
  'n',1,'Nmax',4, ...
  ... % ---- UNSTATED, swept ----
  'rminEl',3.0, ...          % filter radius in ELEMENT units
  'rminPhys',[], ...         % filter radius in PHYSICAL units; overrides rminEl if set
  'tolMult',0.02, ...        % multiplicity tolerance (relative)
  ...  % ---- multiplicity treatment (algo/multRule.m).  'binary' is the frozen
  ...  % memoryless test of Du & Olhoff p.98.  Everything else is an explicitly
  ...  % labelled reconstruction; see audit_multiplicity_reconstruction/.
  'multRule','binary', ... % 'binary' | 'latch' | 'hyst' | 'subspace'
  'tolEnter',0.01,'tolExit',0.05, ...   % 'hyst' only
  'subN',2, ...              % 'subspace' only: fixed cluster window size
  'move',0.05, ...           % move limit on drho (initial value for S1/S3)
  ...  % ---- step/move control (algo/moveControl.m).  NOT specified by the
  ...  % paper: the only printed bound on drho is the box (25f).  See that
  ...  % file's header for the A/B/C evidence classification.
  'moveFamily','S0', ...     % 'S0' fixed | 'S1' contracting | 'S2' staged | 'S3' lineage-derived
  'moveMin',0.002, ...       % floor for S1/S3
  's1Gamma',0.97, ...        % S1 geometric contraction ratio
  's1AfterCoal',true, ...    % S1: start contracting only once N>=2 first seen
  's2Levels',[0.05 0.02 0.01 0.005], ...   % S2 ladder
  's2Window',10,'s2Tol',5e-3, ...          % S2 stall detector window and tolerance
  's2Signal','beta', ...     % WHAT the S2 stall detector watches:
  ...                        % 'beta' legacy -- the OBJECTIVE bound.  Its
  ...                        %        improvement history is a property of the
  ...                        %        optimization path, so the descent time is
  ...                        %        mesh-dependent (k = 79/92/130).
  ...                        % 'drms' the DESIGN update ||drho||_2/sqrt(NE):
  ...                        %        descend when the update has stopped
  ...                        %        DECREASING at the present move scale, i.e.
  ...                        %        when that scale has ceased to be useful.
  ...                        %        Differential, so the mesh-dependent
  ...                        %        baseline cancels.  Same W and tol.
  ...                        %        See audit_s2_design_continuation/.
  's3Lo',0.30,'s3Hi',0.70,'s3Down',0.7,'s3Up',1.1, ...  % S3 gain-ratio band
  'offDiag',true, ...        % true = full (25d) coupling  [BASELINE]
  'innerSolver','mma', ...   % 'mma' = baseline; 'lp' = Krog & Olhoff route (implies eq.22)
  'mmaVariant','published', ...  % 'published' Svanberg Sept-2007 constants | 'asfound'
  ...                            % local lineage copy (see mma_published/README.md)
  'innerVar','drho', ...     % 'drho' = increment coords, MMA state reset per outer
  ...                        % iteration, hard move box (legacy reconstruction);
  ...                        % 'rho'  = design coords, MMA asymptotes/history persist
  ...                        % across outer iterations, move may be inf (innerLoopRho)
  'filterMode','diag', ...   % 'diag' | 'all' | 'none'
  'maxInner',300,'tolInner',1e-2,'minInner',5, ...  % tolInner is a FRACTION of move
  'maxOuter',200,'tolOuter',1e-3, ...
  'outerNorm','l2', ...      % Fig. 1 tests ||drho|| < eps: 'l2' (paper's printed
  ...                        % norm) | 'max' (legacy reconstruction)
  'outerGuard','none', ...   % guard on WHEN the outer test may be evaluated.
  ...                        % 'none'         legacy: test every iteration.
  ...                        % 'settledmove'  test only when the move limit is
  ...                        %                unchanged from the previous outer
  ...                        %                iteration.  Under a move ladder
  ...                        %                ||drho||_inf <= mv_k, so on an
  ...                        %                iteration where mv changed the
  ...                        %                measured step reports the SCHEDULE,
  ...                        %                not the design, and Fig. 1's
  ...                        %                criterion is uninterpretable there.
  ...                        %                See audit_termination_mesh_admission/.
  ... % ---- numerics ----
  'solver','eigs','threads',1,'verbose',true);
end
