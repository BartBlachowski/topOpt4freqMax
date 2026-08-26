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
  'move',0.05, ...           % move limit on drho
  'offDiag',true, ...        % true = full (25d) coupling  [BASELINE]
  'innerSolver','mma', ...   % 'mma' = baseline; 'lp' = Krog & Olhoff route (implies eq.22)
  'filterMode','diag', ...   % 'diag' | 'all' | 'none'
  'maxInner',300,'tolInner',1e-2,'minInner',5, ...  % tolInner is a FRACTION of move
  'maxOuter',200,'tolOuter',1e-3, ...
  ... % ---- numerics ----
  'solver','eigs','threads',1,'verbose',true);
end
